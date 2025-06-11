# Step 1: Install dependencies
!pip install datasets==3.5.0 transformers==4.51.2 torch==2.6.0 pandas==2.2.2 scikit-learn==1.6.1 --no-cache-dir

# Verify GPU
import torch
print("GPU Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# Step 2: Download CSVs
!mkdir -p data/full_dataset
!wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
!wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv
!wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv

# Step 3: Process data
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Load CSVs
dfs = []
for i in range(1, 4):
    df = pd.read_csv(f"data/full_dataset/goemotions_{i}.csv")
    dfs.append(df)
full_df = pd.concat(dfs, ignore_index=True)

# Define emotion columns
emotion_columns = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Aggregate by comment
def aggregate_labels(group):
    labels = [1 if group[emotion].sum() > 0 else 0 for emotion in emotion_columns]
    return pd.Series({
        "text": group["text"].iloc[0],
        "id": group.name,  # Use group.name for id
        "labels": labels
    })

aggregated_df = full_df.groupby("id").apply(aggregate_labels)

# Create splits
train_df, temp_df = train_test_split(aggregated_df, test_size=10853, random_state=42)
validation_df, test_df = train_test_split(temp_df, test_size=5427, random_state=42)

# Convert to DatasetDict
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(validation_df),
    "test": Dataset.from_pandas(test_df)
})

# Step 4: Filter emotions
target_emotions = {
    "anger": 2, "sadness": 23, "joy": 17, "disgust": 11, "fear": 14,
    "surprise": 25, "gratitude": 15, "remorse": 22, "curiosity": 7, "neutral": 27
}

def filter_emotions(examples):
    new_labels = []
    for labels in examples["labels"]:
        filtered = [labels[target_emotions[emotion]] for emotion in target_emotions]
        new_labels.append(filtered)
    return {"text": examples["text"], "labels": new_labels, "id": examples["id"]}

filtered_dataset = dataset.map(filter_emotions, batched=True)

# Step 5: Preprocess
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    return tokenized

def format_labels(examples):
    # Convert labels to float tensors
    return {"labels": [[torch.tensor(label, dtype=torch.float32) for label in labels] for labels in examples["labels"]]}

tokenized_dataset = filtered_dataset.map(preprocess_function, batched=True)
tokenized_dataset = tokenized_dataset.map(format_labels, batched=True)
# Convert to torch tensors
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Step 6: Create DataLoaders
from torch.utils.data import DataLoader

def create_dataloader(dataset, batch_size=16, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,  # Reduced num_workers
        pin_memory=True
    )

train_dataloader = create_dataloader(tokenized_dataset["train"], batch_size=16, shuffle=True)
val_dataloader = create_dataloader(tokenized_dataset["validation"], batch_size=16)
test_dataloader = create_dataloader(tokenized_dataset["test"], batch_size=16)

# Step 7: Model
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=10, problem_type="multi_label_classification"
)
model.to("cuda")

# Step 8: Optimizer and Loss
import torch.optim as optim
import torch.nn as nn

optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
criterion = nn.BCEWithLogitsLoss()  # For multi-label classification

# Step 9: Metrics
from sklearn.metrics import f1_score
import numpy as np

def compute_f1(predictions, labels):
    preds = (predictions > 0).astype(int)
    return f1_score(labels, preds, average="micro")

# Step 10: Training Loop
def train_epoch(model, dataloader, optimizer, criterion, device, epoch, num_epochs):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_idx, batch in enumerate(dataloader):  # Add batch_idx for printing
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].type(torch.float32).to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        if batch_idx % 100 == 0:  # Print every 100 steps
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    f1 = compute_f1(all_preds, all_labels)
    return total_loss / len(dataloader), f1


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].type(torch.float32).to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            total_loss += loss.item()
            all_preds.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    f1 = compute_f1(all_preds, all_labels)
    return total_loss / len(dataloader), f1

# Step 11: Training
device = torch.device("cuda")
num_epochs = 3  # Reduced epochs for faster testing
best_f1 = 0
best_model_state = None

for epoch in range(num_epochs):
    train_loss, train_f1 = train_epoch(model, train_dataloader, optimizer, criterion, device, epoch, num_epochs)
    val_loss, val_f1 = evaluate(model, val_dataloader, criterion, device)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_model_state = model.state_dict()

# Load best model
model.load_state_dict(best_model_state)

# Step 12: Evaluate on Test Set
test_loss, test_f1 = evaluate(model, test_dataloader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}")

# Step 13: Save Model
model.save_pretrained("./emotion_model_10")
tokenizer.save_pretrained("./emotion_model_10")
!zip -r emotion_model_10.zip ./emotion_model_10