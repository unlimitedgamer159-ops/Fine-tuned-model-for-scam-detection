import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset as TorchDataset

MODEL_PATH = "models/scam-detector"
DATA_PATH = "data/dataset.csv"
MAX_LEN = 256
BATCH_SIZE = 32


class TextDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt"
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def get_test_split():
    df = pd.read_csv(DATA_PATH)
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    _, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    _, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])
    return test_df


def run_evaluate():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Run src/train.py first.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print("Loading test split...")
    test_df = get_test_split()
    texts = test_df["text"].tolist()
    true_labels = test_df["label"].tolist()

    dataset = TextDataset(texts, true_labels, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    all_preds = []

    print("Running inference...")
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())

    print("\n" + "=" * 55)
    print("CLASSIFICATION REPORT")
    print("=" * 55)
    print(classification_report(true_labels, all_preds, target_names=["legit", "scam"], digits=4))

    cm = confusion_matrix(true_labels, all_preds)
    print("CONFUSION MATRIX")
    print("=" * 55)
    print(f"              Predicted legit  Predicted scam")
    print(f"Actual legit  {cm[0][0]:>14}  {cm[0][1]:>14}")
    print(f"Actual scam   {cm[1][0]:>14}  {cm[1][1]:>14}")

    os.makedirs("models", exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["legit", "scam"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    plt.title("Scam Detector — Confusion Matrix")
    plt.tight_layout()
    plot_path = "models/confusion_matrix.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nConfusion matrix saved to {plot_path}")


if __name__ == "__main__":
    run_evaluate()
