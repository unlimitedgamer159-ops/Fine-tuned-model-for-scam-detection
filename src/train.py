import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
import evaluate as hf_evaluate

MODEL_NAME = "distilbert-base-uncased"
MODEL_OUTPUT = "models/scam-detector"
DATA_PATH = "data/dataset.csv"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5


def load_and_split(data_path: str):
    df = pd.read_csv(data_path)
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return (
        Dataset.from_pandas(train_df.reset_index(drop=True)),
        Dataset.from_pandas(val_df.reset_index(drop=True)),
        Dataset.from_pandas(test_df.reset_index(drop=True)),
    )


def tokenize_dataset(tokenizer, dataset: Dataset) -> Dataset:
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)
    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def compute_metrics(eval_pred):
    metric_acc = hf_evaluate.load("accuracy")
    metric_f1 = hf_evaluate.load("f1")

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
    f1 = metric_f1.compute(predictions=preds, references=labels, average="binary")["f1"]
    return {"accuracy": acc, "f1": f1}


def train():
    os.makedirs(MODEL_OUTPUT, exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading dataset...")
    train_ds, val_ds, test_ds = load_and_split(DATA_PATH)

    print("Tokenizing...")
    train_ds = tokenize_dataset(tokenizer, train_ds)
    val_ds = tokenize_dataset(tokenizer, val_ds)
    test_ds = tokenize_dataset(tokenizer, test_ds)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")

    train_ds.set_format("torch")
    val_ds.set_format("torch")
    test_ds.set_format("torch")

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "legit", 1: "scam"},
        label2id={"legit": 0, "scam": 1},
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    training_args = TrainingArguments(
        output_dir="models/checkpoints",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="models/logs",
        logging_steps=50,
        fp16=False,
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving model to {MODEL_OUTPUT}")
    trainer.save_model(MODEL_OUTPUT)
    tokenizer.save_pretrained(MODEL_OUTPUT)

    print("\nFinal evaluation on test set:")
    results = trainer.evaluate(test_ds)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    print("\nTraining complete.")


if __name__ == "__main__":
    train()
