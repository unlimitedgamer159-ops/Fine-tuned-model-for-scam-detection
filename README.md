# Scam Detector — Fine-tuned DistilBERT

Detects scam SMS messages, phishing URLs, and scam emails using a fine-tuned `distilbert-base-uncased` model.

**Trained on real public datasets** (UCI SMS Spam, SpamAssassin emails, PhiUSIIL phishing URLs) supplemented with a small synthetic portion for regional patterns.

---

## Project Structure

```
scam-detector/
├── data/
│   ├── prepare_dataset.py    # downloads real data + adds synthetic supplement
│   └── dataset.csv           # generated after running prepare_dataset.py
├── src/
│   ├── train.py              # fine-tuning pipeline
│   ├── evaluate.py           # precision, recall, F1, confusion matrix
│   └── predict.py            # inference on new text/URLs
├── models/
│   ├── scam-detector/        # saved model after training
│   └── confusion_matrix.png  # generated after evaluate.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup (GitHub Codespaces)

### 1. Clone and open in Codespaces
```bash
# Open your repo in GitHub Codespaces (2-core, 8GB is sufficient)
```

### 2. Create your `.env` file
```bash
cp .env.example .env
# Edit .env and paste your HuggingFace token:
# HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

Get your token at: https://huggingface.co/settings/tokens  
(Create a token with **read** access — write only needed if you push the model)

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Prepare the dataset
Downloads real datasets from HuggingFace Hub and builds `data/dataset.csv`.
```bash
python data/prepare_dataset.py
```
Expected output: ~6,000–12,000 labeled samples across SMS, email, and URL categories.

### Step 2 — Fine-tune the model
```bash
python src/train.py
```
- Trains for up to 4 epochs with early stopping
- Saves best model to `models/scam-detector/`
- Takes ~15–25 minutes on Codespaces CPU

### Step 3 — Evaluate
```bash
python src/evaluate.py
```
Prints per-class precision, recall, F1 and saves a confusion matrix PNG.

### Step 4 — Predict
```bash
# Single input
python src/predict.py "Congratulations! You won Rs 50,000. Call now to claim."
python src/predict.py "https://paypal-secure-login.com/verify-account"

# Interactive mode
python src/predict.py
```

---

## Expected Performance

| Metric    | Expected Range |
|-----------|---------------|
| Accuracy  | 96–98%        |
| F1 (scam) | 95–97%        |
| Precision | 94–97%        |
| Recall    | 95–98%        |

*Results vary slightly based on dataset availability at download time.*

---

## Data Sources

| Source | Type | HuggingFace ID |
|--------|------|----------------|
| UCI SMS Spam Collection | SMS | `ucirvine/sms_spam` |
| SpamAssassin | Email | `talby/spamassassin` |
| PhiUSIIL Phishing URL | URL | `pirocheto/phishing-url` |
| Synthetic (regional) | All | Built-in |

---

## Notes

- Model runs on CPU in Codespaces — no GPU required
- `fp16` training is auto-disabled on CPU
- To push model to HuggingFace Hub after training, add `trainer.push_to_hub("your-username/scam-detector")` to `train.py`
