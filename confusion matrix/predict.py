"""
src/predict.py

Loads the fine-tuned model and runs inference on new inputs.
Can be used as a script or imported as a module.

Usage:
    python src/predict.py "Congratulations! You won Rs 50,000. Call now."
    python src/predict.py "https://paypal-secure-update.com/verify"
    python src/predict.py  # enters interactive mode
"""

import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/scam-detector"
MAX_LEN = 256

_tokenizer = None
_model = None
_device = None


def load_model():
    global _tokenizer, _model, _device
    if _model is not None:
        return  # already loaded

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run src/train.py first."
        )

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(_device)
    _model.eval()


def predict(text: str) -> dict:
    """
    Returns:
        {
            "label": "scam" | "legit",
            "confidence": float (0-1),
            "scam_probability": float,
            "legit_probability": float,
        }
    """
    load_model()

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    scam_prob = probs[1].item()
    legit_prob = probs[0].item()
    label = "scam" if scam_prob > 0.5 else "legit"
    confidence = max(scam_prob, legit_prob)

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "scam_probability": round(scam_prob, 4),
        "legit_probability": round(legit_prob, 4),
    }


def print_result(text: str):
    result = predict(text)
    verdict = "🚨 SCAM" if result["label"] == "scam" else "✅ LEGIT"
    print(f"\nInput     : {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"Verdict   : {verdict}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"  scam={result['scam_probability']:.4f}  legit={result['legit_probability']:.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print_result(text)
    else:
        print("Scam Detector — Interactive Mode (type 'quit' to exit)\n")
        load_model()
        while True:
            try:
                text = input("Enter message or URL: ").strip()
                if text.lower() in ("quit", "exit", "q"):
                    break
                if text:
                    print_result(text)
            except (KeyboardInterrupt, EOFError):
                break
