"""
data/prepare_dataset.py

Downloads real public datasets from HuggingFace and supplements with
a small synthetic portion for underrepresented patterns.
Outputs: data/dataset.csv with columns: text, label (0=legit, 1=scam), category
"""

import os
import re
import random
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

random.seed(42)

# ──────────────────────────────────────────────
# 1. REAL DATA LOADERS
# ──────────────────────────────────────────────

def load_sms_data() -> pd.DataFrame:
    """UCI SMS Spam Collection via HuggingFace"""
    print("Loading SMS spam dataset...")
    ds = load_dataset("ucirvine/sms_spam", split="train", trust_remote_code=True)
    df = ds.to_pandas()
    # columns: label (ham/spam), sms
    df = df.rename(columns={"sms": "text"})
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["category"] = "sms"
    return df[["text", "label", "category"]]


def load_email_data() -> pd.DataFrame:
    """Spam email dataset via HuggingFace"""
    print("Loading email spam dataset...")
    ds = load_dataset("talby/spamassassin", split="train", trust_remote_code=True)
    df = ds.to_pandas()
    # columns: label (0=ham,1=spam), text
    df["label"] = df["label"].astype(int)
    df["category"] = "email"
    # truncate very long emails to 512 chars for training efficiency
    df["text"] = df["text"].astype(str).str.slice(0, 512)
    return df[["text", "label", "category"]]


def load_url_data() -> pd.DataFrame:
    """Phishing URL dataset via HuggingFace"""
    print("Loading phishing URL dataset...")
    ds = load_dataset("pirocheto/phishing-url", split="train", trust_remote_code=True)
    df = ds.to_pandas()
    # columns: url, status (phishing/legitimate)
    df = df.rename(columns={"url": "text"})
    df["label"] = df["status"].map({"phishing": 1, "legitimate": 0})
    df["category"] = "url"
    return df[["text", "label", "category"]]


# ──────────────────────────────────────────────
# 2. SYNTHETIC SUPPLEMENT (~10% of final data)
#    Only fills patterns underrepresented in
#    real datasets (e.g. regional scam styles)
# ──────────────────────────────────────────────

SYNTHETIC_SCAM = [
    # SMS scams
    ("Your KYC is expiring. Update now or your account will be blocked: http://sbi-kyc-update.xyz", "sms"),
    ("Congratulations! You won Rs 50,000 in lucky draw. Call 9XXXXXXXXX to claim.", "sms"),
    ("URGENT: Your Aadhaar is linked to illegal activity. Call officer at 011-XXXXXXXX immediately.", "sms"),
    ("Dear customer, your electricity connection will be cut tonight. Pay Rs 2000 now: bit.ly/payhere", "sms"),
    ("Your FASTag wallet is low. Recharge now to avoid fine: fastag-recharge.net/pay", "sms"),
    ("Free gift! You have been selected. Claim your iPhone 15 at: claim-prize.online/iphone", "sms"),
    # Phishing URLs
    ("http://paypal-securelogin.com/verify-account", "url"),
    ("https://amazon-order-confirm.net/invoice/8821", "url"),
    ("http://hdfc-netbanking-update.xyz/login", "url"),
    ("https://sbi.co.in.update-kyc.xyz/form", "url"),
    ("http://fedex-delivery-pending.com/track?id=889221", "url"),
    ("https://instagram-verify-account.net/confirm", "url"),
    # Scam emails
    ("Dear beneficiary, you have been selected to receive $1,500,000 from the UN compensation fund. Send your details to claim@un-funds.org", "email"),
    ("Your Netflix account has been suspended. Update your payment at: netflix-billing-update.com", "email"),
    ("IRS FINAL NOTICE: You owe $3,200 in back taxes. Avoid arrest by paying with gift cards to this number.", "email"),
    ("HSBC Security Alert: Unusual login detected. Verify your identity immediately at hsbc-secure-verify.com", "email"),
    ("You have a pending package from DHL. Pay Rs 150 customs fee at: dhl-delivery-india.net/pay", "email"),
]

SYNTHETIC_LEGIT = [
    ("Your OTP for SBI login is 482910. Valid for 5 minutes. Do not share with anyone.", "sms"),
    ("Your Amazon order #112-XXXXXXX has been shipped. Expected delivery: Thursday.", "sms"),
    ("Hi, are we still on for lunch tomorrow at 1pm?", "sms"),
    ("Your electricity bill of Rs 1,240 is due on 25th. Pay at: mahadiscom.in", "sms"),
    ("IRCTC: Your ticket PNR 8XXXXXXX is confirmed. Train departs at 06:45 from CSTM.", "sms"),
    ("https://www.google.com/search?q=python+tutorial", "url"),
    ("https://github.com/huggingface/transformers", "url"),
    ("https://www.irctc.co.in/nget/train-search", "url"),
    ("https://mail.google.com/mail/u/0/", "url"),
    ("Dear Team, please find attached the Q3 report for your review. Let me know if you have questions.", "email"),
    ("Your appointment at Apollo Hospital is confirmed for March 22 at 10:30 AM.", "email"),
    ("Receipt for your Swiggy order #SW992211. Total charged: Rs 340.", "email"),
]


def build_synthetic_df() -> pd.DataFrame:
    rows = []
    for text, category in SYNTHETIC_SCAM:
        rows.append({"text": text, "label": 1, "category": category})
    for text, category in SYNTHETIC_LEGIT:
        rows.append({"text": text, "label": 0, "category": category})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 3. COMBINE + BALANCE + SAVE
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text).strip()
    # remove null bytes and control chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def prepare(output_path: str = "data/dataset.csv", max_per_source: int = 2000):
    os.makedirs("data", exist_ok=True)

    dfs = []

    # Load real datasets with error handling per source
    for loader in [load_sms_data, load_email_data, load_url_data]:
        try:
            df = loader()
            # cap per source so no single dataset dominates
            scam = df[df["label"] == 1].sample(min(max_per_source, (df["label"]==1).sum()), random_state=42)
            legit = df[df["label"] == 0].sample(min(max_per_source, (df["label"]==0).sum()), random_state=42)
            dfs.append(pd.concat([scam, legit]))
            print(f"  → {loader.__name__}: {len(scam)} scam, {len(legit)} legit")
        except Exception as e:
            print(f"  ✗ {loader.__name__} failed: {e}")
            print("    Skipping this source and continuing...")

    # Add synthetic supplement
    syn_df = build_synthetic_df()
    dfs.append(syn_df)
    print(f"  → synthetic: {(syn_df['label']==1).sum()} scam, {(syn_df['label']==0).sum()} legit")

    if not dfs:
        raise RuntimeError("All data sources failed. Check your internet connection.")

    combined = pd.concat(dfs, ignore_index=True)
    combined["text"] = combined["text"].apply(clean_text)
    combined = combined[combined["text"].str.len() > 5]  # drop near-empty rows
    combined = combined.drop_duplicates(subset="text")
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    combined.to_csv(output_path, index=False)

    print(f"\n✅ Dataset saved to {output_path}")
    print(f"   Total samples : {len(combined)}")
    print(f"   Scam (1)      : {(combined['label']==1).sum()}")
    print(f"   Legit (0)     : {(combined['label']==0).sum()}")
    print(f"\n   By category:")
    print(combined.groupby(["category", "label"]).size().to_string())


if __name__ == "__main__":
    prepare()"""
data/prepare_dataset.py

Downloads real public datasets from HuggingFace and supplements with
a small synthetic portion for underrepresented patterns.
Outputs: data/dataset.csv with columns: text, label (0=legit, 1=scam), category
"""

import os
import re
import random
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

random.seed(42)

# ──────────────────────────────────────────────
# 1. REAL DATA LOADERS
# ──────────────────────────────────────────────

def load_sms_data() -> pd.DataFrame:
    """UCI SMS Spam Collection via HuggingFace"""
    print("Loading SMS spam dataset...")
    ds = load_dataset("ucirvine/sms_spam", split="train", trust_remote_code=True)
    df = ds.to_pandas()
    # columns: label (ham/spam), sms
    df = df.rename(columns={"sms": "text"})
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["category"] = "sms"
    return df[["text", "label", "category"]]


def load_email_data() -> pd.DataFrame:
    """Spam email dataset via HuggingFace"""
    print("Loading email spam dataset...")
    ds = load_dataset("talby/spamassassin", split="train", trust_remote_code=True)
    df = ds.to_pandas()
    # columns: label (0=ham,1=spam), text
    df["label"] = df["label"].astype(int)
    df["category"] = "email"
    # truncate very long emails to 512 chars for training efficiency
    df["text"] = df["text"].astype(str).str.slice(0, 512)
    return df[["text", "label", "category"]]


def load_url_data() -> pd.DataFrame:
    """Phishing URL dataset via HuggingFace"""
    print("Loading phishing URL dataset...")
    ds = load_dataset("pirocheto/phishing-url", split="train", trust_remote_code=True)
    df = ds.to_pandas()
    # columns: url, status (phishing/legitimate)
    df = df.rename(columns={"url": "text"})
    df["label"] = df["status"].map({"phishing": 1, "legitimate": 0})
    df["category"] = "url"
    return df[["text", "label", "category"]]


# ──────────────────────────────────────────────
# 2. SYNTHETIC SUPPLEMENT (~10% of final data)
#    Only fills patterns underrepresented in
#    real datasets (e.g. regional scam styles)
# ──────────────────────────────────────────────

SYNTHETIC_SCAM = [
    # SMS scams
    ("Your KYC is expiring. Update now or your account will be blocked: http://sbi-kyc-update.xyz", "sms"),
    ("Congratulations! You won Rs 50,000 in lucky draw. Call 9XXXXXXXXX to claim.", "sms"),
    ("URGENT: Your Aadhaar is linked to illegal activity. Call officer at 011-XXXXXXXX immediately.", "sms"),
    ("Dear customer, your electricity connection will be cut tonight. Pay Rs 2000 now: bit.ly/payhere", "sms"),
    ("Your FASTag wallet is low. Recharge now to avoid fine: fastag-recharge.net/pay", "sms"),
    ("Free gift! You have been selected. Claim your iPhone 15 at: claim-prize.online/iphone", "sms"),
    # Phishing URLs
    ("http://paypal-securelogin.com/verify-account", "url"),
    ("https://amazon-order-confirm.net/invoice/8821", "url"),
    ("http://hdfc-netbanking-update.xyz/login", "url"),
    ("https://sbi.co.in.update-kyc.xyz/form", "url"),
    ("http://fedex-delivery-pending.com/track?id=889221", "url"),
    ("https://instagram-verify-account.net/confirm", "url"),
    # Scam emails
    ("Dear beneficiary, you have been selected to receive $1,500,000 from the UN compensation fund. Send your details to claim@un-funds.org", "email"),
    ("Your Netflix account has been suspended. Update your payment at: netflix-billing-update.com", "email"),
    ("IRS FINAL NOTICE: You owe $3,200 in back taxes. Avoid arrest by paying with gift cards to this number.", "email"),
    ("HSBC Security Alert: Unusual login detected. Verify your identity immediately at hsbc-secure-verify.com", "email"),
    ("You have a pending package from DHL. Pay Rs 150 customs fee at: dhl-delivery-india.net/pay", "email"),
]

SYNTHETIC_LEGIT = [
    ("Your OTP for SBI login is 482910. Valid for 5 minutes. Do not share with anyone.", "sms"),
    ("Your Amazon order #112-XXXXXXX has been shipped. Expected delivery: Thursday.", "sms"),
    ("Hi, are we still on for lunch tomorrow at 1pm?", "sms"),
    ("Your electricity bill of Rs 1,240 is due on 25th. Pay at: mahadiscom.in", "sms"),
    ("IRCTC: Your ticket PNR 8XXXXXXX is confirmed. Train departs at 06:45 from CSTM.", "sms"),
    ("https://www.google.com/search?q=python+tutorial", "url"),
    ("https://github.com/huggingface/transformers", "url"),
    ("https://www.irctc.co.in/nget/train-search", "url"),
    ("https://mail.google.com/mail/u/0/", "url"),
    ("Dear Team, please find attached the Q3 report for your review. Let me know if you have questions.", "email"),
    ("Your appointment at Apollo Hospital is confirmed for March 22 at 10:30 AM.", "email"),
    ("Receipt for your Swiggy order #SW992211. Total charged: Rs 340.", "email"),
]


def build_synthetic_df() -> pd.DataFrame:
    rows = []
    for text, category in SYNTHETIC_SCAM:
        rows.append({"text": text, "label": 1, "category": category})
    for text, category in SYNTHETIC_LEGIT:
        rows.append({"text": text, "label": 0, "category": category})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 3. COMBINE + BALANCE + SAVE
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text).strip()
    # remove null bytes and control chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def prepare(output_path: str = "data/dataset.csv", max_per_source: int = 2000):
    os.makedirs("data", exist_ok=True)

    dfs = []

    # Load real datasets with error handling per source
    for loader in [load_sms_data, load_email_data, load_url_data]:
        try:
            df = loader()
            # cap per source so no single dataset dominates
            scam = df[df["label"] == 1].sample(min(max_per_source, (df["label"]==1).sum()), random_state=42)
            legit = df[df["label"] == 0].sample(min(max_per_source, (df["label"]==0).sum()), random_state=42)
            dfs.append(pd.concat([scam, legit]))
            print(f"  → {loader.__name__}: {len(scam)} scam, {len(legit)} legit")
        except Exception as e:
            print(f"  ✗ {loader.__name__} failed: {e}")
            print("    Skipping this source and continuing...")

    # Add synthetic supplement
    syn_df = build_synthetic_df()
    dfs.append(syn_df)
    print(f"  → synthetic: {(syn_df['label']==1).sum()} scam, {(syn_df['label']==0).sum()} legit")

    if not dfs:
        raise RuntimeError("All data sources failed. Check your internet connection.")

    combined = pd.concat(dfs, ignore_index=True)
    combined["text"] = combined["text"].apply(clean_text)
    combined = combined[combined["text"].str.len() > 5]  # drop near-empty rows
    combined = combined.drop_duplicates(subset="text")
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    combined.to_csv(output_path, index=False)

    print(f"\n✅ Dataset saved to {output_path}")
    print(f"   Total samples : {len(combined)}")
    print(f"   Scam (1)      : {(combined['label']==1).sum()}")
    print(f"   Legit (0)     : {(combined['label']==0).sum()}")
    print(f"\n   By category:")
    print(combined.groupby(["category", "label"]).size().to_string())


if __name__ == "__main__":
    prepare()
