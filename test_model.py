# test_model.py
import os, json, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.getenv("MODEL_DIR", "app/models/prodcat_model")

# 1) labels
with open(os.path.join(MODEL_DIR, "labels.json"), "r", encoding="utf-8") as f:
    LABELS = json.load(f)

# 2) load locally (no internet)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()

def predict_one(title: str) -> str:
    enc = tokenizer(title, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.inference_mode():
        logits = model(**enc).logits
        idx = int(torch.argmax(logits, dim=-1).item())
    return LABELS[idx]

if __name__ == "__main__":
    samples = [
        "Apple iPhone 14 Pro Max 256GB",
        "Nike Air Zoom Pegasus running shoes",
        "Stainless steel water bottle 1L",
    ]
    for s in samples:
        print(s, "->", predict_one(s))
