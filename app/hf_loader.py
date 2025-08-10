# app/hf_loader.py
import json, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd  # <-- new

class HFAdapter:
    def __init__(self, model_dir: str, max_length: int = 128):
        p = Path(model_dir)
        with open(p / "labels.json", "r", encoding="utf-8") as f:
            self.labels = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(p, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(p, local_files_only=True)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length

    def _normalize_texts(self, data) -> list[str]:
        """Accept str, list[str], pd.Series, or pd.DataFrame with 'description' and return clean list[str]."""
        if isinstance(data, str):
            texts = [data.strip()]
        elif isinstance(data, (list, tuple)):
            texts = [str(x).strip() for x in data]
        elif isinstance(data, pd.Series):
            texts = data.astype("string").fillna("").map(lambda x: x.strip()).tolist()
        elif isinstance(data, pd.DataFrame):
            if "description" not in data.columns:
                raise ValueError("HFAdapter.predict expected a DataFrame with a 'description' column.")
            texts = (
                data["description"]
                .astype("string")
                .fillna("")
                .map(lambda x: x.strip())
                .tolist()
            )
        else:
            # Anything else, just coerce to str
            texts = [str(data).strip()]

        # drop empties
        texts = [t for t in texts if t]
        if not texts:
            raise ValueError("No non-empty text to predict.")
        return texts

    @torch.inference_mode()
    def predict(self, data):
        titles = self._normalize_texts(data)
        enc = self.tokenizer(titles, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        idxs = torch.argmax(logits, dim=-1).cpu().tolist()
        return [self.labels[i] for i in idxs]

    @torch.inference_mode()
    def topk(self, data, k=3):
        import torch.nn.functional as F
        titles = self._normalize_texts(data)
        enc = self.tokenizer(titles, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        probs = F.softmax(self.model(**enc).logits, dim=-1).cpu()
        out = []
        for row in probs:
            kk = min(k, row.shape[-1])
            pvals, idxs = torch.topk(row, k=kk)
            out.append([(self.labels[i], float(p)) for i, p in zip(idxs.tolist(), pvals.tolist())])
        return out
