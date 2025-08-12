# app/prodcat_api.py
from flask import Blueprint, request, jsonify, current_app
from app.hf_loader import HFAdapter
import os

prodcat_api = Blueprint("prodcat_api", __name__)

# Load your model once at blueprint registration
hf_model = None

@prodcat_api.record_once
def load_model(setup_state):
    global hf_model
    model_dir = os.getenv("MODEL_DIR", "app/models/prodcat_model")
    hf_model = HFAdapter(model_dir, max_length=128)

@prodcat_api.post("/api/predict")
def predict():
    data = request.get_json(force=True, silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "title is required"}), 400
    category = hf_model.predict(title)[0]
    return jsonify({"title": title, "category": category})

@prodcat_api.get("/api/health")
def health():
    return {"ok": True}
