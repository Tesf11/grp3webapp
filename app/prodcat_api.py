# app/prodcat_api.py
from flask import Blueprint, request, jsonify, current_app
from app.hf_loader import HFAdapter
import os
from app.db import get_session 
from app.models import Entry


prodcat_api = Blueprint("prodcat_api", __name__)

# Load your model once at blueprint registration
hf_model = None

def _serialize(obj):
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}

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

@prodcat_api.post("/api/save/entries")
def api_save_entry():
    data = request.get_json(force=True, silent=True) or {}
    required = ["sample_type","company","product_title","status","category"]
    missing = [k for k in required if not str(data.get(k) or "").strip()]
    if missing:
        return jsonify({"error": f"Missing required: {', '.join(missing)}"}), 400

    with get_session() as s:
        obj = Entry(**{k: v for k, v in data.items() if k in Entry.__table__.columns})
        s.add(obj)
        s.commit()
        s.refresh(obj)
        return jsonify({"ok": True, "id": obj.id}), 200

@prodcat_api.get("/api/list/entries")
def api_list_entries():
    limit = int(request.args.get("limit", 50))
    with get_session() as s:
        rows = s.query(Entry).order_by(Entry.id.desc()).limit(limit).all()
        def ser(r): return {c.name: getattr(r, c.name) for c in r.__table__.columns}
        return jsonify([ser(r) for r in rows]), 200
