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
    
# --- UPDATE (partial) ---
@prodcat_api.put("/api/update/<int:row_id>")
def api_update_entry(row_id: int):
    """Partial update. Accepts JSON with any subset of columns.
       UI may send 'type' for the 'category' DB column; we map it here.
    """
    payload = request.get_json(force=True, silent=True) or {}

    # Map UI field 'type' -> DB field 'category'
    if "type" in payload and "category" not in payload:
        payload["category"] = payload.pop("type")

    # Whitelist updatable fields from the model
    # (don’t allow id or server-managed timestamps to be written)
    allowed = {c.name for c in Entry.__table__.columns} - {"id", "created_at"}
    updates = {k: v for k, v in payload.items() if k in allowed}

    if not updates:
        return jsonify({"error": "No valid fields to update."}), 400

    with get_session() as s:
        obj = s.get(Entry, row_id)
        if not obj:
            return jsonify({"error": f"id {row_id} not found"}), 404

        for k, v in updates.items():
            setattr(obj, k, v)
        s.add(obj)           # optional; flushed on context exit
        s.flush()            # ensure obj has latest values

        return jsonify({"ok": True, "id": obj.id, "row": _serialize(obj)}), 200


# --- DELETE ---
@prodcat_api.delete("/api/delete/<int:row_id>")
def api_delete_entry(row_id: int):
    with get_session() as s:
        obj = s.get(Entry, row_id)
        if not obj:
            return jsonify({"error": f"id {row_id} not found"}), 404
        s.delete(obj)
        # commit happens on context exit
        return jsonify({"ok": True, "id": row_id}), 200

