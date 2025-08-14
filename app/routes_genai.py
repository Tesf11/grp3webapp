import json
from flask import Blueprint, request, jsonify
from app.ui.ui_StoragePredictorGenai import generate_ideas

genai_bp = Blueprint("genai", __name__)

@genai_bp.route("/api/genideas", methods=["POST"])
def gen_ideas():
    data = request.get_json(force=True) or {}
    description = (data.get("description") or "").strip()
    extra_tags = data.get("tags") or []
    temperature = float(data.get("temperature", 0.7))
    model_name = data.get("model_name") or None  # optional override

    if not description:
        return jsonify({"error": "description is required"}), 400
    try:
        ideas = generate_ideas(
            description,
            extra_tags=extra_tags,
            temperature=temperature,
            model_name=model_name or None
        )
        return jsonify({"description": description, "ideas": ideas})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
