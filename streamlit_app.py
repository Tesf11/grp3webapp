# streamlit_app.py
import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import joblib

# Friend's UI for their sklearn models
from app.ui.ui_StoragePredictor import render_model1_ui

# YOUR adapter (you said HFAdapter is yours)
from app.hf_loader import HFAdapter

# Extra UIs
from app.ui.ui_entries import render_entries_page
from app.ui.ui_image_ranker import render_image_ranker_ui
# 🔗 GenAI sub-tab for Storage Predictor
from app.ui.ui_StoragePredictorGenai import render_storage_predictor_genai  # <- make sure this exists
from app.ui.ui_DisposalPredictor import render_disposal_predictor_ui

from datetime import date

# -----------------------------
# Discover models in app/models
# -----------------------------
MODELS_DIR = Path("app/models")

# Don’t auto-tab these feature folders (they already have their own dedicated pages)
EXCLUDE_MODEL_KEYS = {"prodcat_model", "image_ranker", "storage_predictor", "disposal_predictor"}

def discover_models():
    """Return a dict of available models in app/models.
    - sklearn: *.pkl (+ optional sidecar <name>.json) in root or subfolders
    - huggingface: folders containing config.json + (model.safetensors|pytorch_model.bin)
    """
    info = {}

    # 1) sklearn .pkl in root
    for p in MODELS_DIR.glob("*.pkl"):
        base = p.stem
        if base in EXCLUDE_MODEL_KEYS:
            continue
        mapping = MODELS_DIR / f"{base}.json"
        info[base] = {
            "type": "sklearn",
            "model_path": str(p),
            "mapping_path": str(mapping) if mapping.exists() else None,
        }

    # 2) folders: HF or sklearn-in-folder
    for d in MODELS_DIR.iterdir():
        if not d.is_dir():
            continue
        if d.name in EXCLUDE_MODEL_KEYS:
            continue

        # HF folder
        has_cfg = (d / "config.json").exists()
        has_weights = (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()
        if has_cfg and has_weights:
            base = d.name
            mapping = d / "mapping.json"
            info[base] = {
                "type": "hf",
                "model_path": str(d),
                "mapping_path": str(mapping) if mapping.exists() else None,
            }
            continue

        # sklearn-in-folder
        pkl_candidates = list(d.glob("*.pkl"))
        if pkl_candidates:
            base = d.name
            pkl_path = pkl_candidates[0]
            mapping = d / (pkl_path.stem + ".json")
            if not mapping.exists():
                json_candidates = list(d.glob("*.json"))
                mapping = json_candidates[0] if json_candidates else None
            info[base] = {
                "type": "sklearn",
                "model_path": str(pkl_path),
                "mapping_path": str(mapping) if (mapping and mapping.exists()) else None,
            }

    return info

# -----------------------------
# Helper: load Storage Predictor
# -----------------------------
def load_storage_predictor():
    """Load model + mapping from app/models/storage_predictor/*"""
    d = MODELS_DIR / "storage_predictor"
    if not d.exists():
        return None, None  # folder not present
    # find pkl
    pkl_candidates = list(d.glob("*.pkl"))
    if not pkl_candidates:
        return None, None
    pkl_path = pkl_candidates[0]
    model_obj = joblib.load(pkl_path)

    # mapping: prefer <pklstem>.json, else any *.json (e.g., mapping_data.json)
    mapping = d / (pkl_path.stem + ".json")
    if not mapping.exists():
        json_candidates = list(d.glob("*.json"))
        mapping = json_candidates[0] if json_candidates else None

    mapping_data = None
    if mapping and mapping.exists():
        with open(mapping, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)

    return model_obj, mapping_data

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="ANS Import Export", layout="wide")
models_info = discover_models()

# Build tabs:
# Home + auto-detected ML models (EXCLUDING prodcat_model/image_ranker/storage_predictor)
# + hard-coded Storage Predictor + Image Ranker + Product Category
model_keys = list(models_info.keys())
tab_labels = (
    ["🏠 Home"]
    + model_keys
    + ["🗄️ Storage Predictor", "🗑 Disposal Predictor", "🖼 Image Ranker", "📦 Product Category Predictor"]
)
tabs = st.tabs(tab_labels)

# -----------------------------
# Home
# -----------------------------
with tabs[0]:
    st.title("ANS Import Export Web UI Interface")
    st.write("""
This UI supports:
- **scikit-learn** models (root or folders under `app/models/`)
- **Hugging Face** folder models
- Dedicated tabs for **Storage Predictor**, **Disposal Predictor**, **Image Ranker**, and **Product Category**.
""")
    if not models_info:
        st.info("No models detected yet. Place your files under `app/models/`.")

# -----------------------------
# Auto-detected model tabs
# -----------------------------
for idx, model_name in enumerate(model_keys, start=1):
    with tabs[idx]:
        st.header(f"Model: {model_name}")
        meta = models_info[model_name]

        # Load the model (either sklearn or HF adapted to behave like sklearn)
        if meta["type"] == "sklearn":
            loaded_obj = joblib.load(meta["model_path"])
            mapping_data = None
            if meta["mapping_path"]:
                with open(meta["mapping_path"], "r", encoding="utf-8") as f:
                    mapping_data = json.load(f)
            model_for_ui = loaded_obj
        else:
            model_for_ui = HFAdapter(meta["model_path"], max_length=128)
            mapping_data = None
            if meta["mapping_path"]:
                with open(meta["mapping_path"], "r", encoding="utf-8") as f:
                    mapping_data = json.load(f)

        # Two subtabs per detected model
        sub_tabs = st.tabs(["🧠 Standard Prediction", "🤖 GenAI Model"])
        with sub_tabs[0]:
            render_model1_ui(model_for_ui, model_name, mapping_data=mapping_data, debug=False)
        with sub_tabs[1]:
            st.info("GenAI is available only under the 🗄️ Storage Predictor tab.")

# -----------------------------
# 🗄️ Storage Predictor (hard-coded)
# -----------------------------
storage_tab_index = 1 + len(model_keys)  # after Home + auto models
with tabs[storage_tab_index]:
    st.header("🗄️ Storage Predictor")
    model_for_ui, mapping_data = load_storage_predictor()
    if model_for_ui is None:
        st.error("Folder `app/models/storage_predictor/` wasn’t found or has no `.pkl`.")
    else:
        sub_tabs = st.tabs(["🧠 Standard Prediction", "🧪 Generate Sample Ideas (Gemini)"])
        with sub_tabs[0]:
            # Use the shared UI with your model + mapping
            render_model1_ui(model_for_ui, "storage_predictor", mapping_data=mapping_data, debug=False)
        with sub_tabs[1]:
            # Your GenAI Streamlit UI (lives in app/ui/ui_StoragePredictorGenai.py)
            render_storage_predictor_genai()

# -----------------------------
# 🗑 Disposal Predictor (new)
# -----------------------------
disposal_tab_index = 1 + len(model_keys) + 1  # immediately after Storage Predictor
with tabs[disposal_tab_index]:
    render_disposal_predictor_ui()


# -----------------------------
# 🖼 Image Ranker tab (kept)
# -----------------------------
with tabs[-2]:
    st.header("🖼 Image Ranker")
    st.caption("Upload and rank images using the trained Image Ranker model.")
    render_image_ranker_ui()

# -----------------------------
# 📦 Product Category tab (kept)
# -----------------------------
with tabs[-1]:
    render_entries_page()
