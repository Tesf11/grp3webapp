# streamlit_app.py
import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import joblib

# Friend's UI for their sklearn models
from app.ui.ui_model1 import render_model1_ui

# YOUR adapter (you said HFAdapter is yours)
from app.hf_loader import HFAdapter


# -----------------------------
# Discover models in app/models
# -----------------------------
MODELS_DIR = Path("app/models")

def discover_models():
    """Return a dict of available models in app/models.
    - sklearn: *.pkl (+ optional sidecar <name>.json)
    - huggingface: folders containing config.json + (model.safetensors|pytorch_model.bin)
    """
    info = {}

    # 1) sklearn .pkl models
    for p in MODELS_DIR.glob("*.pkl"):
        base = p.stem
        mapping = MODELS_DIR / f"{base}.json"
        info[base] = {
            "type": "sklearn",
            "model_path": str(p),
            "mapping_path": str(mapping) if mapping.exists() else None,
        }

    # 2) Hugging Face folders
    for d in MODELS_DIR.iterdir():
        if d.is_dir():
            has_cfg = (d / "config.json").exists()
            has_weights = (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()
            if has_cfg and has_weights:
                base = d.name
                mapping = d / "mapping.json"  # optional if you keep extra mappings here
                info[base] = {
                    "type": "hf",
                    "model_path": str(d),
                    "mapping_path": str(mapping) if mapping.exists() else None,
                }

    return info


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="ANS Import Export", layout="wide")
models_info = discover_models()

# Tabs: Home + one per detected model + YOUR product-category tab
tab_labels = ["🏠 Home"] + list(models_info.keys()) + ["📦 Product Category Predictor"]
tabs = st.tabs(tab_labels)

# Home
with tabs[0]:
    st.title("ANS Import Export Web UI Interface")
    st.write("""
This UI supports:
- **scikit-learn `.pkl`** models (optionally with `<name>.json` mapping)
- **Hugging Face** models exported as a folder (config/tokenizer/weights)
- **Your** Product Title → Category predictor (HFAdapter) on its own tab
""")
    if not models_info:
        st.info("No models detected yet. Place your files under `app/models/`.")

# Detected model tabs (friend’s + any HF folders they want to use via the shared UI)
for tab_index, model_name in enumerate(models_info.keys(), start=1):
    with tabs[tab_index]:
        st.header(f"Model: {model_name}")
        meta = models_info[model_name]

        # Load the model (either sklearn or HF adapted to behave like sklearn)
        if meta["type"] == "sklearn":
            loaded_obj = joblib.load(meta["model_path"])
            mapping_data = None
            if meta["mapping_path"]:
                with open(meta["mapping_path"], "r", encoding="utf-8") as f:
                    mapping_data = json.load(f)
            model_for_ui = loaded_obj  # could be dict package or bare estimator

        else:  # Hugging Face (used via the shared UI pattern if desired)
            model_for_ui = HFAdapter(meta["model_path"], max_length=128)
            mapping_data = None
            if meta["mapping_path"]:
                with open(meta["mapping_path"], "r", encoding="utf-8") as f:
                    mapping_data = json.load(f)

        # Friend's two-subtab layout (Standard Prediction + placeholder for GenAI)
        sub_tabs = st.tabs(["🧠 Standard Prediction", "🤖 GenAI Model"])

        with sub_tabs[0]:
            render_model1_ui(
                model_for_ui,
                model_name,
                mapping_data=mapping_data,
                debug=False
            )

        with sub_tabs[1]:
            st.subheader(f"GenAI Model for {model_name}")
            st.write("🚧 GenAI integration will be added here.")


# -----------------------------
# YOUR dedicated Product Category tab (uses your HFAdapter directly)
# -----------------------------
with tabs[-1]:
    st.header("📦 Product Title → Category (Your HFAdapter)")
    st.caption("Enter a product title; prediction uses the Hugging Face model via your HFAdapter.")

    # Change this path if your HF model folder name is different
    HF_MODEL_DIR = os.getenv("MODEL_DIR", "app/models/prodcat_model")

    @st.cache_resource
    def get_my_adapter(model_dir: str):
        return HFAdapter(model_dir, max_length=128)

    # Attempt to load once, show a clear error if folder missing
    try:
        my_model = get_my_adapter(HF_MODEL_DIR)
        model_ready = True
    except Exception as e:
        model_ready = False
        st.error(f"Failed to load HF model from `{HF_MODEL_DIR}`.\n{e}")

    with st.form("my_prodcat_form", clear_on_submit=False):
        title = st.text_input(
            "Product title",
            placeholder="e.g., Apple iPhone 14 Pro Max 256GB",
            max_chars=256,
        )
        submitted = st.form_submit_button("Predict")

    if submitted:
        if not model_ready:
            st.warning("Model not loaded. Please fix the model path/files and reload the app.")
        else:
            t = (title or "").strip()
            if not t:
                st.warning("Please enter a product title.")
            else:
                # Your HFAdapter.predict expects a DataFrame with a 'description' column
                df = pd.DataFrame([{"description": t}])
                try:
                    category = my_model.predict(df)[0]
                    st.success(f"**Predicted category:** {category}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
