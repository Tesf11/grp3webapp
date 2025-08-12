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

import requests

from app.ui.ui_sample_form import render_sample_form

from datetime import date





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
TARGET_MODEL = "product_classifier"  # <-- set this to your pkl's base filename

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

        # Two subtabs per detected model
        sub_tabs = st.tabs(["🧠 Standard Prediction", "🤖 GenAI Model"])

        with sub_tabs[0]:
            render_model1_ui(
                model_for_ui,
                model_name,
                mapping_data=mapping_data,
                debug=False
            )

        with sub_tabs[1]:
            if model_name == TARGET_MODEL:
                st.subheader(f"GenAI Model for {model_name} (Gemini 2.5)")
                st.caption("Generates possible products, industries, and features. Does not predict bins.")

                API_GEN_URL = os.getenv("GEN_API_URL", "http://127.0.0.1:5000/api/genideas")

                gen_desc = st.text_area("Enter sample description", height=120, key=f"gen_desc_{model_name}")
                extra_tags = st.text_input("Optional extra tags (comma-separated)", key=f"gen_tags_{model_name}")
                temp = st.slider("Creativity (temperature)", 0.0, 1.0, 0.7, 0.1, key=f"gen_temp_{model_name}")
                model_choice = st.selectbox(
                    "Model",
                    ["gemini-2.5-flash", "gemini-2.5-pro"],
                    index=0,
                    key=f"gen_model_{model_name}"
                )

                if st.button("Generate ideas", key=f"gen_btn_{model_name}"):
                    if not gen_desc.strip():
                        st.warning("Please enter a description.")
                    else:
                        import requests, json
                        payload = {
                            "description": gen_desc.strip(),
                            "tags": [t.strip() for t in extra_tags.split(",") if t.strip()],
                            "temperature": temp,
                            "model_name": model_choice,
                        }
                        try:
                            r = requests.post(API_GEN_URL, json=payload, timeout=45)
                            r.raise_for_status()
                            data = r.json()
                            if "error" in data:
                                st.error(data["error"])
                            else:
                                ideas = data.get("ideas", {})
                                st.markdown("#### Possible Products")
                                st.write(ideas.get("possible_products", []))
                                st.markdown("#### Industries")
                                st.write(ideas.get("industries", []))
                                st.markdown("#### Features")
                                st.write(ideas.get("features", []))

                                st.download_button(
                                    "Download ideas (JSON)",
                                    data=json.dumps(ideas, ensure_ascii=False, indent=2),
                                    file_name="genai_ideas.json",
                                    mime="application/json",
                                )
                        except requests.RequestException as e:
                            st.error(f"Request failed: {e}")
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                st.info("GenAI is available only under the Product Classifier tab.")



# -----------------------------
# YOUR dedicated Product Category tab (uses your HFAdapter directly)
# -----------------------------
with tabs[-1]:
    st.header("📦 Product Title → Category (via Flask backend)")
    st.caption("Enter a product title; prediction is done by Flask backend's HF model.")

    # Backend API URL
    API_URL = os.getenv("API_URL", "http://127.0.0.1:5000/api/predict")

    render_sample_form(API_URL)
