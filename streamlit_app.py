import streamlit as st
import pandas as pd
import joblib
import json
import os
from ui_model1 import render_model1_ui  # Import the reusable UI function

MODELS_DIR = "app/models"

# --- Auto-detect model files and possible mapping files ---
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
json_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".json")]

models_info = {}
for model_file in model_files:
    base_name = os.path.splitext(model_file)[0]
    mapping_file = f"{base_name}.json"
    models_info[base_name] = {
        "model_path": os.path.join(MODELS_DIR, model_file),
        "mapping_path": os.path.join(MODELS_DIR, mapping_file) if mapping_file in json_files else None
    }

# --- Main UI ---
st.set_page_config(page_title="ANS Import Export", layout="wide")

# Tabs: Home + one per model
tab_labels = ["🏠 Home"] + list(models_info.keys())
tabs = st.tabs(tab_labels)

# --- Home Tab ---
with tabs[0]:
    st.title("ANS Import Export Web UI Interface")
    st.write("""
    Welcome to the ANS Import Export Web UI Interface.

    - **Model Tabs**: Each model has 2 sub-tabs:
        1. Standard Prediction
        2. GenAI Model
    """)

# --- Model Tabs ---
if not models_info:
    with tabs[0]:
        st.error("No model files found in app/models/")
else:
    for tab_index, model_name in enumerate(models_info.keys(), start=1):
        with tabs[tab_index]:
            st.header(f"Model: {model_name}")

            # Load the model or package
            loaded_obj = joblib.load(models_info[model_name]["model_path"])

            # Determine if it's a full package or model-only
            if isinstance(loaded_obj, dict) and "model" in loaded_obj:
                # Full package format
                model_package = loaded_obj
                mapping_data = None
            else:
                # Model-only format with separate JSON
                model_package = loaded_obj
                mapping_data = None
                if models_info[model_name]["mapping_path"]:
                    with open(models_info[model_name]["mapping_path"], "r") as f:
                        mapping_data = json.load(f)

            # Sub-tabs
            sub_tabs = st.tabs([
                "🧠 Standard Prediction",
                "🤖 GenAI Model"
            ])

            # --- Standard Prediction ---
            with sub_tabs[0]:
                render_model1_ui(
                    model_package if isinstance(model_package, dict) else model_package,
                    model_name,
                    mapping_data=mapping_data,
                    debug=False
                )

            # --- GenAI Model Placeholder ---
            with sub_tabs[1]:
                st.subheader(f"GenAI Model for {model_name}")
                st.write("🚧 GenAI integration will be added here.")
