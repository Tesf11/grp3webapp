import streamlit as st
import pandas as pd

# --- SQLAlchemy imports ---
from sqlalchemy import create_engine, Column, Integer, Float, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func

# ========== Lightweight DB layer (SQLite + SQLAlchemy) ==========
@st.cache_resource
def get_engine():
    # SQLite file will live at app/data.db (relative to project root)
    return create_engine("sqlite:///app/data.db", echo=False, connect_args={"check_same_thread": False})

@st.cache_resource
def get_session_maker():
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    # Inputs
    description = Column(Text, nullable=False)
    weight_g = Column(Float, nullable=False)

    # Model outputs
    predicted_product_type = Column(String(64), nullable=False)
    storage_bin = Column(String(64), nullable=False)

    # Optional metadata
    model_name = Column(String(64))
    model_version = Column(String(64))

    created_at = Column(DateTime(timezone=True), server_default=func.now())

# --- NEW: feedback table (for corrections) ---
class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer, nullable=False)
    correct_product_type = Column(String(64))     # optional corrected label
    correct_storage_bin = Column(String(64))      # optional corrected bin
    notes = Column(Text)                          # optional free text
    created_at = Column(DateTime(timezone=True), server_default=func.now())

def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)

def save_prediction(rec: dict) -> int:
    """rec = {
        'description': str, 'weight_g': float,
        'predicted_product_type': str, 'storage_bin': str,
        'model_name': str, 'model_version': str
    }"""
    Session = get_session_maker()
    with Session() as s:
        row = Prediction(**rec)
        s.add(row)
        s.commit()
        s.refresh(row)
        return row.id

# --- NEW: save feedback ---
def save_feedback(prediction_id: int, correct_type: str | None, correct_bin: str | None, notes: str | None) -> int:
    Session = get_session_maker()
    with Session() as s:
        row = Feedback(
            prediction_id=prediction_id,
            correct_product_type=(correct_type or None),
            correct_storage_bin=(correct_bin or None),
            notes=(notes or None),
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        return row.id

def fetch_recent_predictions(limit=25) -> pd.DataFrame:
    Session = get_session_maker()
    with Session() as s:
        rows = s.query(Prediction).order_by(Prediction.id.desc()).limit(limit).all()
        data = [{
            "id": r.id,
            "created_at": r.created_at,
            "description": r.description,
            "weight_g": r.weight_g,
            "product_type": r.predicted_product_type,
            "storage_bin": r.storage_bin,
            "model_name": r.model_name,
            "model_version": r.model_version
        } for r in rows]
        return pd.DataFrame(data)

# Ensure DB/tables exist once per app run
init_db()
# ===============================================================


def render_model1_ui(model_input, model_name, mapping_data=None, debug=False):
    """
    Renders the UI for ANS Import & Export Storage Bin Prediction.

    Parameters:
        model_input: Either a full model_package dict or a trained sklearn model.
        model_name: Name of the model (string).
        mapping_data: Optional dict containing 'category_to_product_type' and 'product_type_to_bin'
                      if not included in model_input (for .pkl + .json setup).
        debug: Bool to print debug output.
    """

    st.subheader("🔍 ANS Import & Export - Storage Bin Prediction")

    st.markdown("""
    **Problem Statement:**  
    Samples are split between the **Office** and **Warehouse Bins**, making them hard to find.  
    This model predicts the **product type** and assigns the proper **storage bin** using **text description** and **weight**.

    **Inputs:** Text description, Weight (grams)  
    **Outputs:** Product Type → Storage Bin (Bin 1/2/3 or Office)
    """)

    # --- Determine if model_input is a package or just a model ---
    if isinstance(model_input, dict) and "model" in model_input:
        model = model_input["model"]
        label_encoder = model_input.get("label_encoder", None)
        category_to_product_type = {k.lower(): v for k, v in model_input.get("category_to_product_type", {}).items()}
        product_type_to_bin = {k.lower(): v for k, v in model_input.get("product_type_to_bin", {}).items()}
        model_version = "v1"
    else:
        model = model_input
        label_encoder = None
        if mapping_data is None:
            st.error("Mapping data is required when using model-only format (.pkl + .json).")
            return
        category_to_product_type = {k.lower(): v for k, v in mapping_data.get("category_to_product_type", {}).items()}
        product_type_to_bin = {k.lower(): v for k, v in mapping_data.get("product_type_to_bin", {}).items()}
        model_version = "v1"

    # --- User Inputs ---
    description = st.text_input(f"[{model_name}] Enter product description", key=f"desc_{model_name}")
    weight_g = st.number_input(f"[{model_name}] Enter product weight (grams)", min_value=0.0, key=f"weight_{model_name}")

    col_predict, col_history = st.columns([1, 1])

    with col_predict:
        if st.button(f"Predict with {model_name}"):
            if description and weight_g > 0:
                input_df = pd.DataFrame([{"description": description, "weight_g": weight_g}])

                # Step 1: Predict encoded label
                try:
                    predicted_encoded = model.predict(input_df)[0]
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    return

                # Step 2: Decode if label_encoder is available
                if label_encoder:
                    try:
                        predicted_category = label_encoder.inverse_transform([predicted_encoded])[0]
                    except Exception as e:
                        st.error(f"Label decoding failed: {e}")
                        return
                else:
                    predicted_category = predicted_encoded  # assume model returns string labels

                # Step 3: Map to product type
                product_type = category_to_product_type.get(str(predicted_category).lower(), predicted_category)

                # Step 4: Map to bin
                storage_bin = product_type_to_bin.get(str(product_type).lower(), "Unknown")

                # --- persist to DB ---
                try:
                    rec_id = save_prediction({
                        "description": description,
                        "weight_g": float(weight_g),
                        "predicted_product_type": str(product_type),
                        "storage_bin": str(storage_bin),
                        "model_name": str(model_name),
                        "model_version": model_version
                    })
                    st.success(f"Saved prediction (ID: {rec_id})")
                    st.session_state[f"last_rec_id_{model_name}"] = rec_id
                except Exception as e:
                    st.error(f"Could not save to database: {e}")

                # Output to user
                st.success(f"**Predicted Category:** {predicted_category}")
                st.info(f"**Product Type:** {product_type}")
                st.success(f"**Assigned Storage Bin:** {storage_bin}")

                # --- #3 Explainability: show top TF-IDF tokens present in input ---
                try:
                    vec = None
                    # Try the standard access path for ColumnTransformer
                    pre = getattr(model, "named_steps", {}).get("preprocess", None)
                    if pre is not None and hasattr(pre, "named_transformers_"):
                        vec = pre.named_transformers_.get("text", None)
                    # Fallback if needed (older sklearn)
                    if vec is None and hasattr(pre, "transformers_"):
                        for name, transformer, cols in pre.transformers_:
                            if name == "text":
                                vec = transformer
                                break

                    if vec is not None and hasattr(vec, "get_feature_names_out"):
                        vocab = vec.get_feature_names_out()
                        X_text = vec.transform([description])
                        arr = X_text.toarray()[0]
                        top_idx = arr.argsort()[-8:][::-1]
                        top_tokens = [vocab[i] for i in top_idx if arr[i] > 0]
                        if top_tokens:
                            st.caption("Most influential tokens (TF-IDF present in your text):")
                            st.write(", ".join(top_tokens))
                        else:
                            st.caption("No meaningful TF-IDF tokens detected in the input.")
                    else:
                        st.caption("Explainability unavailable (vectorizer not found).")
                except Exception as e:
                    st.warning(f"Explainability step skipped: {e}")

                # --- #2 Feedback: correction form (linked to this prediction) ---
                with st.expander("Was this correct? Submit a correction"):
                    rec_id_for_feedback = st.session_state.get(f"last_rec_id_{model_name}")
                    if rec_id_for_feedback:
                        corrected_type = st.text_input("Correct product type (optional)", key=f"fix_type_{model_name}")
                        corrected_bin = st.text_input("Correct storage bin (optional)", key=f"fix_bin_{model_name}")
                        notes = st.text_area("Notes (optional)", key=f"fix_notes_{model_name}")
                        if st.button("Submit correction", key=f"submit_fix_{model_name}"):
                            try:
                                fb_id = save_feedback(rec_id_for_feedback, corrected_type.strip() or None,
                                                      corrected_bin.strip() or None, notes.strip() or None)
                                st.success(f"Thanks! Correction saved (Feedback ID: {fb_id}).")
                            except Exception as e:
                                st.error(f"Could not save feedback: {e}")
                    else:
                        st.caption("Make a prediction first to enable corrections.")

                if debug:
                    st.write("🔍 Debug Info:", {
                        "Encoded Prediction": predicted_encoded,
                        "Predicted Category": predicted_category,
                        "Product Type": product_type,
                        "Storage Bin": storage_bin
                    })
            else:
                st.warning("Please enter both a description and a valid weight.")

    with col_history:
        st.markdown("#### Recent predictions")
        try:
            df_hist = fetch_recent_predictions(limit=25)
            if df_hist.empty:
                st.caption("No records yet.")
            else:
                st.dataframe(df_hist, use_container_width=True, hide_index=True)

                # --- #7 Export: download CSV of recent predictions ---
                csv_bytes = df_hist.to_csv(index=False).encode()
                st.download_button(
                    label="Download CSV",
                    data=csv_bytes,
                    file_name="predictions_recent.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Could not read history: {e}")
