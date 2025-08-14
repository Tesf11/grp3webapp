# app/ui/ui_model1.py
import streamlit as st
import pandas as pd
from typing import Optional

# ✅ Use the shared SQLAlchemy base/session + models
from app.db import Base, get_session, create_all
from app.models import Prediction, Feedback

# =========================
# ADDITIONS (no removals)
# =========================
# Extra imports for new helper model/table
from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from sqlalchemy.sql import func

# ---------- UI TWEAKS ----------
st.set_page_config(layout="wide")
_TABLE_HEIGHT = 420  # roomy table height for all three tables

# A simple "Main Storage" table to hold finalized items
class MainStorageItem(Base):
    __tablename__ = "main_storage"
    id = Column(Integer, primary_key=True)
    product_type = Column(String(64), nullable=False)
    description = Column(Text, nullable=False)
    weight_g = Column(Float, nullable=False)
    storage_bin = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# Ensure all tables (including the new one) exist
create_all()


# ------------------------------
# DB helper functions (shared DB)
# ------------------------------
def save_prediction(rec: dict) -> int:
    """
    rec = {
        'description': str, 'weight_g': float,
        'predicted_product_type': str, 'storage_bin': str,
        'model_name': str, 'model_version': str
    }
    """
    with get_session() as s:
        row = Prediction(**rec)
        s.add(row)
        s.flush()  # assigns PK without closing the session
        rid = row.id
        s.commit()  # ✅ explicit commit
        return rid


def save_feedback(
    prediction_id: int,
    correct_type: Optional[str],
    correct_bin: Optional[str],
    notes: Optional[str],
) -> int:
    with get_session() as s:
        row = Feedback(
            prediction_id=prediction_id,
            correct_product_type=(correct_type or None),
            correct_storage_bin=(correct_bin or None),
            notes=(notes or None),
        )
        s.add(row)
        s.flush()
        fid = row.id
        s.commit()  # ✅ explicit commit
        return fid


def fetch_recent_predictions(limit: int = 25) -> pd.DataFrame:
    with get_session() as s:
        rows = (
            s.query(Prediction)
            .order_by(Prediction.id.desc())
            .limit(limit)
            .all()
        )
        data = [
            {
                "id": r.id,
                "created_at": r.created_at,
                "description": r.description,
                "weight_g": r.weight_g,
                "product_type": r.predicted_product_type,
                "storage_bin": r.storage_bin,
                "model_name": r.model_name,
                "model_version": r.model_version,
            }
            for r in rows
        ]
        return pd.DataFrame(data)


# =========================
# ADDITIONS (no removals)
# Helpers for Main Storage + actions
# =========================
def move_prediction_to_main(prediction_id: int) -> Optional[int]:
    """Copy a prediction into the main storage table."""
    with get_session() as s:
        obj = s.get(Prediction, prediction_id)
        if not obj:
            return None
        ms = MainStorageItem(
            product_type=obj.predicted_product_type,
            description=obj.description,
            weight_g=obj.weight_g,
            storage_bin=obj.storage_bin,
        )
        s.add(ms)
        s.flush()
        mid = ms.id
        s.commit()  # ✅ explicit commit
        return mid


def delete_prediction(prediction_id: int) -> bool:
    with get_session() as s:
        obj = s.get(Prediction, prediction_id)
        if not obj:
            return False
        s.delete(obj)
        s.commit()  # ✅ explicit commit
        return True


def fetch_main_storage(limit: int = 200) -> pd.DataFrame:
    with get_session() as s:
        rows = (
            s.query(MainStorageItem)
            .order_by(MainStorageItem.id.desc())
            .limit(limit)
            .all()
        )
        data = [
            {
                "id": r.id,
                "created_at": r.created_at,
                "product_type": r.product_type,
                "description": r.description,
                "weight_g": r.weight_g,
                "storage_bin": r.storage_bin,
            }
            for r in rows
        ]
        return pd.DataFrame(data)


def delete_main_item(item_id: int) -> bool:
    with get_session() as s:
        obj = s.get(MainStorageItem, item_id)
        if not obj:
            return False
        s.delete(obj)
        s.commit()  # ✅ explicit commit
        return True


# =========================
# ADDITIONS (new helper): Feedback view/delete
# =========================
def fetch_recent_feedback(limit: int = 200) -> pd.DataFrame:
    with get_session() as s:
        rows = (
            s.query(Feedback)
            .order_by(Feedback.id.desc())
            .limit(limit)
            .all()
        )
        data = [
            {
                "id": r.id,
                "created_at": r.created_at,
                "prediction_id": r.prediction_id,
                "correct_product_type": r.correct_product_type,
                "correct_storage_bin": r.correct_storage_bin,
                "notes": r.notes,
            }
            for r in rows
        ]
        return pd.DataFrame(data)


def delete_feedback(feedback_id: int) -> bool:
    with get_session() as s:
        obj = s.get(Feedback, feedback_id)
        if not obj:
            return False
        s.delete(obj)
        s.commit()  # ✅ explicit commit
        return True


# ------------------------------
# Main UI
# ------------------------------
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

    st.markdown(
        """
**Problem Statement:**  
Samples are split between the **Office** and **Warehouse Bins**, making them hard to find.  
This model predicts the **product type** and assigns the proper **storage bin** using **text description** and **weight**.

**Inputs:** Text description, Weight (grams)  
**Outputs:** Product Type → Storage Bin (Bin 1/2/3 or Office)

**Storage Locations**
- **Bin 1:** Bulk & Perishables *(food, rice, fruits, snacks, drinks)*
- **Bin 2:** Durables & Utilities *(tools, furniture, toys, appliances)*
- **Bin 3:** Sensitive & Spill-Risk *(liquids, cosmetics, toiletries)*
- **Office:** Controlled & Tracked items *(electronics, office supplies, media, baby products)*
"""
    )

    # --- Determine if model_input is a package or just a model ---
    if isinstance(model_input, dict) and "model" in model_input:
        model = model_input["model"]
        label_encoder = model_input.get("label_encoder", None)
        category_to_product_type = {
            k.lower(): v for k, v in model_input.get("category_to_product_type", {}).items()
        }
        product_type_to_bin = {
            k.lower(): v for k, v in model_input.get("product_type_to_bin", {}).items()
        }
        model_version = "v1"
    else:
        model = model_input
        label_encoder = None
        if mapping_data is None:
            st.error("Mapping data is required when using model-only format (.pkl + .json).")
            return
        category_to_product_type = {
            k.lower(): v for k, v in mapping_data.get("category_to_product_type", {}).items()
        }
        product_type_to_bin = {
            k.lower(): v for k, v in mapping_data.get("product_type_to_bin", {}).items()
        }
        model_version = "v1"

    # --- User Inputs ---
    with st.container():
        c1, c2, c3 = st.columns([3, 1.3, 1.0])
        with c1:
            description = st.text_input(f"Enter product description", key=f"desc_{model_name}")
        with c2:
            weight_g = st.number_input(
                f"Enter product weight (grams)", min_value=0.0, key=f"weight_{model_name}"
            )
        with c3:
            run_pred = st.button("Start Prediction", use_container_width=True)

    if run_pred:
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
            product_type = category_to_product_type.get(
                str(predicted_category).lower(), predicted_category
            )

            # Step 4: Map to bin
            storage_bin = product_type_to_bin.get(str(product_type).lower(), "Unknown")

            # --- persist to DB ---
            try:
                rec_id = save_prediction(
                    {
                        "description": description,
                        "weight_g": float(weight_g),
                        "predicted_product_type": str(product_type),
                        "storage_bin": str(storage_bin),
                        "model_name": str(model_name),
                        "model_version": model_version,
                    }
                )
                st.success(f"Saved prediction (ID: {rec_id})")
                st.session_state[f"last_rec_id_{model_name}"] = rec_id
                st.session_state[f"feedback_open_{model_name}"] = True  # 🔒 keep feedback panel open
            except Exception as e:
                st.error(f"Could not save to database: {e}")

            # Output to user
            st.success(f"**Predicted Category:** {predicted_category}")
            st.info(f"**Product Type:** {product_type}")
            st.success(f"**Assigned Storage Bin:** {storage_bin}")

            # --- Explainability: TF-IDF tokens present in input ---
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

        else:
            st.warning("Please enter both a description and a valid weight.")

    # --- Feedback: streamlined UX (Yes/No + corrected bin) ---
    with st.expander(
        "Is this prediction correct?",
        expanded=st.session_state.get(f"feedback_open_{model_name}", False)
    ):
        rec_id_for_feedback = st.session_state.get(f"last_rec_id_{model_name}")

        if not rec_id_for_feedback:
            st.caption("Make a prediction first to enable feedback.")
        else:
            # Locations with purposes (display text = label with purpose)
            bin_options = [
                "Bin 1 (Bulk & Perishables)",
                "Bin 2 (Durables & Utilities)",
                "Bin 3 (Sensitive & Spill-Risk)",
                "Office (Controlled & Tracked)",
            ]

            choice = st.radio(
                "Was this correct?",
                options=["Yes", "No"],
                horizontal=True,
                key=f"fb_correct_{model_name}"
            )

            if choice == "Yes":
                notes = st.text_area(
                    "Notes (optional)",
                    key=f"fb_notes_yes_{model_name}",
                    placeholder="e.g., Looks correct for this item"
                )
                if st.button("Confirm", key=f"fb_submit_yes_{model_name}", use_container_width=True):
                    try:
                        # Confirmation: we record notes only (no overrides)
                        fb_id = save_feedback(
                            rec_id_for_feedback,
                            correct_type=None,
                            correct_bin=None,
                            notes=notes.strip() or None,
                        )
                        st.success(f"Thanks! Confirmation saved (Feedback ID: {fb_id}).")
                        st.session_state[f"feedback_open_{model_name}"] = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not save feedback: {e}")

            else:  # choice == "No"
                corrected_bin = st.selectbox(
                    "Select the correct storage location",
                    bin_options,
                    index=0,
                    key=f"fb_bin_{model_name}",
                )
                corrected_type = st.text_input(
                    "Correct product type (optional)",
                    key=f"fb_type_{model_name}",
                    placeholder="e.g., Toiletries / Electronics / Dry Food ..."
                )
                notes = st.text_area(
                    "Notes (optional)",
                    key=f"fb_notes_no_{model_name}",
                    placeholder="Why is it wrong? Any hints to learn from next time?"
                )
                if st.button("Submit correction", key=f"fb_submit_no_{model_name}", use_container_width=True):
                    try:
                        fb_id = save_feedback(
                            rec_id_for_feedback,
                            correct_type=corrected_type.strip() or None,
                            correct_bin=corrected_bin,
                            notes=notes.strip() or None,
                        )
                        st.success(f"Thanks! Correction saved (Feedback ID: {fb_id}).")
                        st.session_state[f"feedback_open_{model_name}"] = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not save feedback: {e}")

    st.divider()
    st.markdown("## 📊 Data Management")

    # ---------- Three roomy tabs: Predictions / Main Storage / Feedback ----------
    tab_pred, tab_main, tab_fb = st.tabs(
        ["Recent Predictions", "Main Storage", "Feedback"]
    )

    # ========== TAB: Recent Predictions ==========
    with tab_pred:
        st.caption("Review predictions and perform bulk actions.")
        try:
            df_hist = fetch_recent_predictions(limit=25)
            if df_hist.empty:
                st.caption("No records yet.")
            else:
                df_hist_disp = df_hist.copy()
                df_hist_disp.insert(0, "select", False)

                edited_hist = st.data_editor(
                    df_hist_disp,
                    use_container_width=True,
                    hide_index=True,
                    height=_TABLE_HEIGHT,
                    column_config={
                        "select": st.column_config.CheckboxColumn("✓", help="Select rows to act on"),
                        "description": st.column_config.TextColumn(width="medium"),
                        "created_at": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                    },
                    num_rows="fixed",
                )

                sel_ids = edited_hist.loc[edited_hist["select"], "id"].tolist()

                cA, cB = st.columns([1, 1])
                with cA:
                    st.button(
                        "Move selected to main storage",
                        use_container_width=True,
                        disabled=not sel_ids,
                        key="btn_move_selected",
                        on_click=lambda: _bulk_move(sel_ids),
                    )
                with cB:
                    st.button(
                        "Delete selected predictions",
                        use_container_width=True,
                        disabled=not sel_ids,
                        key="btn_delete_selected",
                        on_click=lambda: _bulk_delete_predictions(sel_ids),
                    )
        except Exception as e:
            st.error(f"Could not read history: {e}")

    # ========== TAB: Main Storage ==========
    with tab_main:
        st.caption("Finalized items. You can delete them here.")
        try:
            df_main = fetch_main_storage(limit=200)
            if df_main.empty:
                st.caption("Main storage is empty.")
            else:
                show_cols = ["id", "product_type", "description", "weight_g", "storage_bin", "created_at"]
                for col in show_cols:
                    if col not in df_main.columns:
                        df_main[col] = None

                df_main_disp = df_main[show_cols].copy()
                df_main_disp.insert(0, "select", False)

                edited_main = st.data_editor(
                    df_main_disp,
                    use_container_width=True,
                    hide_index=True,
                    height=_TABLE_HEIGHT,
                    column_config={
                        "select": st.column_config.CheckboxColumn("✓", help="Select items to delete"),
                        "description": st.column_config.TextColumn(width="medium"),
                        "created_at": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                    },
                    num_rows="fixed",
                )

                sel_main_ids = edited_main.loc[edited_main["select"], "id"].tolist()

                st.button(
                    "Delete selected main storage items",
                    use_container_width=True,
                    disabled=not sel_main_ids,
                    key="btn_delete_main_selected",
                    on_click=lambda: _bulk_delete_main(sel_main_ids),
                )

        except Exception as e:
            st.error(f"Could not read main storage: {e}")

    # ========== TAB: Feedback ==========
    with tab_fb:
        st.caption("All feedback entries (confirmations and corrections).")
        try:
            df_fb = fetch_recent_feedback(limit=200)
            if df_fb.empty:
                st.caption("No feedback yet.")
            else:
                df_fb_disp = df_fb.copy()
                df_fb_disp.insert(0, "select", False)

                edited_fb = st.data_editor(
                    df_fb_disp,
                    use_container_width=True,
                    hide_index=True,
                    height=_TABLE_HEIGHT,
                    column_config={
                        "select": st.column_config.CheckboxColumn("✓", help="Select feedback to delete"),
                        "notes": st.column_config.TextColumn(width="medium"),
                        "created_at": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                    },
                    num_rows="fixed",
                )

                sel_fb_ids = edited_fb.loc[edited_fb["select"], "id"].tolist()

                st.button(
                    "Delete selected feedback",
                    use_container_width=True,
                    disabled=not sel_fb_ids,
                    key="btn_delete_feedback_selected",
                    on_click=lambda: _bulk_delete_feedback(sel_fb_ids),
                )

        except Exception as e:
            st.error(f"Could not read feedback: {e}")

    # --- optional debug ---
    if debug:
        st.caption("🔍 Debug enabled.")


# =========================
# ADDITIONS: bulk-action helpers for neat buttons
# =========================
def _bulk_move(ids):
    moved = 0
    for rid in ids:
        try:
            if move_prediction_to_main(int(rid)):
                moved += 1
        except Exception:
            pass
    st.success(f"Moved {moved} item(s) to main storage.")
    st.rerun()


def _bulk_delete_predictions(ids):
    deleted = 0
    for rid in ids:
        try:
            if delete_prediction(int(rid)):
                deleted += 1
        except Exception:
            pass
    st.success(f"Deleted {deleted} prediction(s).")
    st.rerun()


def _bulk_delete_main(ids):
    deleted = 0
    for mid in ids:
        try:
            if delete_main_item(int(mid)):
                deleted += 1
        except Exception:
            pass
    st.success(f"Deleted {deleted} main storage item(s).")
    st.rerun()


def _bulk_delete_feedback(ids):
    deleted = 0
    for fid in ids:
        try:
            if delete_feedback(int(fid)):
                deleted += 1
        except Exception:
            pass
    st.success(f"Deleted {deleted} feedback entry(ies).")
    st.rerun()
