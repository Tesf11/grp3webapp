# app/ui/ui_model1.py
import streamlit as st
import pandas as pd
from typing import Optional

# ✅ Shared SQLAlchemy base/session + models
from app.db import get_session, create_all
from app.models import Prediction, Feedback, MainStorageItem  # ← use shared models

# --- put near the top of app/ui/ui_model1.py (after imports) ---
from sqlalchemy import text

def _ensure_feedback_is_correct_column():
    """
    Adds feedback.is_correct if missing and backfills:
      - Yes (True) when no corrections were provided
      - No  (False) when a correction exists
    Works for SQLite and Postgres.
    """
    with get_session() as s:
        # detect column
        try:
            # SQLite: PRAGMA table_info
            cols = s.execute(text("PRAGMA table_info(feedback)")).fetchall()
            colnames = {c[1] for c in cols}  # (cid, name, type, notnull, dflt_value, pk)
        except Exception:
            # Postgres fallback
            cols = s.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'feedback'
            """)).fetchall()
            colnames = {c[0] for c in cols}

        if "is_correct" not in colnames:
            # add the column, default True so existing rows become "Yes"
            s.execute(text("ALTER TABLE feedback ADD COLUMN is_correct BOOLEAN DEFAULT 1"))
            # Backfill explicit True/False based on presence of corrections
            s.execute(text("""
                UPDATE feedback
                SET is_correct = CASE
                    WHEN (correct_product_type IS NULL OR TRIM(correct_product_type) = '')
                     AND (correct_storage_bin  IS NULL OR TRIM(correct_storage_bin)  = '')
                    THEN 1 ELSE 0 END
                WHERE is_correct IS NULL
            """))
            s.commit()


# ---------- UI TWEAKS ----------
st.set_page_config(layout="wide")
_TABLE_HEIGHT = 420  # roomy table height for all three tables

# Ensure all tables exist (uses models from app.models)
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
    is_correct: bool,                      # ← store Yes/No as a bool
    correct_type: Optional[str],
    correct_bin: Optional[str],
    notes: Optional[str],
) -> int:
    with get_session() as s:
        row = Feedback(
            prediction_id=prediction_id,
            is_correct=bool(is_correct),
            # keep corrections only when incorrect
            correct_product_type=None if is_correct else (correct_type or None),
            correct_storage_bin=None if is_correct else (correct_bin or None),
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
# Feedback view/delete (with description + stored answer)
# =========================
def fetch_recent_feedback(limit: int = 200) -> pd.DataFrame:
    """Return recent feedback rows, including the original prediction description and Yes/No answer."""
    with get_session() as s:
        rows = (
            s.query(Feedback, Prediction.description)
            .join(Prediction, Feedback.prediction_id == Prediction.id, isouter=True)
            .order_by(Feedback.id.desc())
            .limit(limit)
            .all()
        )
        data = []
        for fb_row, pred_desc in rows:
            data.append(
                {
                    "id": fb_row.id,
                    "created_at": fb_row.created_at,
                    "prediction_id": fb_row.prediction_id,
                    "description": pred_desc,
                    "answer": "Yes" if fb_row.is_correct else "No",
                    "correct_product_type": fb_row.correct_product_type,
                    "correct_storage_bin": fb_row.correct_storage_bin,
                    "notes": fb_row.notes,
                }
            )
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
# Tiny cache wrappers to make tab switching feel instant
# ------------------------------
@st.cache_data(show_spinner=False, ttl=5)
def _cached_recent_predictions(limit: int = 25) -> pd.DataFrame:
    return fetch_recent_predictions(limit)

@st.cache_data(show_spinner=False, ttl=5)
def _cached_main_storage(limit: int = 200) -> pd.DataFrame:
    return fetch_main_storage(limit)

@st.cache_data(show_spinner=False, ttl=5)
def _cached_recent_feedback(limit: int = 200) -> pd.DataFrame:
    return fetch_recent_feedback(limit)

def _clear_caches():
    st.cache_data.clear()  # Clear all cached table queries after any write


# ------------------------------
# Main UI
# ------------------------------
def render_model1_ui(model_input, model_name, mapping_data=None, debug=False):
    """
    Renders the UI for ANS Import & Export Storage Bin Prediction.
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

    # --- User Inputs (aligned button) ---
    with st.container():
        c1, c2, c3 = st.columns([3, 1.3, 1.0])
        with c1:
            description = st.text_input(f"Enter product description", key=f"desc_{model_name}")
        with c2:
            weight_g = st.number_input(
                f"Enter product weight (grams)", min_value=0.0, key=f"weight_{model_name}"
            )
        with c3:
            # spacer to vertically align the button with the inputs
            st.markdown("<div style='height: 1.75rem'></div>", unsafe_allow_html=True)
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
                st.session_state[f"feedback_open_{model_name}"] = True  # keep feedback panel open
                _clear_caches()  # ✅ refresh cached tables
            except Exception as e:
                st.error(f"Could not save to database: {e}")

            # Output to user (category line removed)
            st.info(f"**Product Type:** {product_type}")
            st.success(f"**Assigned Storage Bin:** {storage_bin}")

            # --- Explainability: TF-IDF tokens present in input ---
            try:
                vec = None
                pre = getattr(model, "named_steps", {}).get("preprocess", None)
                if pre is not None and hasattr(pre, "named_transformers_"):
                    vec = pre.named_transformers_.get("text", None)
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
                        fb_id = save_feedback(
                            rec_id_for_feedback,
                            True,                # ← is_correct
                            correct_type=None,
                            correct_bin=None,
                            notes=notes.strip() or None,
                        )
                        st.success(f"Thanks! Confirmation saved (Feedback ID: {fb_id}).")
                        st.session_state[f"feedback_open_{model_name}"] = False
                        _clear_caches()  # ✅ refresh cached tables
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
                            False,               # ← is_correct
                            correct_type=corrected_type.strip() or None,
                            correct_bin=corrected_bin,
                            notes=notes.strip() or None,
                        )
                        st.success(f"Thanks! Correction saved (Feedback ID: {fb_id}).")
                        st.session_state[f"feedback_open_{model_name}"] = False
                        _clear_caches()  # ✅ refresh cached tables
                    except Exception as e:
                        st.error(f"Could not save feedback: {e}")

    st.divider()
    st.markdown("## 📊 Data Management")

    # ---------- CSS: remove focus ring & keep buttons transparent ----------
    st.markdown(
        """
        <style>
        /* No focus red ring or border on any Streamlit button */
        .stButton > button:focus,
        .stButton > button:focus-visible,
        .stButton > button:active {
            outline: none !important;
            box-shadow: 0 0 0 0 rgba(0,0,0,0) !important;
            border-color: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Segmented “tab-like” buttons (sticky) ----------
    seg_key = f"data_tab_{model_name}"
    if seg_key not in st.session_state:
        st.session_state[seg_key] = "Recent Predictions"
    labels = ["Recent Predictions", "Main Storage", "Feedback"]
    cA, cB, cC = st.columns(3)
    for i, lbl in enumerate(labels):
        col = [cA, cB, cC][i]
        is_selected = st.session_state[seg_key] == lbl
        # Use secondary for both states to avoid any theme red; keep transparent feel.
        # Small visual cue: prefix a dot when selected.
        btn_label = f"● {lbl}" if is_selected else lbl
        if col.button(
            btn_label,
            use_container_width=True,
            type="secondary",               # ← no red primary background
            key=f"{seg_key}_{i}"
        ):
            st.session_state[seg_key] = lbl
    st.markdown("<hr style='margin-top:0.5rem;margin-bottom:0.75rem'/>", unsafe_allow_html=True)

    tab_choice = st.session_state[seg_key]

    # ========== SECTION: Recent Predictions ==========
    if tab_choice == "Recent Predictions":
        st.caption("Review predictions and perform bulk actions.")
        try:
            with st.spinner("Loading…"):
                df_hist = _cached_recent_predictions(limit=25)
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

                ca, cb = st.columns([1, 1])
                with ca:
                    if st.button(
                        "Move selected to main storage",
                        use_container_width=True,
                        disabled=not sel_ids,
                        key=f"btn_move_selected_{model_name}",
                    ):
                        # move → then delete moved rows so they disappear from this list
                        moved = _bulk_move(sel_ids)
                        deleted_after_move = _bulk_delete_predictions(sel_ids)
                        _clear_caches()
                        st.success(f"Moved {moved} item(s) to main storage. Removed {deleted_after_move} from recent.")
                with cb:
                    if st.button(
                        "Delete selected predictions",
                        use_container_width=True,
                        disabled=not sel_ids,
                        key=f"btn_delete_selected_{model_name}",
                    ):
                        deleted = _bulk_delete_predictions(sel_ids)
                        _clear_caches()
                        st.success(f"Deleted {deleted} prediction(s).")
        except Exception as e:
            st.error(f"Could not read history: {e}")

    # ========== SECTION: Main Storage ==========
    if tab_choice == "Main Storage":
        st.caption("Finalized items. You can delete them here.")
        try:
            with st.spinner("Loading…"):
                df_main = _cached_main_storage(limit=200)
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

                if st.button(
                    "Delete selected main storage items",
                    use_container_width=True,
                    disabled=not sel_main_ids,
                    key=f"btn_delete_main_selected_{model_name}",
                ):
                    deleted = _bulk_delete_main(sel_main_ids)
                    _clear_caches()
                    st.success(f"Deleted {deleted} main storage item(s).")

        except Exception as e:
            st.error(f"Could not read main storage: {e}")

    # ========== SECTION: Feedback ==========
    if tab_choice == "Feedback":
        st.caption("All feedback entries (confirmations and corrections).")
        try:
            with st.spinner("Loading…"):
                df_fb = _cached_recent_feedback(limit=200)
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
                        "description": st.column_config.TextColumn(width="medium"),
                        "answer": st.column_config.TextColumn(width="small"),
                        "notes": st.column_config.TextColumn(width="medium"),
                        "created_at": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                    },
                    num_rows="fixed",
                )

                sel_fb_ids = edited_fb.loc[edited_fb["select"], "id"].tolist()

                if st.button(
                    "Delete selected feedback",
                    use_container_width=True,
                    disabled=not sel_fb_ids,
                    key=f"btn_delete_feedback_selected_{model_name}",
                ):
                    deleted = _bulk_delete_feedback(sel_fb_ids)
                    _clear_caches()
                    st.success(f"Deleted {deleted} feedback entry(ies).")

        except Exception as e:
            st.error(f"Could not read feedback: {e}")

    # --- optional debug ---
    if debug:
        st.caption("🔍 Debug enabled.")


# =========================
# Bulk-action helpers
# =========================
def _bulk_move(ids):
    moved = 0
    for rid in ids:
        try:
            if move_prediction_to_main(int(rid)):
                moved += 1
        except Exception:
            pass
    return moved


def _bulk_delete_predictions(ids):
    deleted = 0
    for rid in ids:
        try:
            if delete_prediction(int(rid)):
                deleted += 1
        except Exception:
            pass
    return deleted


def _bulk_delete_main(ids):
    deleted = 0
    for mid in ids:
        try:
            if delete_main_item(int(mid)):
                deleted += 1
        except Exception:
            pass
    return deleted


def _bulk_delete_feedback(ids):
    deleted = 0
    for fid in ids:
        try:
            if delete_feedback(int(fid)):
                deleted += 1
        except Exception:
            pass
    return deleted
