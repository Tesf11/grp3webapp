# app/ui/ui_sample_form.py
import os
import requests
import streamlit as st
from datetime import date

def render_sample_form(api_url: str | None = None, title: str = "📝 New Sample Entry"):
    """
    Streamlit form for creating a sample record.
    - Calls Flask backend to predict category from Product Title.
    - On Save, POSTs the full record to /api/save/entries (same backend).
    """
    PREDICT_URL = api_url or os.getenv("API_URL", "http://127.0.0.1:5000/api/predict")
    SAVE_URL    = os.getenv("SAVE_API_URL", "http://127.0.0.1:5000/api/save/entries")

    st.header(title)
    st.caption("Fill the fields. Category is predicted from **Product Title** via the backend; you can edit it before saving.")

    SAMPLE_TYPES = ["Incoming", "Outgoing"]
    STATUSES = ["Received", "Sent", "In transit", "Pending"]

    if "sample_pred_category" not in st.session_state:
        st.session_state.sample_pred_category = ""

    with st.form("sample_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            sample_type = st.selectbox("Sample Type", SAMPLE_TYPES, index=0)
            company = st.text_input("Company", placeholder="e.g., Welles Paper")
            contact_person = st.text_input("Contact Person", placeholder="e.g., Annie")

        with col2:
            tracking_number = st.text_input("Tracking Number", placeholder="e.g., SF0261879005639")
            status = st.selectbox("Status", STATUSES, index=0)
            courier_cost = st.number_input("Courier Cost", min_value=0.0, step=0.5, format="%.2f")

        product_title = st.text_input(
            "Product Title (used to predict Category)",
            placeholder="e.g., Disposable Linen (SIA sample)"
        )
        eta = st.date_input("ETA", value=date.today())

        c1, c2 = st.columns(2)
        with c1:
            pred_btn = st.form_submit_button("🔮 Predict Category")
        with c2:
            save_btn = st.form_submit_button("💾 Save Entry")

        # Predict
        if pred_btn:
            t = (product_title or "").strip()
            if not t:
                st.warning("Enter a Product Title first.")
            else:
                try:
                    with st.spinner("Predicting…"):
                        resp = requests.post(PREDICT_URL, json={"title": t}, timeout=15)
                    if resp.status_code == 200:
                        st.session_state.sample_pred_category = resp.json().get("category", "")
                        st.success(f"Predicted Category: **{st.session_state.sample_pred_category}**")
                    else:
                        st.error(f"Backend error ({resp.status_code}): {resp.text}")
                except requests.RequestException as e:
                    st.error(f"Failed to connect to backend: {e}")

        # Editable category (prefilled if predicted)
        category = st.text_input("Category", value=st.session_state.sample_pred_category, placeholder="Edit if needed")

        # Save (ONLY runs when save_btn is pressed and validation passes)
        if save_btn:
            required = {
                "Sample Type": sample_type,
                "Company": company,
                "Product Title": product_title,
                "Status": status,
                "Category": category,
            }
            missing = [k for k, v in required.items() if not str(v).strip()]
            if missing:
                st.error(f"Please fill required fields: {', '.join(missing)}")
            else:
                record = {
                    "sample_type": sample_type,
                    "company": company,
                    "contact_person": contact_person,
                    "product_title": product_title,
                    "category": category,
                    "tracking_number": tracking_number,
                    "status": status,
                    "courier_cost": float(courier_cost or 0),
                    "eta": str(eta),
                }
                try:
                    with st.spinner("Saving…"):
                        r = requests.post(SAVE_URL, json=record, timeout=15)
                    if r.status_code == 200:
                        st.success(f"✅ Saved! ID = {r.json().get('id')}")
                        # Optional: auto-return to entries page (if your entries page checks this flag)
                        st.session_state.show_add_form = False
                        st.session_state.redirect_to_entries = True
                    else:
                        st.error(f"Save failed ({r.status_code}): {r.text}")
                except requests.RequestException as e:
                    st.error(f"Failed to save: {e}")
