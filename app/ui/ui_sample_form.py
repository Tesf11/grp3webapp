# app/ui/ui_sample_form.py
import os
import requests
import streamlit as st
from datetime import date

def render_sample_form(api_url: str | None = None, title: str = "📝 New Sample Entry"):
    """
    Streamlit form for creating a sample record.
    - Calls Flask backend to predict category from Product Title.
    - Returns the submitted record as a dict (you can insert into DB later).
    """
    API_URL = api_url or os.getenv("API_URL", "http://127.0.0.1:5000/api/predict")

    st.header(title)
    st.caption("Fill the fields. Category is predicted from **Product Title** via the backend; you can edit it before saving.")

    # choices (adjust as your team agrees)
    SAMPLE_TYPES = ["Incoming", "Outgoing"]
    STATUSES = ["Received", "Sent", "In transit", "Pending"]

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

        # Product title drives category
        product_title = st.text_input("Product Title (used to predict Category)",
                                      placeholder="e.g., Disposable Linen (SIA sample)")

        eta = st.date_input("ETA", value=date.today())

        # Predict + allow edit
        pred_btn = st.form_submit_button("Predict Category")
        save_btn = st.form_submit_button("Save Entry")

        # state to hold last prediction
        if "sample_pred_category" not in st.session_state:
            st.session_state.sample_pred_category = ""

        # When Predict pressed, call backend
        if pred_btn:
            t = (product_title or "").strip()
            if not t:
                st.warning("Enter a Product Title first.")
            else:
                try:
                    resp = requests.post(API_URL, json={"title": t}, timeout=15)
                    if resp.status_code == 200:
                        st.session_state.sample_pred_category = resp.json().get("category", "")
                        st.success(f"Predicted Category: **{st.session_state.sample_pred_category}**")
                    else:
                        st.error(f"Backend error ({resp.status_code}): {resp.text}")
                except requests.RequestException as e:
                    st.error(f"Failed to connect to backend: {e}")

        # Let user edit/override category (whether predicted or not)
        category = st.text_input("Category", value=st.session_state.sample_pred_category, placeholder="Edit if needed")

        # Save pressed → basic validation & return dict
        record = None
        if save_btn:
            required = {
                "Sample Type": sample_type,
                "Company": company,
                "Product Title": product_title,
                "Status": status,
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
                st.success("✅ Entry ready to save.")
        return record
