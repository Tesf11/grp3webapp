# app/ui/ui_entries.py
import os
import requests
import streamlit as st
from typing import List, Dict

def _fetch_entries(limit: int = 100) -> List[Dict]:
    url = os.getenv("LIST_API_URL", "http://127.0.0.1:5000/api/list/entries")
    try:
        r = requests.get(url, params={"limit": limit}, timeout=15)
        if r.status_code == 200:
            return r.json()
        st.error(f"List API error ({r.status_code}): {r.text}")
    except requests.RequestException as e:
        st.error(f"Failed to reach List API: {e}")
    return []

def render_entries_page():
    """
    View Entries page with 'Add Sample' drawer at the TOP,
    then dynamic dropdown filters (Status, Category) and the table.
    """
    from app.ui.ui_sample_form import render_sample_form

    st.header("📚 View Entries")

    # UI state
    if "show_add_form" not in st.session_state:
        st.session_state.show_add_form = False

    # Redirect after save (from the form)
    if st.session_state.get("redirect_to_entries"):
        st.session_state.redirect_to_entries = False
        st.session_state.show_add_form = False
        st.session_state.cached_entries = _fetch_entries(limit=100)
        st.toast("Saved! List refreshed.", icon="✅")

    # ---------- TOP: Add Sample drawer ----------
    top_left, _top_right = st.columns([1, 3])
    with top_left:
        if not st.session_state.show_add_form:
            if st.button("➕ Add Sample", type="primary", use_container_width=True):
                st.session_state.sample_pred_category = ""  # clear previous prediction
                st.session_state.show_add_form = True
        else:
            if st.button("⬅️ Back to list", use_container_width=True):
                st.session_state.show_add_form = False
                st.session_state.cached_entries = _fetch_entries(limit=100)

    if st.session_state.show_add_form:
        st.markdown("### ➕ Add New Sample")
        render_sample_form()  # sets redirect_to_entries=True after successful save
        st.divider()

    # ---------- BELOW: Fetch (ensure we have data to build dropdowns) ----------
    # If no cache yet, or after page load, fetch initial set to build filter choices.
    if "cached_entries" not in st.session_state:
        st.session_state.cached_entries = _fetch_entries(limit=100)

    # Build dynamic dropdown choices from cached rows
    base_rows = st.session_state.cached_entries or []
    categories = sorted({str(r.get("category", "")).strip() for r in base_rows if r.get("category")})
    statuses   = sorted({str(r.get("status", "")).strip()   for r in base_rows if r.get("status")})

    # Add "All" option
    category_options = ["All"] + categories
    status_options   = ["All"] + statuses

    # ---------- Filters + Refresh ----------
    c1, c2, c3, c4 = st.columns([1, 1.5, 1.5, 1])
    with c1:
        limit = st.number_input("Rows", min_value=10, max_value=500, value=100, step=10)
    with c2:
        sel_category = st.selectbox("Filter by Category", options=category_options, index=0)
    with c3:
        sel_status = st.selectbox("Filter by Status", options=status_options, index=0)
    with c4:
        st.write("")  # spacing
        refresh = st.button("🔄 Refresh", use_container_width=True)

    # Re-fetch if user clicked refresh or changed the limit (optional: tie to refresh only)
    if refresh:
        st.session_state.cached_entries = _fetch_entries(limit=limit)
        base_rows = st.session_state.cached_entries or []

    # ---------- Apply filters locally ----------
    rows = base_rows
    if sel_category != "All":
        rows = [r for r in rows if str(r.get("category", "")).strip() == sel_category]
    if sel_status != "All":
        rows = [r for r in rows if str(r.get("status", "")).strip() == sel_status]

    # ---------- Table ----------
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(rows)} row(s).")
    else:
        st.info("No entries found with current filters.")
