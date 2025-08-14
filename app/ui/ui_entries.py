# app/ui/ui_entries.py
import os
import requests
import streamlit as st
from typing import List, Dict
import pandas as pd
import numpy as np

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

    # ---------- Unified table ----------

    if rows:
        st.write(f"Showing {len(rows)} row(s).")

        # Build DF from filtered rows
        df_orig = pd.DataFrame(rows)

        # Rename category -> type if exists
        if "category" in df_orig.columns:
            df_orig = df_orig.rename(columns={"category": "type"})

        # Ensure id exists
        if "id" not in df_orig.columns:
            df_orig.insert(0, "id", range(1, len(df_orig) + 1))

        # Add checkbox columns
        df = df_orig.copy()
        if "_delete" not in df.columns:
            df["_delete"] = False
        if "_tag" not in df.columns:
            df["_tag"] = False

        # Lock id + action cols
        not_editable = {"id"}
        editable_cols = [c for c in df.columns if c not in not_editable]

        # Column config
        col_cfg = {}

        # sample_type dropdown
        if "sample_type" in df.columns:
            col_cfg["sample_type"] = st.column_config.SelectboxColumn(
                "sample_type",
                options=["Incoming", "Outgoing"],
                required=False,
                width="medium",
            )

        # status dropdown
        if "status" in df.columns:
            col_cfg["status"] = st.column_config.SelectboxColumn(
                "status",
                options=["Received", "Sent", "In transit"],
                required=False,
                width="medium",
            )

        # type dropdown (renamed category)
        if "type" in df.columns:
            # Use distinct values in data + example default if empty
            type_opts = sorted({t for t in df["type"].dropna().unique() if t}) or ["uncategorized"]
            col_cfg["type"] = st.column_config.SelectboxColumn(
                "type",
                options=type_opts,
                required=False,
                width="medium",
            )

        # Action checkboxes
        col_cfg["_delete"] = st.column_config.CheckboxColumn("delete", help="Mark for deletion", width="small")
        col_cfg["_tag"] = st.column_config.CheckboxColumn("tag", help="Mark for tag generation", width="small")

        # Action bar
        ab1, ab2, ab3, _sp = st.columns([1.2, 1.2, 1.6, 3])
        with ab1:
            save_changes = st.button("💾 Save changes", use_container_width=True)
        with ab2:
            delete_selected = st.button("🗑 Delete selected", use_container_width=True)
        with ab3:
            gen_tags = st.button("🏷 Generate tags", use_container_width=True)

        # Editable table
        edited = st.data_editor(
            df,
            hide_index=True,
            use_container_width=True,
            column_config=col_cfg,
            disabled=["id"],
            key="entries_editor",
        )

        # ==== Save edits ====
        if save_changes:
            try:
                cmp_cols = ["id"] + editable_cols
                left = edited[cmp_cols].copy()
                right = df_orig[cmp_cols].copy().add_suffix("_old")
                left = left.merge(right, left_on="id", right_on="id_old").drop(columns=["id_old"])

                changed_rows = []
                for _, row in left.iterrows():
                    rid = int(row["id"])
                    diffs = {}
                    for c in editable_cols:
                        newv, oldv = row[c], row[f"{c}_old"]
                        if isinstance(newv, (np.generic,)):
                            newv = newv.item()
                        if isinstance(oldv, (np.generic,)):
                            oldv = oldv.item()
                        if hasattr(newv, "isoformat"):
                            newv = newv.isoformat()
                        if hasattr(oldv, "isoformat"):
                            oldv = oldv.isoformat()
                        if not ((pd.isna(newv) and pd.isna(oldv)) or newv == oldv):
                            diffs[c] = newv
                    if diffs:
                        changed_rows.append((rid, diffs))

                if not changed_rows:
                    st.info("No changes to save.")
                else:
                    update_url_tmpl = os.getenv("UPDATE_API_URL_TMPL", "http://127.0.0.1:5000/api/update/{}")
                    ok = 0
                    for rid, diffs in changed_rows:
                        try:
                            resp = requests.put(update_url_tmpl.format(rid), json=diffs, timeout=10)
                            if resp.status_code == 200:
                                ok += 1
                            else:
                                st.error(f"Update {rid} failed: {resp.text}")
                        except requests.RequestException as e:
                            st.error(f"Update {rid} failed: {e}")
                    st.success(f"Saved {ok} row(s).")
                    st.session_state.cached_entries = _fetch_entries(limit=limit)
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to process changes: {e}")

        # ==== Delete ====
        if delete_selected:
            try:
                to_delete = edited.loc[edited["_delete"] == True, "id"].tolist()
                if not to_delete:
                    st.info("No rows marked for deletion.")
                else:
                    delete_url_tmpl = os.getenv("DELETE_API_URL", "http://127.0.0.1:5000/api/delete/{}")
                    ok = 0
                    for rid in to_delete:
                        try:
                            resp = requests.delete(delete_url_tmpl.format(int(rid)), timeout=10)
                            if resp.status_code == 200:
                                ok += 1
                            else:
                                st.error(f"Delete {rid} failed: {resp.text}")
                        except requests.RequestException as e:
                            st.error(f"Delete {rid} failed: {e}")
                    st.success(f"Deleted {ok} row(s).")
                    st.session_state.cached_entries = _fetch_entries(limit=limit)
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to delete: {e}")

        # ==== Tag generation ====
        if gen_tags:
            try:
                ids_for_tag = edited.loc[edited["_tag"] == True, "id"].tolist()
                if not ids_for_tag:
                    st.info("No rows marked for tag generation.")
                else:
                    st.info(f"Would generate tags for rows: {ids_for_tag}")
            except Exception as e:
                st.error(f"Failed to prepare tags: {e}")

    else:
        st.info("No entries found with current filters.")






