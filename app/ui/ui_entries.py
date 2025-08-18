# app/ui/ui_entries.py - Jialong
import os
import requests
import streamlit as st
from typing import List, Dict
import pandas as pd
import numpy as np
from fpdf import FPDF  # <-- ADDED: We will use this for the PDF download button

GENAI_AGG_URL_DEFAULT = "http://127.0.0.1:5000/api/metrics/genai"

def _fetch_genai_agg(days: int = 7):
    url = os.getenv("GENAI_AGG_URL", GENAI_AGG_URL_DEFAULT)
    r = requests.get(url, params={"days": days}, timeout=15)
    r.raise_for_status()
    return r.json()

def render_label_card(plan: dict, code: str):
    import re

    headline = str(plan.get("headline", "")).strip()
    subhead  = [str(x).strip() for x in (plan.get("subhead") or []) if str(x).strip()]
    bullets  = [str(x).strip() for x in (plan.get("bullets") or []) if str(x).strip()]

    # pull category + other context from the plan’s qr_payload if present
    qp = plan.get("qr_payload") or {}
    category     = str(qp.get("category", "")).strip()
    sample_type  = str(qp.get("sample_type", "") or plan.get("sample_type", "")).strip()
    status       = str(qp.get("status", "") or plan.get("status", "")).strip()

    # --- 1) Filter out quantities/packs like “20 pcs”, “x10”, “500ml”, “2kg”, etc.
    qty_re = re.compile(
        r"(?i)\b(\d+(\.\d+)?)\s*(pcs?|pieces?|pack|packs?|ct|count|x|×|kg|g|mg|l|ml)\b"
    )
    clean_bullets = [b for b in bullets if not qty_re.search(b)]

    # optional: keep at least 1 bullet if we removed all
    if bullets and not clean_bullets:
        clean_bullets = [bullets[0]]

    # chips row
    chips = []
    if sample_type: chips.append(sample_type)
    if status:      chips.append(status)

    # build HTML
    html = f"""
    <div style="border:1px solid #ddd;border-radius:12px;padding:14px 16px;max-width:560px;">
      <div style="font-weight:800;font-size:20px;line-height:1.1;margin-bottom:2px;">{headline}</div>
      {f'<div style="font-size:12px;color:#666;margin:2px 0 8px 0;"><b>Category:</b> {category}</div>' if category else ''}

      {''.join(f'<div style="font-size:14px;color:#444;margin:2px 0;">{s}</div>' for s in subhead)}

      {"".join([
        '<hr style="border:none;border-top:1px solid #eee;margin:8px 0;">',
        '<ul style="margin:0 0 8px 18px;padding:0;">',
        ''.join(f'<li style="margin:3px 0;font-size:14px;">{b}</li>' for b in clean_bullets),
        '</ul>'
      ]) if clean_bullets else ''}

      {"".join([
        '<div style="margin-top:6px;">',
        ''.join(f'<span style="display:inline-block;background:#f2f2f2;border:1px solid #e5e5e5;border-radius:999px;padding:4px 10px;margin-right:6px;font-size:12px;color:#444;">{c}</span>' for c in chips),
        '</div>'
      ]) if chips else ''}

      <div style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;margin-top:10px;">{code}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


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
    import io, zipfile  # local imports used for bundling files
    from app.ui.ui_sample_form import render_sample_form

    st.header("📚 View Entries")

    # ---------- UI state ----------
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

    # ---------- Fetch (ensure we have data to build dropdowns) ----------
    if "cached_entries" not in st.session_state:
        st.session_state.cached_entries = _fetch_entries(limit=100)

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

        df_orig = pd.DataFrame(rows)
        # Rename category -> type if exists (so table column is "type")
        if "category" in df_orig.columns:
            df_orig = df_orig.rename(columns={"category": "type"})
        # Ensure id exists
        if "id" not in df_orig.columns:
            df_orig.insert(0, "id", range(1, len(df_orig) + 1))

        # Add action checkboxes if missing
        df = df_orig.copy()
        if "_delete" not in df.columns:
            df["_delete"] = False
        if "_tag" not in df.columns:
            df["_tag"] = False

        # Which columns can the user edit
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
                options=["Received", "Sent", "In transit", "Pending"],
                required=False,
                width="medium",
            )
        # type dropdown (renamed category)
        if "type" in df.columns:
            type_opts = sorted({t for t in df["type"].dropna().unique() if t}) or ["uncategorized"]
            col_cfg["type"] = st.column_config.SelectboxColumn(
                "type",
                options=type_opts,
                required=False,
                width="medium",
            )

        # Action checkboxes
        col_cfg["_delete"] = st.column_config.CheckboxColumn("delete", help="Mark for deletion", width="small")
        col_cfg["_tag"]    = st.column_config.CheckboxColumn("tag", help="Mark for tag generation", width="small")

        # Label language selector (shared for generate)
        lang = st.selectbox("Label language", ["en", "ms", "zh"], index=0, key="tag_lang_select")

        # Action bar
        ab1, ab2, ab3, _sp = st.columns([1.2, 1.2, 1.6, 3])
        with ab1:
            save_changes = st.button("💾 Save changes", use_container_width=True)
        with ab2:
            delete_selected = st.button("🗑 Delete selected", use_container_width=True)
        with ab3:
            btn_generate_tags = st.button("🏷 Generate tags", use_container_width=True)

        # Editable table
        edited = st.data_editor(
            df,
            hide_index=True,
            use_container_width=True,
            column_config=col_cfg,
            disabled=["id"],  # id is read-only; checkboxes remain editable
            key="entries_editor",
        )

        # ==== Save edits ====
        
    if save_changes:
        try:
            ACTION_COLS = {"_delete", "_tag"}
            READONLY_COLS = {"id"}

            # Only compare real, editable DB fields
            editable_cols = [c for c in df.columns if c not in (READONLY_COLS | ACTION_COLS)]

            # Columns to pull from the edited grid
            cmp_cols_left = ["id"] + editable_cols

            # Columns to pull from the original (it doesn't have action cols)
            cmp_cols_right = ["id"] + [c for c in editable_cols if c in df_orig.columns]

            left = edited[cmp_cols_left].copy()
            right = df_orig[cmp_cols_right].copy()

            # Suffix right-hand columns so we can compare col-by-col
            # Keep 'id' unsuffixed to merge on it.
            right.columns = [
                (c if c == "id" else f"{c}_old") for c in right.columns
            ]

            # Merge on id
            left = left.merge(right, on="id", how="left")

            changed_rows = []
            for _, row in left.iterrows():
                rid = int(row["id"])
                diffs = {}

                for c in editable_cols:
                    newv = row.get(c)
                    oldv = row.get(f"{c}_old")

                    # Normalize numpy scalars
                    if hasattr(newv, "item"): newv = newv.item()
                    if hasattr(oldv, "item"): oldv = oldv.item()

                    # Normalize dates
                    if hasattr(newv, "isoformat"): newv = newv.isoformat()
                    if hasattr(oldv, "isoformat"): oldv = oldv.isoformat()

                    # Treat NaN == NaN
                    same = (
                        (pd.isna(newv) and pd.isna(oldv)) or
                        (newv == oldv)
                    )
                    if not same:
                        diffs[c] = newv

                if diffs:
                    changed_rows.append((rid, diffs))

            if not changed_rows:
                st.info("No changes to save.")
            else:
                update_url_tmpl = os.getenv("UPDATE_API_URL_TMPL", "http://127.0.0.1:5000/api/update/{}")
                ok = 0
                for rid, diffs in changed_rows:
                    # Optional: map UI alias 'type' -> API 'category' here if your backend didn’t already
                    # if "type" in diffs and "category" not in diffs:
                    #     diffs["category"] = diffs.pop("type")

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
                to_delete = edited.loc[edited["_delete"] == True, "id"].tolist()  # noqa: E712
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

        # ==== Tag generation (multi-row) ====
        if btn_generate_tags:
            GEN_SMART_TAG_URL = os.getenv(
                "GEN_SMART_TAG_URL",
                "http://127.0.0.1:5000/api/generate_smart_tag"
            )

            ids_for_tag = edited.loc[edited["_tag"] == True, "id"].tolist()  # noqa: E712
            if not ids_for_tag:
                st.info("No rows marked for tag generation.")
            else:
                st.write(f"Generating tags for {len(ids_for_tag)} row(s)...")

                def _row_to_payload(row: pd.Series) -> dict:
                    d = row.to_dict()
                    clean = {}
                    for k, v in d.items():
                        # remove action flags from payload
                        if k in ("_delete", "_tag"):
                            continue
                        # normalize numpy / NaN / dates
                        if isinstance(v, (np.generic,)):
                            v = v.item()
                        if isinstance(v, float) and (pd.isna(v)):
                            v = None
                        if hasattr(v, "isoformat"):
                            v = v.isoformat()
                        clean[k] = v
                    return clean

                edited_by_id = {int(rid): edited.loc[edited["id"] == rid].iloc[0] for rid in ids_for_tag}
                prog = st.progress(0.0)
                results = []
                total = len(ids_for_tag)

                bundle = []  # for zip/pdf collection

                for i, rid in enumerate(ids_for_tag, start=1):
                    row_series = edited_by_id[int(rid)]
                    payload = _row_to_payload(row_series)
                    payload["lang"] = lang  # pass language to backend

                    try:
                        resp = requests.post(GEN_SMART_TAG_URL, json=payload, timeout=45)
                        if resp.status_code == 200:
                            data = resp.json()
                            results.append((rid, True, data))
                            # collect file bytes for bundling
                            zpl = (data.get("zpl") or "").strip()
                            plain = (data.get("label_text") or "").strip()
                            md_preview = (data.get("markdown_preview") or "").strip()
                            code = data.get("short_code", f"tag_{rid}")
                            file_bytes = (zpl + "\n").encode("utf-8") if zpl else ((plain or md_preview) + "\n").encode("utf-8")
                            ext = "zpl" if zpl else "txt"
                            bundle.append((f"{code}.{ext}", file_bytes))
                        else:
                            results.append((rid, False, f"HTTP {resp.status_code}: {resp.text}"))
                    except requests.RequestException as e:
                        results.append((rid, False, str(e)))
                    finally:
                        prog.progress(i / total)

                st.divider()
                st.subheader("🔖 Tag Previews")

                any_ok = False
                for rid, ok, data in results:
                    with st.expander(f"Row ID {rid} — {'✅ OK' if ok else '❌ Failed'}", expanded=not ok):
                        if ok:
                            plan = data.get("plan") or {}
                            code = data.get("short_code", f"tag_{rid}")
                            render_label_card(plan, code)

                            # Also offer individual downloads again for convenience
                            zpl = (data.get("zpl") or "").strip()
                            plain = (data.get("label_text") or "").strip()
                            md_preview = (data.get("markdown_preview") or "").strip()

                            cdl1, cdl2 = st.columns(2)
                            with cdl1:
                                st.download_button(
                                    label=f"⬇️ ZPL ({code}.zpl)",
                                    data=(zpl + "\n").encode("utf-8"),
                                    file_name=f"{code}.zpl",
                                    mime="text/plain",
                                    use_container_width=True,
                                    disabled=(not zpl),
                                )
                            with cdl2:
                                st.download_button(
                                    label=f"⬇️ Plain text ({code}.txt)",
                                    data=(plain + "\n").encode("utf-8") if plain else (md_preview + "\n").encode("utf-8"),
                                    file_name=f"{code}.txt",
                                    mime="text/plain",
                                    use_container_width=True,
                                )
                            any_ok = True
                        else:
                            st.error(str(data))

                # ---- Bulk download buttons (ZIP + PDF) ----
                if any_ok and bundle:
                    # Build ZIP now (zip_buf exists here)
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                        for fname, fbytes in bundle:
                            zf.writestr(fname, fbytes)
                    zip_buf.seek(0)

                    cz1, cz2 = st.columns([1, 1])
                    with cz1:
                        st.download_button(
                            label="📦 Download ALL tags (.zip)",
                            data=zip_buf.getvalue(),
                            file_name="tags_bundle.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )

                    with cz2:
                        # Build a single PDF containing all tag contents (one page per tag)
                        try:
                            pdf = FPDF()
                            pdf.set_auto_page_break(auto=True, margin=15)

                            def _latin1_safe(s: str) -> str:
                                return s.encode("latin-1", "replace").decode("latin-1")

                            for fname, fbytes in bundle:
                                pdf.add_page()
                                try:
                                    pdf.set_font("Helvetica", size=12)
                                except Exception:
                                    pdf.set_font("Arial", size=12)
                                pdf.multi_cell(0, 8, txt=_latin1_safe(fname))
                                pdf.ln(4)

                                try:
                                    content = fbytes.decode("utf-8", errors="replace")
                                except Exception:
                                    content = fbytes.hex()

                                try:
                                    pdf.set_font("Courier", size=10)
                                except Exception:
                                    pdf.set_font("Helvetica", size=10)

                                for line in content.splitlines():
                                    # keep lines manageable for the page width
                                    chunks = [line[i:i+120] for i in range(0, len(line), 120)] or [""]
                                    for ch in chunks:
                                        pdf.multi_cell(0, 5, txt=_latin1_safe(ch))
                                pdf.ln(2)

                            pdf_out = pdf.output(dest="S")
                            pdf_bytes = pdf_out if isinstance(pdf_out, (bytes, bytearray)) else pdf_out.encode("latin-1", "replace")

                            st.download_button(
                                label="📄 Download ALL tags (PDF)",
                                data=pdf_bytes,
                                file_name="tags_bundle.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                        except Exception as e:
                            st.error(f"Failed to build PDF: {e}")

                st.divider()

    else:
        st.info("No entries found with current filters.")

    # ================================
    # 🤖 GenAI Metrics (on-demand)
    # ================================
    st.markdown("### 🤖 GenAI Metrics")
    g1, g2, _g3 = st.columns([1, 1, 3])
    with g1:
        days = st.number_input("Window (days)", min_value=1, max_value=90, value=7, step=1)
    with g2:
        show = st.button("Show metrics", type="primary", use_container_width=True)

    if show:
        try:
            agg = _fetch_genai_agg(days=days)
        except Exception as e:
            st.error(f"Failed to fetch GenAI metrics: {e}")
        else:
            # KPIs
            k1, k2, k3, k4 = st.columns(4)
            total   = agg["counts"]["total"]
            ok      = agg["counts"]["ok"]
            blocked = agg["counts"]["blocked"]
            errors  = agg["counts"]["errors"]
            with k1: st.metric("Requests", total)
            with k2: st.metric("Success rate", f"{(ok/max(1,total))*100:.1f}%")
            with k3: st.metric("Blocked", blocked)
            with k4: st.metric("Errors", errors)

            l1, l2 = st.columns(2)
            with l1: st.metric("Avg latency (ms)", f"{agg['latency_ms']['avg']:.0f}")
            with l2:
                p95 = agg["latency_ms"].get("p95")
                st.metric("p95 latency (ms)", f"{p95:.0f}" if p95 else "N/A")

            t1, t2, t3 = st.columns(3)
            with t1: st.metric("Input tokens",  agg["tokens"]["input"])
            with t2: st.metric("Output tokens", agg["tokens"]["output"])
            with t3: st.metric("Total tokens",  agg["tokens"]["total"])

            # By model (if multiple)
            by_model = agg.get("by_model", [])
            if by_model:
                st.markdown("**Requests by model**")
                st.dataframe(pd.DataFrame(by_model), use_container_width=True, hide_index=True)

            st.markdown("**Interpretation tips**")
            st.caption("- Rising **blocked** suggests inputs trigger safety → tighten prompts/inputs.")
            st.caption("- High **p95 latency** hurts UX → shorter prompts or a smaller model.")
            st.caption("- **Token totals** drive cost → watch verbose outputs.")
    else:
        st.info("Click **Show metrics** to compute usage, latency, success/blocks and token totals.")



    
