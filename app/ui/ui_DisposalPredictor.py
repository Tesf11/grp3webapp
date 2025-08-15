# app/ui/ui_DisposalPredictor.py
import os, json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

from app.disposal_loader import predict as disposal_predict
from app.db import get_session, create_all
from app.models import DisposalPrediction

# make sure tables exist (safe if called multiple times)
create_all()

# --- Gemini init (cached) ---
@st.cache_resource
def _gemini_model():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None  # UI will fall back to local text
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

# --- Minimal rules (kept) ---
DISPOSAL = {
    "aerosol_cans": {
        "pretty": "Aerosol can",
        "steps": [
            "Ensure the can is completely empty (no sound, no spray).",
            "Do not puncture or crush.",
            "If pressurized or with residue: take to hazardous-waste drop-off.",
            "If fully empty and local rules allow: recycle as metal."
        ],
        "cautions": [
            "Propellant can be flammable.",
            "Local rules vary; follow municipal guidance."
        ],
    },
    "plastic_detergent_bottles": {
        "pretty": "Plastic detergent bottle",
        "steps": [
            "Empty and rinse thoroughly.",
            "Leave cap on if your local program accepts caps; otherwise follow local instructions.",
            "Recycle as rigid plastic if clean; trash if significant residue remains."
        ],
        "cautions": [
            "Residue contaminates recycling.",
            "Check local acceptance for caps and labels."
        ],
    },
    "plastic_trash_bags": {
        "pretty": "Plastic trash bag (film plastic)",
        "steps": [
            "Do not place in curbside recycling.",
            "Dispose with general waste, or use store drop-off if film-plastic collection exists."
        ],
        "cautions": [
            "Film plastic jams sorting equipment.",
            "Keep bags out of mixed recycling streams."
        ],
    },
}

def _gen_guidance(label: str, prob: float, alt=None) -> str:
    rules = DISPOSAL.get(label)
    if rules is None:
        return f"Guidance unavailable for `{label}`. Check local regulations."

    mdl = _gemini_model()
    payload = {
        "label": label,
        "pretty": rules["pretty"],
        "confidence_pct": round(float(prob) * 100, 1),
        "rules": {"steps": rules["steps"], "cautions": rules["cautions"]},
    }
    if alt:
        payload["alternative"] = {
            "label": alt[0],
            "confidence_pct": round(float(alt[1]) * 100, 1),
        }

    if mdl is None:
        lines = [
            f"**Item:** {rules['pretty']}",
            f"**Confidence:** {payload['confidence_pct']}%",
            "**How to dispose:**",
            *[f"- {s}" for s in rules["steps"]],
            "**Cautions:**",
            *[f"- {c}" for c in rules["cautions"]],
        ]
        if alt:
            lines.append(f"*Low confidence. Alternative: {alt[0]} ({payload['alternative']['confidence_pct']}%).*")
        return "\n".join(lines)

    prompt = (
        "You are a recycling assistant. Using ONLY the provided rules, write concise guidance.\n"
        "Format:\n"
        "- First line: Item name and confidence.\n"
        "- Then 'How to dispose:' as bullets from the rules.\n"
        "- Then 'Cautions:' as bullets from the rules.\n"
        "Do NOT invent new rules. If an alternative label is provided, add one sentence noting low confidence.\n"
        f"INPUT_JSON:\n{json.dumps(payload)}"
    )
    try:
        resp = mdl.generate_content(prompt)
        return resp.text or "Guidance unavailable."
    except Exception:
        lines = [
            f"**Item:** {rules['pretty']}",
            f"**Confidence:** {payload['confidence_pct']}%",
            "**How to dispose:**",
            *[f"- {s}" for s in rules['steps']],
            "**Cautions:**",
            *[f"- {c}" for c in rules['cautions']],
        ]
        if alt:
            lines.append(f"*Low confidence. Alternative: {alt[0]} ({payload['alternative']['confidence_pct']}%).*")
        return "\n".join(lines)

def render_disposal_predictor_ui():
    st.header("🗑 Disposal Predictor")
    st.caption("Upload an image. The TensorFlow model in `app/models/disposal_predictor/` will classify it.")
    up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    col_img, col_info = st.columns([1.1, 1.4])

    if up is not None:
        img = Image.open(up).convert("RGB")
        col_img.image(img, caption="Uploaded", width=300)

        rgb = np.array(img)
        with st.spinner("Predicting..."):
            out = disposal_predict(rgb)

        # Display
        label = out["label"]
        prob  = float(out["prob"])
        cands = out.get("candidates", [])
        low_conf = bool(out.get("low_conf"))

        col_info.markdown(f"**Item:** {label}")
        col_info.markdown(f"**Confidence:** {prob*100:.1f}%")

        alt = None
        if low_conf and len(cands) > 1:
            alt = cands[1]
            col_info.warning(
                f"Low confidence. Top-2: "
                f"{cands[0][0]} ({cands[0][1]*100:.1f}%), "
                f"{cands[1][0]} ({cands[1][1]*100:.1f}%)"
            )

        guidance = _gen_guidance(
            cands[0][0] if low_conf and cands else label,
            cands[0][1] if low_conf and cands else prob,
            alt=alt
        )
        col_info.markdown("**Disposal guidance**")
        col_info.markdown(guidance)

        # ---- Save to DB prompt ----
        st.markdown("---")
        if st.button("💾 Save this prediction to database", type="primary", key="disposal_save_btn"):
            try:
                with get_session() as s:
                    row = DisposalPrediction(
                        file_name=getattr(up, "name", None),
                        label=label,
                        prob=prob,
                        low_conf=low_conf,
                        candidates_json=json.dumps(cands),
                        guidance=guidance,
                    )
                    s.add(row)
                    s.flush()              # assign PK
                    new_id = row.id        # capture while session is open
                st.success(f"Saved prediction #{new_id} ✅")
            except Exception as e:
                st.error(f"Save failed: {e}")


    # ---- Saved entries table ----
    st.divider()
    st.subheader("📚 Saved predictions")
    if st.button("🔄 Refresh list", key="disposal_refresh_list"):
        st.rerun()

    try:
        # Build plain dicts BEFORE closing the session
        with get_session() as s:
            q = (
                s.query(
                    DisposalPrediction.id,
                    DisposalPrediction.file_name,
                    DisposalPrediction.label,
                    DisposalPrediction.prob,
                    DisposalPrediction.low_conf,
                    DisposalPrediction.created_at,
                )
                .order_by(DisposalPrediction.id.desc())
            )
            rows = q.all()
            payload = [dict(r._mapping) for r in rows]  # Row -> plain dict

        if not payload:
            st.info("No saved predictions yet.")
        else:
            # format confidence
            for p in payload:
                p["confidence"] = round(float(p.pop("prob")), 3)
            df = pd.DataFrame(payload, columns=["id","file_name","label","confidence","low_conf","created_at"])
            st.dataframe(df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Could not load saved predictions: {e}")
