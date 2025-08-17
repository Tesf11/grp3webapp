# app/ui/ui_StoragePredictorGenai.py - Tesfay
import os
import json
import time
from typing import List, Optional, Dict, Any

import streamlit as st
import google.generativeai as genai
from io import BytesIO
from fpdf import FPDF


# ------------------------------
# Core GenAI logic
# ------------------------------
MODEL_NAME_DEFAULT = os.getenv("GEN_MODEL_NAME", "gemini-2.5-flash")  # or "gemini-2.5-pro"
FIXED_TEMPERATURE = 0.7  # fixed creativity (no UI control)
MAX_ITEMS = 5            # cap list lengths

SYSTEM_RULES = (
    "You are a product ideation assistant for ANS Import & Export (cleaning & home-care focus).\n"
    "- DO NOT predict storage bins or product_type; a separate model handles that.\n"
    "- Focus ONLY on: possible cleaning products and features/benefits.\n"
    "- Ground outputs in household/commercial cleaning contexts (surfaces, usage, safety, dilution, fragrance, residue).\n"
    "- Be concise, safety-aware, and practical for B2B.\n"
)

SCHEMA_HINT = {
    "possible_products": ["string", "..."],
    "features": ["string", "..."]
}

def _get_key() -> Optional[str]:
    return os.getenv("GOOGLE_API_KEY")

def _prompt(description: str) -> str:
    return f"""{SYSTEM_RULES}

Return JSON ONLY with this exact shape:
{json.dumps(SCHEMA_HINT, ensure_ascii=False)}

Sample description:
\"\"\"{description.strip()}\"\"\""""

def _try_json(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    try:
        return json.loads(t)
    except Exception:
        try:
            s = t[t.find("{"): t.rfind("}") + 1]
            return json.loads(s)
        except Exception:
            return {"possible_products": [], "features": []}

def generate_ideas(
    description: str,
    extra_tags: Optional[List[str]] = None,
    model_name: str = MODEL_NAME_DEFAULT,
    temperature: float = FIXED_TEMPERATURE,
    max_retries: int = 2,
    timeout_s: int = 25,
) -> Dict[str, Any]:
    if not description or not description.strip():
        return {"possible_products": [], "features": []}

    api_key = _get_key()
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY environment variable")

    genai.configure(api_key=api_key)
    prompt = _prompt(description)

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(
                prompt,
                generation_config={
                    "temperature": float(temperature),
                    "response_mime_type": "application/json",
                },
                request_options={"timeout": int(timeout_s)},
            )
            text = (resp.text or "").strip()
            ideas = _try_json(text)

            possible_products = ideas.get("possible_products", []) or []
            features = ideas.get("features", []) or []

            if extra_tags:
                tags = [t.strip() for t in extra_tags if t and t.strip()]
                # dedupe while preserving order
                features = list(dict.fromkeys(features + tags))

            return {
                "possible_products": possible_products,
                "features": features,
            }
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(0.7 * (attempt + 1))
            else:
                raise last_err


# ------------------------------
# Backend or local call
# ------------------------------
def _call_backend_or_local(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    If GEN_API_URL is set, POST to Flask (/api/genideas).
    Otherwise call generate_ideas(...) locally with GOOGLE_API_KEY.
    """
    api_url = os.getenv("GEN_API_URL")
    if api_url:
        try:
            import requests
            r = requests.post(api_url, json=payload, timeout=45)
            r.raise_for_status()
            data = r.json()
            ideas = data.get("ideas", data) or {}
            return {
                "possible_products": ideas.get("possible_products", []) or [],
                "features": ideas.get("features", []) or [],
            }
        except Exception as e:
            st.warning(f"Backend unavailable, using local Gemini call. ({e})")

    ideas = generate_ideas(
        description=payload.get("description", ""),
        extra_tags=payload.get("tags", []),
        model_name=os.getenv("GEN_MODEL_NAME", MODEL_NAME_DEFAULT),
        temperature=FIXED_TEMPERATURE,
    )
    return {
        "possible_products": ideas.get("possible_products", []) or [],
        "features": ideas.get("features", []) or [],
    }


# ------------------------------
# Presentation helpers
# ------------------------------
def _one_liner(ideas: Dict[str, Any]) -> str:
    p = (ideas.get("possible_products") or [])
    f = (ideas.get("features") or [])
    if not p:
        return ""
    head = p[0]
    feat = ", ".join([x for x in f if x][:2]) if f else ""
    return f"{head}" + (f" — {feat}" if feat else "")

def _render_generated_ideas(ideas: Dict[str, Any], max_items: int = MAX_ITEMS):
    """
    Single clean tab (main): Generated Product Ideas
      1) Paired list: Product — Top feature (1:1 by index)
      2) Expanders to view full Products and full Features
    """
    tab, = st.tabs(["🧩 Generated Product Ideas"])
    with tab:
        prods = [s for s in (ideas.get("possible_products") or []) if str(s).strip()][:max_items]
        feats = [s for s in (ideas.get("features") or []) if str(s).strip()][:max_items]

        if not prods and not feats:
            st.info("No ideas yet.")
            return

        # Primary: quick pairs
        if prods and feats:
            st.markdown("**Product → Top Feature**")
            n = min(len(prods), len(feats))
            for i in range(n):
                st.markdown(f"**{i+1}. {prods[i]}** — {feats[i]}")
        elif prods:
            st.markdown("**Product Ideas**")
            for i, p in enumerate(prods, 1):
                st.markdown(f"**{i}. {p}**")
        elif feats:
            st.markdown("**Features & Benefits**")
            for f in feats:
                st.markdown(f"- {f}")

        # Secondary: full lists in expanders
        with st.expander("See all possible products"):
            if prods:
                for i, p in enumerate(prods, 1):
                    st.markdown(f"**{i}. {p}**")
            else:
                st.caption("_None_")

        with st.expander("See all features & benefits"):
            if feats:
                for f in feats:
                    st.markdown(f"- {f}")
            else:
                st.caption("_None_")


# ------------------------------
# Streamlined, chat-like UI
# ------------------------------
def render_storage_predictor_genai():
    """
    Streamlined UI for sample ideation (cleaning-only):
    - Read-only model badge at top
    - Chat input UX
    - One tab: 'Generated Product Ideas' with pairs first
    - Clears previous result on new prompt
    - Unique keys for download buttons to avoid duplicate-ID error
    """
    st.subheader("🧪 Generate Sample Ideas (Gemini)")
    st.caption(
        "This assistant **does not** predict storage bins. It only suggests "
        "**possible cleaning products** and **features/benefits** based on your description."
    )

    cols = st.columns([1, 3, 3])
    with cols[0]:
        st.text_input(
            "Model",
            value=os.getenv("GEN_MODEL_NAME", MODEL_NAME_DEFAULT),
            disabled=True,
            help="LLM used for ideation (fixed)."
        )
    with cols[1]:
        extra_tags_text = st.text_input(
            "Extra tags (optional)",
            placeholder="disinfectant, residue-free, pet-safe"
        )
    with cols[2]:
        st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    # --- Session state for single-result view ---
    if "sp_run" not in st.session_state:
        st.session_state.sp_run = 0              # increments every submit
    if "sp_last_result" not in st.session_state:
        st.session_state.sp_last_result = None   # latest ideas dict
    if "sp_last_summary" not in st.session_state:
        st.session_state.sp_last_summary = ""    # latest one-liner

    # Render the latest result (if any)
    if st.session_state.sp_last_result:
        ideas = st.session_state.sp_last_result
        ol = st.session_state.sp_last_summary
        if ol:
            st.markdown(f"**Summary:** {ol}")

        _render_generated_ideas(ideas, MAX_ITEMS)

        # Downloads with unique keys for this run
        colA, colB = st.columns(2)
        with colA:
            ()
        with colB:
            # Build PDF via FPDF (lightweight)
            pdf = FPDF()
            pdf.add_page()

            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Generative Ideas", ln=True)

            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Possible Products:", ln=True)
            pdf.set_font("Arial", "", 12)
            for i, p in enumerate(ideas.get("possible_products", []), 1):
                pdf.multi_cell(0, 8, f"{i}. {p}")

            pdf.ln(3)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Features & Benefits:", ln=True)
            pdf.set_font("Arial", "", 12)
            for f in ideas.get("features", []):
                pdf.multi_cell(0, 8, f"- {f}")

            pdf_bytes = BytesIO(pdf.output(dest="S").encode("latin1"))

            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name="Generative_Ideas.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"dl_pdf_{st.session_state.sp_run}",
            )

    # Chat input (new prompt)
    user_text = st.chat_input("Describe the cleaning sample (e.g., 'solid moth balls', 'citrus degreaser concentrate').")
    if user_text:
        # Clear previous result so there are no duplicate buttons
        st.session_state.sp_run += 1
        st.session_state.sp_last_result = None
        st.session_state.sp_last_summary = ""

        tags_list = [t.strip() for t in (extra_tags_text or "").split(",") if t.strip()]
        payload = {
            "description": user_text.strip(),
            "tags": tags_list,  # Extra tags ARE used
            "temperature": FIXED_TEMPERATURE,
            "model_name": os.getenv("GEN_MODEL_NAME", MODEL_NAME_DEFAULT),
        }

        with st.chat_message("user"):
            st.write(user_text)
        with st.chat_message("assistant"):
            with st.spinner("Generating ideas…"):
                ideas = _call_backend_or_local(payload)

                # Cache latest result + summary
                ol = _one_liner(ideas)
                st.session_state.sp_last_result = ideas
                st.session_state.sp_last_summary = ol or "Ideas"

                # Render immediately for this turn
                if ol:
                    st.markdown(f"**Summary:** {ol}")
                _render_generated_ideas(ideas, MAX_ITEMS)

                # Downloads with keys tied to current run
                colA, colB = st.columns(2)
                with colA:
                    ()
                with colB:
                    pdf = FPDF()
                    pdf.add_page()

                    pdf.set_font("Arial", "B", 16)
                    pdf.cell(0, 10, "Generative Ideas", ln=True)

                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 10, "Possible Products:", ln=True)
                    pdf.set_font("Arial", "", 12)
                    for i, p in enumerate(ideas.get("possible_products", []), 1):
                        pdf.multi_cell(0, 8, f"{i}. {p}")

                    pdf.ln(3)
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 10, "Features & Benefits:", ln=True)
                    pdf.set_font("Arial", "", 12)
                    for f in ideas.get("features", []):
                        pdf.multi_cell(0, 8, f"- {f}")

                    pdf_bytes = BytesIO(pdf.output(dest="S").encode("latin1"))

                    st.download_button(
                        "Download PDF",
                        data=pdf_bytes,
                        file_name="Generative_Ideas.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key=f"dl_pdf_{st.session_state.sp_run}",
                    )
