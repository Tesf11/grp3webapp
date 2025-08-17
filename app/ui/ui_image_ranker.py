# app/ui/ui_image_ranker.py - Isaac
from __future__ import annotations
from pathlib import Path
import io, os, json, base64
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import joblib
import torch

# --- .env + Gemini ---
from dotenv import load_dotenv
import google.generativeai as genai

# ---- DB ----
from app.db import get_session, create_all
from app.models import ImageRankBatch, ImageRankItem, ImageAltText

# ---- SQLAlchemy eager loading
from sqlalchemy.orm import selectinload

# ---------- Paths ----------
HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parent
CANDIDATES = [
    APP_DIR / "image_ranker",
    APP_DIR / "models" / "image_ranker",
]
BUNDLE_DIR = None
for p in CANDIDATES:
    if (p / "model.pkl").exists() or (p / "model.joblib").exists():
        BUNDLE_DIR = p
        break
if BUNDLE_DIR is None:
    st.error("Couldn't find model bundle. Expected at app/image_ranker/ or app/models/image_ranker/")
    st.stop()

MODEL_PATH = (BUNDLE_DIR / "model.pkl") if (BUNDLE_DIR / "model.pkl").exists() else (BUNDLE_DIR / "model.joblib")
META_PATH  = BUNDLE_DIR / "metadata.json"

# Ensure tables exist (safe to call multiple times)
create_all()


# ---------- Loaders (cached) ----------
@st.cache_resource
def load_model_and_meta():
    clf = joblib.load(MODEL_PATH)
    meta = json.loads(Path(META_PATH).read_text())
    feat_cols = meta.get("feat_cols", [f"feat{i+1}" for i in range(512)])
    return clf, feat_cols, meta


@st.cache_resource
def load_clip_featurizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prefer original OpenAI CLIP; fallback to open_clip if unavailable
    try:
        import clip  # pip install git+https://github.com/openai/CLIP.git
        # If you bundled weights locally, you can pass download_root=...
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()

        def featurize(pil_list):
            xs = [preprocess(img).unsqueeze(0).to(device) for img in pil_list]
            x  = torch.cat(xs, dim=0)
            with torch.no_grad():
                feats = model.encode_image(x)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.float().cpu().numpy()
        return featurize

    except Exception:
        import open_clip  # pip install open_clip_torch
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device
        )
        model.eval()

        def featurize(pil_list):
            x = torch.stack([preprocess(img) for img in pil_list]).to(device)
            with torch.no_grad():
                feats = model.encode_image(x)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.float().cpu().numpy()
        return featurize


# ---------- Gemini ----------
@st.cache_resource
def _configure_gemini():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    model_name = "gemini-1.5-flash"
    model = genai.GenerativeModel(model_name)
    return {"model": model, "name": model_name}


def _pil_to_part(pil_img: Image.Image) -> dict:
    """Encode PIL image to base64 JPEG part for Gemini."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"inline_data": {"mime_type": "image/jpeg", "data": b64}}


def _gen_alt_text(model_dict, pil_image: Image.Image, extra_context: str | None = None) -> str:
    """Call Gemini to generate concise, product-style alt text."""
    if not model_dict:
        raise RuntimeError("Gemini not configured (GEMINI_API_KEY missing).")

    model = model_dict["model"]
    sys_prompt = (
        "You generate concise, factual ALT text for e-commerce product images. "
        "Focus on the main product, color/material, and key visible attributes. "
        "Avoid salesy language. One sentence, max 20 words."
    )
    if extra_context:
        sys_prompt += f"\nContext: {extra_context.strip()}"

    resp = model.generate_content([_pil_to_part(pil_image), sys_prompt])
    text = (getattr(resp, "text", "") or "").strip()
    return text if text else "Product image."


# ---------- Delete Helper ----------
def _delete_batch(bid: int) -> bool:
    """Delete a batch and its children (items, alt texts)."""
    try:
        with get_session() as s:
            s.query(ImageAltText).filter(ImageAltText.batch_id == bid).delete(synchronize_session=False)
            s.query(ImageRankItem).filter(ImageRankItem.batch_id == bid).delete(synchronize_session=False)
            b = s.get(ImageRankBatch, bid)
            if not b:
                return False
            s.delete(b)
            return True
    except Exception as e:
        st.error(f"Delete failed: {e}")
        return False


# ---------- Saved Batches Viewer ----------
def render_saved_batches(section_title="📚 Saved Batches", limit: int | None = None):
    st.header(section_title)

    if st.button("🔄 Refresh list"):
        st.rerun()

    # Eager-load items and build plain dict payload BEFORE session closes
    with get_session() as s:
        q = (s.query(ImageRankBatch)
               .options(selectinload(ImageRankBatch.items))
               .order_by(ImageRankBatch.id.desc()))
        batches = q.limit(limit).all() if limit else q.all()

        payload = []
        # Preload any alt text for each batch
        alt_map = {}
        if batches:
            batch_ids = [b.id for b in batches]
            alts = (s.query(ImageAltText)
                      .filter(ImageAltText.batch_id.in_(batch_ids))
                      .all())
            for a in alts:
                alt_map.setdefault(a.batch_id, []).append(a)

        for b in batches:
            items_sorted = sorted(
                b.items, key=lambda it: (it.rank if it.rank is not None else 1_000_000)
            )
            payload.append({
                "id": b.id,
                "item_name": b.item_name,
                "best_name": b.best_name,
                "best_score": float(b.best_score),
                "created_at": b.created_at,
                "model_kind": b.model_kind,
                "model_path": b.model_path,
                "items": [
                    {"rank": it.rank, "file_name": it.file_name, "score": float(it.score), "is_best": bool(it.is_best)}
                    for it in items_sorted
                ],
                "alt_texts": [
                    {"id": a.id, "text": a.alt_text, "model": a.model, "created_at": a.created_at}
                    for a in alt_map.get(b.id, [])
                ],
            })

    if not payload:
        st.info("No batches saved yet.")
        return

    # Summary table
    df = pd.DataFrame([{
        "id": p["id"],
        "item_name": p["item_name"],
        "best_name": p["best_name"],
        "best_score": round(p["best_score"], 3),
        "n_items": len(p["items"]),
        "created_at": p["created_at"],
    } for p in payload])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Details + delete buttons
    for p in payload:
        with st.expander(f"Batch #{p['id']} — {p['item_name']} · Best: {p['best_name']} ({p['best_score']:.3f})"):
            st.caption(f"Model: {p['model_kind'] or '-'} · Path: {p['model_path'] or '-'} · Created: {p['created_at']}")

            if p.get("alt_texts"):
                st.markdown("**Alt text**")
                for a in p["alt_texts"]:
                    st.write(f"- {a['text']}  _(model: {a['model']}, {a['created_at']})_")
            else:
                st.caption("No alt text saved for this batch.")

            df_items = pd.DataFrame(p["items"]).sort_values("rank", na_position="last")
            st.dataframe(df_items, use_container_width=True, hide_index=True)

            # Delete controls (with confirmation)
            key_prefix = f"del_{p['id']}"
            if st.button(f"🗑️ Delete batch #{p['id']}", key=f"{key_prefix}_ask"):
                st.session_state[f"{key_prefix}_confirm"] = True
                st.rerun()  # <-- was st.experimental_rerun()

            if st.session_state.get(f"{key_prefix}_confirm", False):
                st.warning("This will permanently delete the batch and its items (and alt texts).", icon="⚠️")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Confirm delete", key=f"{key_prefix}_confirm_btn"):
                        if _delete_batch(p["id"]):
                            st.success(f"Deleted batch #{p['id']} ✅")
                            st.session_state.pop(f"{key_prefix}_confirm", None)
                            st.rerun()
                        else:
                            st.warning("Batch not found.")
                with c2:
                    if st.button("Cancel", key=f"{key_prefix}_cancel"):
                        st.session_state.pop(f"{key_prefix}_confirm", None)
                        st.rerun()

# ---------- Main UI ----------
def render_image_ranker_ui():
    st.title("🖼️ Image Ranker (Best Image Picker)")
    st.caption(f"Model: {MODEL_PATH.relative_to(APP_DIR.parent)}")

    # Existing batches first
    render_saved_batches(section_title="📚 Saved Batches (existing)")

    st.divider()
    st.subheader("Rank a new set of images")

    clf, feat_cols, meta = load_model_and_meta()
    featurize = load_clip_featurizer()
    gem = _configure_gemini()  # may be None if no key

    uploaded = st.file_uploader(
        "Upload images of the **same item** (2+ images)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
    )

    if not uploaded:
        st.info("Select multiple images to rank, or just browse existing batches above.")
        return

    # Read as PIL
    images, names = [], []
    for f in uploaded:
        try:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
            images.append(img)
            names.append(f.name)
        except Exception as e:
            st.warning(f"Skipping {f.name}: {e}")

    if len(images) < 2:
        st.warning("Please upload at least 2 images of the same item.")
        return

    with st.spinner("Extracting CLIP features..."):
        X = featurize(images)  # (N, 512)

    # Score with classifier
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X)[:, 1]
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
        smin, smax = float(scores.min()), float(scores.max())
        scores = (scores - smin) / (smax - smin + 1e-9)
    else:
        scores = clf.predict(X).astype(float)

    best_idx = int(np.argmax(scores))
    best_name = names[best_idx]
    best_score = float(scores[best_idx])
    best_image = images[best_idx]

    st.subheader("Results")
    cols = st.columns(min(4, len(images)))
    for i, (img, name, s) in enumerate(zip(images, names, scores)):
        with cols[i % len(cols)]:
            st.image(img, caption=f"{name}\nScore: {s:.3f}", use_container_width=True)
            if i == best_idx:
                st.success("Best", icon="🏆")

    st.markdown(f"**Selected best:** `{best_name}`  (score={best_score:.3f})")

    # --- Alt-text generation (Gemini) ---
    st.divider()
    st.subheader("Generate ALT text (Gemini)")
    if gem is None:
        st.info("GEMINI_API_KEY not found in environment. Add it to your .env to enable this feature.")
    else:
        extra_context = st.text_input("Optional product title/notes (to guide alt text)", value="")
        if st.button("✨ Generate ALT text from best image"):
            with st.spinner("Calling Gemini..."):
                try:
                    alt_text = _gen_alt_text(gem, best_image, extra_context)
                    st.session_state["latest_alt_text"] = alt_text
                except Exception as e:
                    st.error(f"Alt text generation failed: {e}")
                    st.session_state["latest_alt_text"] = ""

        if st.session_state.get("latest_alt_text"):
            st.text_area("Generated ALT text", value=st.session_state["latest_alt_text"], height=80)

    # Optional: JSON download
    result = {
        "best_index": best_idx,
        "best_name": best_name,
        "scores": {n: float(s) for n, s in zip(names, scores)},
        "model_kind": meta.get("model_kind"),
    }
    st.download_button("Download result JSON", data=json.dumps(result, indent=2), file_name="image_ranker_result.json")

    # ====== Save to DB form (batch + items) ======
    st.divider()
    st.subheader("Save this result to the database")

    default_item_name = Path(best_name).stem
    item_name = st.text_input("Item / Batch name", value=default_item_name, help="Short label for this set of images (e.g., SKU)")

    if st.button("Save to database", type="primary"):
        if not item_name.strip():
            st.error("Please provide an Item / Batch name.")
        else:
            try:
                order = np.argsort(-scores)  # descending
                rank_of = {int(i): int(r) for r, i in enumerate(order)}

                with get_session() as s:
                    # 1) Create batch
                    batch = ImageRankBatch(
                        item_name=item_name.strip(),
                        best_index=best_idx,
                        best_name=best_name,
                        best_score=best_score,
                        model_kind=meta.get("model_kind"),
                        model_path=str(MODEL_PATH),
                    )
                    s.add(batch)
                    s.flush()  # get batch.id

                    # 2) Create items
                    best_item_id = None
                    for i, (fname, sc) in enumerate(zip(names, scores)):
                        it = ImageRankItem(
                            batch_id=batch.id,
                            file_name=fname,
                            score=float(sc),
                            is_best=(i == best_idx),
                            rank=rank_of[i],
                        )
                        s.add(it)
                        s.flush()
                        if i == best_idx:
                            best_item_id = it.id

                    # 3) If ALT text generated, save it
                    alt_text = st.session_state.get("latest_alt_text", "").strip()
                    if alt_text:
                        s.add(ImageAltText(
                            batch_id=batch.id,
                            item_id=best_item_id,
                            alt_text=alt_text,
                            provider="gemini",
                            model=gem["name"] if gem else "gemini",
                        ))

                    bid = batch.id

                st.success(f"Saved batch #{bid} with {len(names)} images ✅")
                # Clear alt text after save to avoid reusing it unintentionally
                st.session_state.pop("latest_alt_text", None)
                st.rerun()

            except Exception as e:
                st.error(f"DB save failed: {e}")


# Allow running this file directly for quick testing
if __name__ == "__main__":
    render_image_ranker_ui()