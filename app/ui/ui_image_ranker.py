# app/ui/ui_image_ranker.py
from __future__ import annotations
from pathlib import Path
import io, json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import joblib
import torch
from sqlalchemy.orm import selectinload

# ---- DB ----
from app.db import get_session, create_all
from app.models import ImageRankBatch, ImageRankItem

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


# ---------- Helpers ----------
def render_saved_batches(section_title="📚 Saved Batches", limit: int | None = None):
    """Show existing batches (with expanders for items)."""
    st.header(section_title)

    if st.button("🔄 Refresh list"):
        st.rerun()

    # 1) Query with eager load, and build plain payloads while the session is open
    with get_session() as s:
        q = (s.query(ImageRankBatch)
               .options(selectinload(ImageRankBatch.items))  # eager-load children
               .order_by(ImageRankBatch.id.desc()))
        batches = q.limit(limit).all() if limit else q.all()

        payload = []
        for b in batches:
            items_sorted = sorted(
                b.items,
                key=lambda it: (it.rank if it.rank is not None else 1_000_000)
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
                    {
                        "rank": it.rank,
                        "file_name": it.file_name,
                        "score": float(it.score),
                        "is_best": bool(it.is_best),
                    }
                    for it in items_sorted
                ],
            })

    # 2) Now outside the session: render from plain dicts
    if not payload:
        st.info("No batches saved yet.")
        return

    import pandas as pd
    summary_rows = [{
        "id": p["id"],
        "item_name": p["item_name"],
        "best_name": p["best_name"],
        "best_score": round(p["best_score"], 3),
        "n_items": len(p["items"]),
        "created_at": p["created_at"],
    } for p in payload]
    df = pd.DataFrame(summary_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    for p in payload:
        with st.expander(f"Batch #{p['id']} — {p['item_name']} · Best: {p['best_name']} ({p['best_score']:.3f})"):
            st.caption(f"Model: {p['model_kind'] or '-'} · Path: {p['model_path'] or '-'} · Created: {p['created_at']}")
            df_items = pd.DataFrame(p["items"]).sort_values("rank", na_position="last")
            st.dataframe(df_items, use_container_width=True, hide_index=True)


# ---------- UI ----------
def render_image_ranker_ui():
    st.title("🖼️ Image Ranker (Best Image Picker)")
    st.caption(f"Model: {MODEL_PATH.relative_to(APP_DIR.parent)}")

    # If we just saved a batch in a previous run, show a confirmation
    if "saved_bid" in st.session_state:
        st.success(f"Saved batch #{st.session_state['saved_bid']} ✅")
        del st.session_state["saved_bid"]

    # --- Show existing batches FIRST (no need to upload yet) ---
    render_saved_batches(section_title="📚 Saved Batches (existing)")

    st.divider()
    st.subheader("Rank a new set of images")

    clf, feat_cols, meta = load_model_and_meta()
    featurize = load_clip_featurizer()

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
        X = featurize(images)  # shape (N, 512)

    # Score with classifier
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X)[:, 1]
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
        smin, smax = float(scores.min()), float(scores.max())
        scores = (scores - smin) / (smax - smin + 1e-9)  # normalize for display
    else:
        scores = clf.predict(X).astype(float)

    best_idx = int(np.argmax(scores))
    best_name = names[best_idx]
    best_score = float(scores[best_idx])

    st.subheader("Results")
    cols = st.columns(min(4, len(images)))
    for i, (img, name, s) in enumerate(zip(images, names, scores)):
        with cols[i % len(cols)]:
            st.image(img, caption=f"{name}\nScore: {s:.3f}", use_container_width=True)
            if i == best_idx:
                st.success("Best", icon="🏆")

    st.markdown(f"**Selected best:** `{best_name}`  (score={best_score:.3f})")

    # JSON result download (optional)
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
    item_name = st.text_input(
        "Item / Batch name",
        value=default_item_name,
        help="Short label for this set of images (e.g., SKU, product code)"
    )

    if st.button("Save to database", type="primary"):
        if not item_name.strip():
            st.error("Please provide an Item / Batch name.")
        else:
            try:
                order = np.argsort(-scores)  # descending
                rank_of = {int(i): int(r) for r, i in enumerate(order)}

                with get_session() as s:
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

                    for i, (fname, sc) in enumerate(zip(names, scores)):
                        s.add(ImageRankItem(
                            batch_id=batch.id,
                            file_name=fname,
                            score=float(sc),
                            is_best=(i == best_idx),
                            rank=rank_of[i],
                        ))

                    bid = batch.id

                # Store and rerun to refresh "Saved Batches" section
                st.session_state["saved_bid"] = bid
                st.rerun()

            except Exception as e:
                st.error(f"DB save failed: {e}")


# Allow running this file directly for quick testing
if __name__ == "__main__":
    render_image_ranker_ui()