# Streamlit UI: Image Ranker
from __future__ import annotations
from pathlib import Path
import io, json
import numpy as np
from PIL import Image
import streamlit as st
import joblib
import torch

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

    # Try original CLIP first
    try:
        import clip  # pip install git+https://github.com/openai/CLIP.git  (optional)
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
        # Fallback to open_clip (pip install open_clip_torch)
        import open_clip
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

# ---------- UI ----------
def render_image_ranker_ui():
    st.title("🖼️ Image Ranker (Best Image Picker)")
    st.caption(f"Model: {MODEL_PATH.relative_to(APP_DIR.parent)}")

    clf, feat_cols, meta = load_model_and_meta()
    featurize = load_clip_featurizer()

    uploaded = st.file_uploader(
        "Upload images of the **same item** (2+ images)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
    )

    if not uploaded:
        st.info("Select multiple images to rank.")
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

    # Score with your classifier
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X)[:, 1]
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
        # normalize to 0..1 for display
        smin, smax = float(scores.min()), float(scores.max())
        scores = (scores - smin) / (smax - smin + 1e-9)
    else:
        # Fallback: use predictions as 0/1
        scores = clf.predict(X).astype(float)

    best_idx = int(np.argmax(scores))
    best_name = names[best_idx]

    st.subheader("Results")
    cols = st.columns(min(4, len(images)))
    for i, (img, name, s) in enumerate(zip(images, names, scores)):
        with cols[i % len(cols)]:
            st.image(img, caption=f"{name}\nScore: {s:.3f}", use_container_width=True)
            if i == best_idx:
                st.success("Best", icon="🏆")

    st.markdown(f"**Selected best:** `{best_name}`  (score={scores[best_idx]:.3f})")

    # Optional: JSON result for your backend
    result = {
        "best_index": best_idx,
        "best_name": best_name,
        "scores": {n: float(s) for n, s in zip(names, scores)},
        "model_kind": meta.get("model_kind"),
    }
    st.download_button("Download result JSON", data=json.dumps(result, indent=2), file_name="image_ranker_result.json")

# If you want to run this file standalone for quick testing:
if __name__ == "__main__":
    import streamlit as st  # noqa
    render_image_ranker_ui()