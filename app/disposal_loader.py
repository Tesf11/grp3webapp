# app/disposal_loader.py
import os, json, numpy as np, tensorflow as tf
from tensorflow.keras.layers import TFSMLayer

ROOT = "app/models/disposal_predictor"

with open(os.path.join(ROOT, "mapping_data.json"), "r") as f:
    MAP = json.load(f)

IMG_SIZE = tuple(MAP["img_size"])          # e.g. [320, 320]
THR      = float(MAP.get("threshold", 0.55))
CLASSES  = MAP["class_names"]

# Load SavedModel using Keras 3 TFSMLayer
INFER = TFSMLayer(os.path.join(ROOT, "tf_savedmodel"), call_endpoint="serving_default")

# Warm up
_ = INFER(tf.zeros([1, IMG_SIZE[0], IMG_SIZE[1], 3], tf.float32))

def predict(rgb_uint8: np.ndarray):
    """rgb_uint8: HxWx3, dtype=uint8. Returns dict with label, prob, low_conf, candidates."""
    x = tf.convert_to_tensor(rgb_uint8, dtype=tf.float32)
    x = tf.image.resize(x, IMG_SIZE)
    x = tf.expand_dims(x, 0)  # (1,H,W,3)
    y = INFER(x)
    if isinstance(y, dict):
        y = next(iter(y.values()))
    probs = tf.convert_to_tensor(y, tf.float32).numpy()[0]  # (C,)
    top2 = probs.argsort()[-2:][::-1]
    label, p = CLASSES[top2[0]], float(probs[top2[0]])
    return {
        "label": label,
        "prob": p,
        "low_conf": p < THR,
        "candidates": [(CLASSES[i], float(probs[i])) for i in top2]
    }
