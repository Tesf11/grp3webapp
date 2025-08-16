# app/models.py
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db import Base  # this should be your shared SQLAlchemy Base

class Entry(Base):
    __tablename__ = "entries"
    id = Column(Integer, primary_key=True)
    sample_type = Column(String(50), nullable=False)
    company = Column(String(200), nullable=False)
    contact_person = Column(String(200))
    product_title = Column(Text, nullable=False)
    category = Column(String(200), nullable=False)
    tracking_number = Column(String(200))
    status = Column(String(50), nullable=False)
    courier_cost = Column(Float, default=0.0)
    eta = Column(String(20))  # ISO date string

class ImageRankBatch(Base):
    __tablename__ = "image_rank_batches"

    id = Column(Integer, primary_key=True)
    item_name = Column(String(255), nullable=False)

    best_index = Column(Integer, nullable=False)
    best_name  = Column(String(255), nullable=False)
    best_score = Column(Float, nullable=False)

    model_kind = Column(String(100))
    model_path = Column(String(500))

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    items = relationship(
        "ImageRankItem",
        back_populates="batch",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ImageRankItem.rank.asc()",
    )

# ---- Image Ranker: items in a batch ----
class ImageRankItem(Base):
    __tablename__ = "image_rank_items"

    id = Column(Integer, primary_key=True)
    batch_id = Column(
        Integer,
        ForeignKey("image_rank_batches.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    file_name = Column(String(255), nullable=False)
    score     = Column(Float, nullable=False)
    is_best   = Column(Boolean, nullable=False, default=False)
    rank      = Column(Integer)  # 0 = best, 1 = next, ...

    batch = relationship("ImageRankBatch", back_populates="items")

# ---- NEW: Alt text saved for a batch’s best image ----
class ImageAltText(Base):
    __tablename__ = "image_alt_texts"

    id = Column(Integer, primary_key=True)
    batch_id = Column(Integer, ForeignKey("image_rank_batches.id", ondelete="CASCADE"),
                      index=True, nullable=False)
    # optional link to the exact best item row
    item_id  = Column(Integer, ForeignKey("image_rank_items.id", ondelete="CASCADE"),
                      index=True, nullable=True)

    alt_text = Column(Text, nullable=False)
    provider = Column(String(50), default="gemini")
    model    = Column(String(100), default="gemini-1.5-flash")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)

    # Inputs
    description = Column(Text, nullable=False)
    weight_g = Column(Float, nullable=False)

    # Model outputs
    predicted_product_type = Column(String(64), nullable=False)
    storage_bin = Column(String(64), nullable=False)

    # Optional metadata
    model_name = Column(String(64))
    model_version = Column(String(64))

    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer, nullable=False)
    correct_product_type = Column(String(64))     # optional corrected label
    correct_storage_bin = Column(String(64))      # optional corrected bin
    notes = Column(Text)                          # optional free text
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# ---- Disposal Predictor: saved predictions ----
class DisposalPrediction(Base):
    __tablename__ = "disposal_predictions"

    id        = Column(Integer, primary_key=True)
    file_name = Column(String(255))
    label     = Column(String(128), nullable=False)
    prob      = Column(Float,      nullable=False)
    low_conf  = Column(Boolean,    nullable=False)
    # store top-k as JSON string to avoid dialect issues
    candidates_json = Column(Text)           # e.g. '[["aerosol_cans",0.91],["... ",0.07]]'
    guidance  = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class GenAILog(Base):
    __tablename__ = "genai_logs"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime(timezone=True), server_default=func.now())

    model = Column(String(64), nullable=False)
    prompt_type = Column(String(64))            # e.g., "ideas", "tagging", "summary"
    prompt_chars = Column(Integer)              # len(prompt)
    response_chars = Column(Integer)

    input_tokens = Column(Integer)              # from response.usage_metadata
    output_tokens = Column(Integer)
    total_tokens = Column(Integer)

    latency_ms = Column(Float)
    status = Column(String(32), default="ok")   # ok | blocked | error
    error_code = Column(String(64))
    safety_blocked = Column(Boolean, default=False)

    # optional: short samples for QA (do NOT store full PII)
    prompt_sample = Column(Text)
    response_sample = Column(Text)
