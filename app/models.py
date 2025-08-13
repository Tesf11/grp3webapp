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
    """
    One row per upload set (batch). Holds the best image summary and audit info.
    """
    __tablename__ = "image_rank_batches"

    id = Column(Integer, primary_key=True)

    # A human label for the upload (SKU, product name, etc.)
    item_name = Column(String(255), nullable=False)

    # Best selection (summary at batch level)
    best_index = Column(Integer, nullable=False)
    best_name  = Column(String(255), nullable=False)
    best_score = Column(Float, nullable=False)

    # Model info (audit)
    model_kind = Column(String(100))
    model_path = Column(String(500))

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Children
    items = relationship(
        "ImageRankItem",
        back_populates="batch",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ImageRankItem.rank.asc()",
    )

    def __repr__(self):
        return f"<ImageRankBatch id={self.id} item_name={self.item_name} best={self.best_name} score={self.best_score:.3f}>"


class ImageRankItem(Base):
    """
    One row per image within a batch. Stores filename, score, best flag, and rank (0 = best).
    """
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

    def __repr__(self):
        return f"<ImageRankItem batch_id={self.batch_id} file_name={self.file_name} score={self.score:.3f} best={self.is_best}>"