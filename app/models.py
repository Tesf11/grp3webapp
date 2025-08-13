# app/models.py
from sqlalchemy import Column, Integer, String, Float, Text
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
