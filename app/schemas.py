from sqlalchemy import Column, Integer, Float, Boolean
from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    probability = Column(Float)
    prediction = Column(Integer)
    anomaly = Column(Boolean)