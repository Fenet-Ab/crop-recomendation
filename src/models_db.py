# src/models_db.py
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship # type: ignore
import datetime
from src.database import Base

class Farmer(Base):
    __tablename__ = "farmers"
    
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    phone_number = Column(String, unique=True, index=True, nullable=False)  # e.g., 251911223344
    region = Column(String, nullable=False)
    language_pref = Column(String, default="am")  # "am" for Amharic, "om" for Afan Oromo
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    sms_logs = relationship("SmsLog", back_populates="farmer")

class SmsLog(Base):
    __tablename__ = "sms_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    farmer_id = Column(Integer, ForeignKey("farmers.id"))
    crop_name = Column(String)
    message_body = Column(Text)
    gateway_status = Column(String)
    sent_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    farmer = relationship("Farmer", back_populates="sms_logs")