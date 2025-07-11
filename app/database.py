
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DATABASE_URL = "sqlite:///./documents.db"

# Drop existing database to recreate with new schema
if os.path.exists("documents.db"):
    os.remove("documents.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    department = Column(String)
    department_id = Column(Integer)
    extracted_date = Column(Date)
    priority = Column(String)
    upload_time = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)
