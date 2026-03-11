from sqlalchemy import Column, Integer, String, DateTime
from database import Base
import datetime
from pydantic import BaseModel, Field
from typing import Optional

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), nullable=False)
    email = Column(String(50), unique=True, nullable=False)
    password = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_length: Optional[int] = Field(50)

class GenerationResponse(BaseModel):
    generated_text: str