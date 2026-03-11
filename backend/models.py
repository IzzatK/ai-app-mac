from sqlalchemy import Column, Integer, String, DateTime, Boolean
from database import Base
import datetime

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), nullable=False)
    email = Column(String(50), unique=True, nullable=False)
    password = Column(String(50), nullable=False)

class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The text prompt to send to the LLM")
    max_length: Optional[int] = Field(50, description="The maximum length of the generated text")

class GenerationResponse(BaseModel):
    generated_text: str

