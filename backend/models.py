from sqlalchemy import Column, Integer, String, DateTime
import datetime
from pydantic import BaseModel, Field
from typing import Optional
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select

#windows version 
# class Files(SQLModel, table=True):
#     id: int | None = Field(default=None, primary_key=True)
#     filename: str = Field(index=True)
#     s3url: str = Field(index=True)

class Files(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    s3key: str

# Code below omitted 👇


class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_length: Optional[int] = Field(50)

class GenerationResponse(BaseModel):
    generated_text: str

class AnalyzeRequest(BaseModel):
    item_id: int
    item_id2: int