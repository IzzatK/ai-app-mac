from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlmodel import SQLModel
from typing import Optional
from transformers import pipeline
from routes import router
import fitz 
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
import psycopg2
from database import engine
from models import Files







app = FastAPI(
    title="LLM FastAPI Demo",
    description="A simple API for text generation using a local LLM"
)

app.include_router(router)
#    allow_origins=["http://localhost:3000"],

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
cursor = None


SQLModel.metadata.create_all(engine)

try:
    connection = psycopg2.connect(
        host="localhost",
        database="postgres",
        user="postgres",
        password="getlucky15",
        port="5432" # Default port is 5432
    )
    print("Connected to the PostgreSQL server successfully!")
    cursor = conn.cursor()

except (psycopg2.DatabaseError, Exception) as error:
    print(f"Error connecting to the database: {error}")

finally:
    # Optional: ensure connection is closed, though using a 'with' statement is better practice
    if 'connection' in locals() and connection is not None:
        # connection.close()
        pass
#localhost 8000

#instructions:
#open venv python virtual terminal and then type uvicorn main:app --reload to run main.python
#ollama run llama3 in a terminal
#to push changes run git push -u origin current
