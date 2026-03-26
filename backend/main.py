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

# try:
#    connection = psycopg2.connect(
#        host="localhost",
#        database="postgres",
#        user="izzatkhadim",
#        port="5432" # Default port is 5432
#    )
#    print("Connected to the PostgreSQL server successfully!")


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
#git push origin master inside of my-ai-app directory to push frontend changes

#on mac to setup postgres or start:
#brew install postgresql
#initdb /usr/local/var/postgres for intel macs
#initdb /opt/homebrew/bin/brew for silicon m1 macs
#if previous initdb commands return an error try these lines:
#pg_ctl -D /usr/local/var/postgres start for intel macs
#pg_ctl -D /opt/homebrew/var/postgres start for silicon m1 macs
#psql postgres to test it is working
#create a user izzatkhadim and make him the admin of db named postgres

#on mac also install awscli using the .pkg installer 

#on mac in backend folder:
#izzatkhadim@izzats-Mac-mini backend % python3 -m venv venv
#izzatkhadim@izzats-Mac-mini backend % source venv/bin/activate
#in another terminal:
#✅ Next step — connect to it


#psql postgres

#You should see:

#postgres=#
# createdb postgres

#-- List databases
#\l

#-- Switch database
#\c postgres
#or postgres
#then create table/schema with this model
# CREATE TABLE Files (
#     id SERIAL PRIMARY KEY,
#     filename TEXT NOT NULL,
#     s3url TEXT NOT NULL,
#     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );
#run \dt to confirm Files table/schema was created

#if you get error 'role posrgres does not exist' then exec these commands
# Option 2 - Create a postgres role with password

# Find your macOS username:
# psql postgres
# CREATE ROLE postgres WITH LOGIN PASSWORD 'getlucky15' SUPERUSER CREATEDB;