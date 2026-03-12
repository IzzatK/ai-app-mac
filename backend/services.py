from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
from transformers import pipeline
import fitz 
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
import psycopg2

model = SentenceTransformer("all-mpnet-base-v2")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def getPDFs():
    pdf_content = extract_text_from_pdf("Example-PDF1.pdf")
    pdf_content2 = extract_text_from_pdf("example-2.pdf")
    return pdf_content, pdf_content2

def getInputPDFs(pdf1, pdf2):
    pdf_content_1 = extract_text_from_pdf(pdf1)
    pdf_content_2 = extract_text_from_pdf(pdf2)
    return pdf_content_1, pdf_content_2
#grab the PDFs from BackEnd folder using its postgresSQL primary key ID sent in to this above function from Redux, extract text, clean it, chunk it, vectorize it, build a faiss index, then retrieve top policy chunks for each regulation chunk and send them to an LLM to compare and analyze
#When the user saves a PDF into postgresSQL, we don't exec getInputPDFs. We exec it only when the user execs the POST analyze controller endpoint
pdf_content, pdf_content2 = getPDFs()
# change it to getInputPDFs(pdf1, pdf2) when you want to use the function that takes in pdfs as input

# is_over_500 = word_count > 500
# print(is_over_500)

def clean_text(text):
    text = text.replace('\n', ' ').replace('\r', ' ') # Remove newlines
    text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with one
    text = text.strip()
    return text

def get_chunks(text, chunk_size=400, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectors(chunks):
    model = SentenceTransformer("all-mpnet-base-v2")
    vectors = model.encode(chunks)
    return vectors

def build_faiss_index(vectors):
    dimension = vectors[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors))
    return index


def build_prompt(regulation_ch, policy_ch):

    policy_text = "\n\n".join(
        [f"Policy {i+1}: {chunk}" for i, chunk in enumerate(policy_ch)]
    )

    prompt = f"""
You are a regulatory compliance expert.

Regulation:
{regulation_ch}

Company Policies:
{policy_text}

Determine whether the company policies satisfy the regulation.

Respond with:

Compliance Status: COMPLIANT or NON-COMPLIANT

Explanation: Explain why.

If NON-COMPLIANT, suggest what policy change is required.
"""

    return prompt


# client = OpenAI(api_key="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")

def get_prompt(regulation_chunks, top_policy_chunks):
     prompt = build_prompt(regulation_chunks[0], top_policy_chunks)
     return prompt
# prompt = build_prompt(regulation_chunks, top_policy_chunks)

def get_analysis(prompt):
    response = ollama.chat(
    model="llama3",
    messages=[{"role": "user", "content": prompt}]
)

    analysis = response["message"]["content"]

    return analysis