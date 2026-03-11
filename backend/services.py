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

pdf_content, pdf_content2 = getPDFs()
# word_count = len(pdf_content.split())

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
cleanedtext = clean_text(pdf_content)
cleanedtext_reg = clean_text(pdf_content2)

policy_chunks = get_chunks(cleanedtext)
regulation_chunks = get_chunks(cleanedtext_reg)

def get_vectors(chunks):
    model = SentenceTransformer("all-mpnet-base-v2")
    vectors = model.encode(chunks)
    return vectors
model = SentenceTransformer("all-mpnet-base-v2")
policy_vectors = get_vectors(policy_chunks)
regulation_vectors = get_vectors(regulation_chunks)

def build_faiss_index(vectors):
    dimension = vectors[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors))
    return index
faiss.normalize_L2(policy_vectors)
faiss.normalize_L2(regulation_vectors)
# dimension = policy_vectors[0].shape[0]
# index = faiss.IndexFlatL2(dimension)
index = build_faiss_index(policy_vectors)




#retrive top policy chunks, send them to an LLM to compare with regulation chunks
def get_reg_vec(chunks):
    reg_vec = model.encode(chunks)
    return reg_vec

reg_vec = get_reg_vec(regulation_chunks)

def get_D_I(reg_vec):
    D, I = index.search(np.array(reg_vec), k=3)
    return D, I

D, I = get_D_I(reg_vec)

def get_top_policy_chunks(I):
    top_policy_chunks = [policy_chunks[i] for i in I[0]]
    return top_policy_chunks

top_policy_chunks = get_top_policy_chunks(I)

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
prompt = build_prompt(regulation_chunks, top_policy_chunks)

def get_analysis(prompt):
    response = ollama.chat(
    model="llama3",
    messages=[{"role": "user", "content": prompt}]
)

    analysis = response["message"]["content"]

    return analysis