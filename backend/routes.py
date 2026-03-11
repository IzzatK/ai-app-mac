from services import build_faiss_index, build_prompt, clean_text, extract_text_from_pdf, get_chunks, get_vectors, getPDFs
from models import GenerationResponse, PromptRequest
from fastapi import APIRouter, UploadFile, File
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import fitz
import faiss
import numpy as np
import ollama
import re


router = APIRouter()
try:
    nlp = pipeline("text-generation", model="gpt2")
except Exception as e:
    print(f"Error loading model: {e}")
    # You might want to handle this more gracefully in production
model = SentenceTransformer("all-mpnet-base-v2")
@router.post("/generate", response_model=GenerationResponse)
async def generate_text(request: PromptRequest):
    # The pipeline call itself is synchronous, but the endpoint is async
    # In a real-world scenario with large models, you might offload this
    generated_text_list = nlp(
        request.prompt,
        max_length=request.max_length,
        num_return_sequences=1
    )
    generated_text = generated_text_list[0]["generated_text"]
    return GenerationResponse(generated_text=generated_text)

@router.post("/analyze")
def analyze():
    pdf_content, pdf_content2 = getPDFs()
    cleanedtext = clean_text(pdf_content)
    cleanedtext_reg = clean_text(pdf_content2)
    policy_chunks = get_chunks(cleanedtext)
    regulation_chunks = get_chunks(cleanedtext_reg)
    policy_vectors = get_vectors(policy_chunks)
    regulation_vectors = get_vectors(regulation_chunks)
    index = build_faiss_index(policy_vectors)
    
    report = []
    for reg_chunk in regulation_chunks:
        reg_vec = model.encode([reg_chunk])
        D, I = index.search(np.array(reg_vec), k=3)
        top_policies = [policy_chunks[i] for i in I[0]]
        prompt = build_prompt(reg_chunk, top_policies)
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        result = response["message"]["content"]
        report.append({"regulation": reg_chunk, "analysis": result})
    
    return {"report": report}

    