from services import auto_sectionize, build_faiss_index, build_prompt, clean_text, extract_regulations, extract_text_from_pdf, get_chunks, get_vectors, getPDFs, getInputPDFs, split_into_clauses, second_prompt
from models import GenerationResponse, PromptRequest, Files, AnalyzeRequest, AnalyzeResponse
from fastapi import APIRouter, UploadFile, File, Depends
from typing_extensions import Annotated
from typing import Optional
from sqlmodel import Session
from database import get_session
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import fitz
import faiss
import numpy as np
import ollama
import re
import boto3
import uuid
import requests

SessionDep = Annotated[Session, Depends(get_session)]
s3 = boto3.client('s3')
 
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

#take PDF ids as inputs here, and place them inside the parameters for getInpuPDFs
@router.post("/analyze")
async def analyze(data: AnalyzeRequest, session: Session = Depends(get_session)):

    return await run_in_threadpool(run_analysis, data, session)
    
def run_analysis(data, session):

    # get DB rows
    item = session.get(Files, data.item_id)
    item2 = session.get(Files, data.item_id2)

    # # error handling if not found
    # if not item or not item2:
    #     raise HTTPException(status_code=404, detail="File ID not found")

    # === Download PDFs ===
    presigned1 = s3.generate_presigned_url(
    "get_object",
    Params={"Bucket": "izzat-demo-s3-v1", "Key": item.s3key},
    ExpiresIn=3600
)

    response = requests.get(presigned1)
    with open("temp1.pdf", "wb") as f:
        f.write(response.content)

    presigned2 = s3.generate_presigned_url(
    "get_object",
    Params={"Bucket": "izzat-demo-s3-v1", "Key": item2.s3key},
    ExpiresIn=3600
)
    response = requests.get(presigned2)
    with open("temp2.pdf", "wb") as f:
        f.write(response.content)

    # === Extract text ===
    text1 = extract_text_from_pdf("temp1.pdf")
    text2 = extract_text_from_pdf("temp2.pdf")

    # === Clean text ===
    cleanedtext = clean_text(text1)
    cleanedtext_reg = clean_text(text2)
    reg_sections = extract_regulations(cleanedtext_reg)
    # sectionioned_policy = auto_sectionize(cleanedtext)
    # sectionized_regulation = auto_sectionize(cleanedtext_reg)
    # === Chunk + Embed + FAISS ===
    policy_chunks = get_chunks(cleanedtext)
    reg_clauses = split_into_clauses(cleanedtext_reg)
    chunks = get_chunks("\n\n".join(reg_clauses))
    # regulation_chunks = get_chunks(cleanedtext_reg)
    policy_vectors = get_vectors(policy_chunks)
    faiss.normalize_L2(policy_vectors)

    index = build_faiss_index(policy_vectors)
    
    report = []
    regulation_chunks = []
    for section in reg_sections:
        regulation_chunks.extend(get_chunks(section))
    # === Compare ===
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

@router.post("/responseanalyze")
async def second_analysis(data: AnalyzeResponse):
    """
    Takes a single non-compliant regulation + analysis from the frontend,
    sends it to an Ollama model (Llama 3), and returns a rewritten or
    improved compliance recommendation list.
    """

    # Build the prompt
    prompt = second_prompt(data.analysis, data.regulation)

    # Call the local Ollama model
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Strongly recommended: safe extraction
    result = response.get("message", {}).get("content", "")

    # Enforce consistent API shape for frontend
    return {
        "report": [
            {
                "regulation": data.regulation,
                "analysis": result  # enhanced compliance guidance
            }
        ]
    }
    

@router.post("/upload")
async def upload_file(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    session: Session = Depends(get_session)
):
    bucket_name = "izzat-demo-s3-v1"

    # --- Generate unique object keys ---
    key1 = f"{uuid.uuid4()}_{file1.filename}"
    key2 = f"{uuid.uuid4()}_{file2.filename}"

    # --- Upload to S3 without ACLs (bucket owner enforced) ---
    s3.upload_fileobj(
        file1.file,
        bucket_name,
        key1
        # No ACL parameter allowed in modern S3 buckets
    )

    s3.upload_fileobj(
        file2.file,
        bucket_name,
        key2
    )

    # --- Store ONLY the keys in the database ---
    db_file1 = Files(filename=file1.filename, s3key=key1)
    db_file2 = Files(filename=file2.filename, s3key=key2)

    session.add(db_file1)
    session.add(db_file2)
    session.commit()
    session.refresh(db_file1)
    session.refresh(db_file2)

    return {
        "file1_id": db_file1.id,
        "file2_id": db_file2.id,
        "message": "Files uploaded successfully"
    }