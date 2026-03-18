from services import build_faiss_index, build_prompt, clean_text, extract_text_from_pdf, get_chunks, get_vectors, getPDFs
from models import GenerationResponse, PromptRequest, Files
from fastapi import APIRouter, UploadFile, File, Depends
# from typing import Annotated
from typing_extensions import Annotated
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
def analyze():
    # pdf_url1 = db_record.s3_url

    # response = requests.get(pdf_url1)

    # with open("temp1.pdf", "wb") as f:
    #     f.write(response.content)

    # text = extract_text_from_pdf("temp1.pdf")
    # # pdf_url2 = db_record.s3_url

    # response = requests.get(pdf_url2)

    # with open("temp2.pdf", "wb") as f:
    #     f.write(response.content)

    # text = extract_text_from_pdf("temp2.pdf")
    #pdf_1, pdf_2 = getInputPDFs("temp1.pdf", "temp2.pdf")
    pdf_content, pdf_content2 = getPDFs()
    cleanedtext = clean_text(pdf_content)
    cleanedtext_reg = clean_text(pdf_content2)
    policy_chunks = get_chunks(cleanedtext)
    regulation_chunks = get_chunks(cleanedtext_reg)
    policy_vectors = get_vectors(policy_chunks)
    faiss.normalize_L2(policy_vectors)
    # faiss.normalize_L2(regulation_vectors)    
    index = build_faiss_index(policy_vectors)
    
    report = []
    #maybe make report into an object
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

@router.post("/upload")
async def upload_file(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    session: Session = Depends(get_session)
):
    bucket_name = "izzat-demo-s3-v1"
    unique_name1 = f"{uuid.uuid4()}_{file1.filename}"
    unique_name2 = f"{uuid.uuid4()}_{file2.filename}"

    s3.upload_fileobj(file1.file, bucket_name, unique_name1)
    s3.upload_fileobj(file2.file, bucket_name, unique_name2)

    url1 = f"https://{bucket_name}.s3.amazonaws.com/{unique_name1}"
    url2 = f"https://{bucket_name}.s3.amazonaws.com/{unique_name2}"

    # Create SQLModel objects
    db_file1 = Files(filename=file1.filename, s3url=url1)
    db_file2 = Files(filename=file2.filename, s3url=url2)

    # Add to DB
    session.add(db_file1)
    session.add(db_file2)
    session.commit()  # don't forget commit to actually insert
    session.refresh(db_file1)
    session.refresh(db_file2)

    return {
        "policy_url": url1,
        "regulation_url": url2,
        "file1_id": db_file1.id,
        "file2_id": db_file2.id
    }