"""
CV Chatbot with RAG (Retrieval-Augmented Generation)
FastAPI backend that uses semantic search to answer questions about your CV
"""

import json
import os
from typing import List, Dict, Optional
import numpy as np
import torch
import httpx
import inspect
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import huggingface_hub

# Provide backward-compatible alias for deprecated cached_download expected by older sentence-transformers.
if not hasattr(huggingface_hub, "cached_download"):
    from pathlib import Path
    from urllib.parse import urlparse

    import requests
    from huggingface_hub.utils import build_hf_headers

    def cached_download(  # type: ignore[override]
        url: str,
        *,
        cache_dir: str | None = None,
        force_filename: str | None = None,
        library_name: str | None = None,
        library_version: str | None = None,
        user_agent: str | None = None,
        use_auth_token: str | None = None,
        **_: dict
    ) -> str:
        """
        Minimal shim replicating the deprecated huggingface_hub.cached_download API.
        Downloads the file to the requested cache directory while supporting
        the keyword arguments used by sentence-transformers==2.2.2.
        """
        cache_root = Path(cache_dir or huggingface_hub.constants.HUGGINGFACE_HUB_CACHE)
        filename = force_filename or Path(urlparse(url).path).name
        target_path = cache_root / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists():
            return str(target_path)

        headers = build_hf_headers(
            library_name=library_name,
            library_version=library_version,
            user_agent=user_agent,
            token=use_auth_token,
        )

        with requests.get(url, stream=True, headers=headers) as response:
            response.raise_for_status()
            with open(target_path, "wb") as file_out:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file_out.write(chunk)

        return str(target_path)

    huggingface_hub.cached_download = cached_download  # type: ignore[attr-defined]

from sentence_transformers import SentenceTransformer
import faiss

# Patch httpx to gracefully ignore deprecated `proxies` argument used by groq client when running with httpx>=0.28.
if "proxies" not in inspect.signature(httpx.Client.__init__).parameters:
    _original_httpx_client_init = httpx.Client.__init__

    def _httpx_client_init_with_proxies(self, *args, proxies=None, **kwargs):
        return _original_httpx_client_init(self, *args, **kwargs)

    httpx.Client.__init__ = _httpx_client_init_with_proxies  # type: ignore[assignment]

if "proxies" not in inspect.signature(httpx.AsyncClient.__init__).parameters:
    _original_httpx_async_client_init = httpx.AsyncClient.__init__

    def _httpx_async_client_init_with_proxies(self, *args, proxies=None, **kwargs):
        if proxies is not None and "proxy" not in kwargs:
            kwargs["proxy"] = proxies
        return _original_httpx_async_client_init(self, *args, **kwargs)

    httpx.AsyncClient.__init__ = _httpx_async_client_init_with_proxies  # type: ignore[assignment]

from groq import Groq

# Import configuration
from config import (
    LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_MODEL,
    HUGGINGFACE_API_KEY,
    HUGGINGFACE_MODEL,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    SYSTEM_PROMPT
)

# Initialize FastAPI
app = FastAPI(title="CV Chatbot RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    context_used: List[str]

# Global variables for RAG components
embedding_model = None
model_device = "cpu"
cv_chunks = []
cv_embeddings = None
faiss_index = None
llm_client = None


def load_cv_data(file_path: str = "cv_data.json") -> str:
    """Load and flatten CV data from JSON into a single text"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Convert structured JSON to readable text
    text_parts = []

    # Personal info
    if "personal_info" in data:
        info = data["personal_info"]
        text_parts.append(f"Name: {info.get('name', '')}")
        text_parts.append(f"Title: {info.get('title', '')}")
        text_parts.append(f"Bio: {info.get('bio', '')}")
        text_parts.append(f"Contact: {info.get('email', '')}, {info.get('location', '')}")

    # Summary
    if "summary" in data:
        text_parts.append(f"Professional Summary: {data['summary']}")

    # Skills
    if "skills" in data:
        for category, items in data["skills"].items():
            text_parts.append(f"{category.replace('_', ' ').title()}: {', '.join(items)}")

    # Experience
    if "experience" in data:
        for exp in data["experience"]:
            text_parts.append(
                f"Experience: {exp['title']} at {exp['company']} ({exp['duration']}). "
                f"{exp['description']} Achievements: {' '.join(exp.get('achievements', []))}"
            )

    # Education
    if "education" in data:
        for edu in data["education"]:
            text_parts.append(
                f"Education: {edu['degree']} from {edu['institution']} ({edu.get('graduation', '')})"
            )

    # Projects
    if "projects" in data:
        for proj in data["projects"]:
            text_parts.append(
                f"Project: {proj['name']}. {proj['description']} "
                f"Technologies: {', '.join(proj.get('technologies', []))}. "
                f"{' '.join(proj.get('highlights', []))}"
            )

    # Certifications
    if "certifications" in data:
        for cert in data["certifications"]:
            text_parts.append(f"Certification: {cert['name']} from {cert['issuer']}")

    # Interests
    if "interests" in data:
        text_parts.append(f"Interests: {', '.join(data['interests'])}")

    return "\n\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundary
        if end < text_length:
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)

            if break_point > chunk_size * 0.5:  # Only break if we're past halfway
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks


def initialize_rag():
    """Initialize RAG components: embeddings, vector store"""
    global embedding_model, cv_chunks, cv_embeddings, faiss_index, model_device

    print("Loading embedding model...")
    model_device = "cpu"
    if torch.cuda.is_available():
        try:
            embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
            model_device = "cuda"
            print("Embedding model loaded on CUDA")
        except Exception as cuda_err:
            print(f"CUDA initialization failed ({cuda_err}); falling back to CPU.")
            embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    else:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    print(f"Embedding model using device: {model_device}")

    print("Loading CV data...")
    cv_text = load_cv_data()

    print("Chunking CV text...")
    cv_chunks = chunk_text(cv_text)
    print(f"Created {len(cv_chunks)} chunks")

    print("Generating embeddings...")
    try:
        cv_embeddings = embedding_model.encode(cv_chunks, convert_to_numpy=True)
    except RuntimeError as err:
        if "cuda" in str(err).lower():
            print(f"CUDA error during embedding generation ({err}); retrying on CPU.")
            embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
            model_device = "cpu"
            cv_embeddings = embedding_model.encode(cv_chunks, convert_to_numpy=True)
        else:
            raise

    print("Building FAISS index...")
    dimension = cv_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(cv_embeddings)

    print("RAG initialization complete!")


def initialize_llm():
    """Initialize LLM client based on provider"""
    global llm_client

    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in environment variables")
        llm_client = Groq(api_key=GROQ_API_KEY)
        print(f"Initialized Groq client with model: {GROQ_MODEL}")
    elif LLM_PROVIDER == "huggingface":
        # Will use requests for HF Inference API
        if not HUGGINGFACE_API_KEY:
            raise ValueError("HUGGINGFACE_API_KEY not set in environment variables")
        print(f"Initialized HuggingFace Inference API with model: {HUGGINGFACE_MODEL}")
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")


def retrieve_relevant_chunks(query: str, top_k: int = TOP_K_RESULTS) -> List[str]:
    """Retrieve most relevant CV chunks for the query"""
    # Embed the query
    try:
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    except RuntimeError as err:
        if "cuda" in str(err).lower():
            print(f"CUDA error during query embedding ({err}); moving model to CPU.")
            embedding_model.to("cpu")
            query_embedding = embedding_model.encode([query], convert_to_numpy=True)
        else:
            raise

    # Search in FAISS index
    distances, indices = faiss_index.search(query_embedding, top_k)

    # Get the relevant chunks
    relevant_chunks = [cv_chunks[idx] for idx in indices[0]]

    return relevant_chunks


def generate_response_groq(prompt: str) -> str:
    """Generate response using Groq API"""
    try:
        chat_completion = llm_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            model=GROQ_MODEL,
            temperature=0.7,
            max_tokens=500,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")


def generate_response_huggingface(prompt: str) -> str:
    """Generate response using HuggingFace Inference API (OpenAI-compatible endpoint)."""
    import requests

    if not HUGGINGFACE_API_KEY:
        raise HTTPException(status_code=500, detail="HUGGINGFACE_API_KEY is not set")

    api_url = "https://router.huggingface.co/v1/chat/completions"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {
        "model": HUGGINGFACE_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 500,
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        print("HuggingFace status:", response.status_code)
        print("HuggingFace response text:", response.text[:500])
        response.raise_for_status()

        result = response.json()
        if isinstance(result, dict):
            choices = result.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message", {})
                content = message.get("content")
                if content:
                    return content.strip()
        return str(result)
    except Exception as e:
        print("HuggingFace API error occurred:", repr(e))
        raise HTTPException(status_code=500, detail=f"HuggingFace API error: {str(e)}")

def generate_response(context: str, question: str) -> str:
    """Generate response using configured LLM provider"""
    prompt = f"""Context from CV:
{context}

Question: {question}

Answer based on the context above:"""

    if LLM_PROVIDER == "groq":
        return generate_response_groq(prompt)
    elif LLM_PROVIDER == "huggingface":
        return generate_response_huggingface(prompt)
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")


@app.on_event("startup")
async def startup_event():
    """Initialize RAG and LLM on startup"""
    print("Starting up...")
    initialize_rag()
    initialize_llm()
    print("Ready to serve requests!")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "CV Chatbot RAG API is running",
        "llm_provider": LLM_PROVIDER,
        "chunks_loaded": len(cv_chunks)
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with RAG"""
    try:
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(request.message)

        # Build context from chunks
        context = "\n\n".join(relevant_chunks)

        # Generate response
        response = generate_response(context, request.message)

        return ChatResponse(
            response=response,
            context_used=relevant_chunks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "rag_initialized": embedding_model is not None,
        "llm_initialized": llm_client is not None or LLM_PROVIDER == "huggingface",
        "chunks_count": len(cv_chunks),
        "llm_provider": LLM_PROVIDER
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
