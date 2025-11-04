"""
CV Chatbot with RAG (Retrieval-Augmented Generation)
FastAPI backend that uses semantic search to answer questions about your CV
"""

import json
import os
import re
import threading
import time
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from fastapi import Depends, FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import huggingface_hub
from huggingface_hub import hf_hub_download
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

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

# Import configuration
from config import (
    LLM_PROVIDER,
    HUGGINGFACE_API_KEY,
    HUGGINGFACE_MODEL,
    LOCAL_MODEL_REPO,
    LOCAL_MODEL_FILENAME,
    LOCAL_MODEL_CONTEXT_LENGTH,
    LOCAL_MODEL_THREADS,
    LOCAL_MODEL_BATCH_SIZE,
    LOCAL_MODEL_MAX_OUTPUT_TOKENS,
    LOCAL_MODEL_HF_TOKEN,
    CLIENT_APP_ORIGINS,
    API_ACCESS_TOKEN,
    SESSION_TOKEN_SECRET,
    SESSION_TOKEN_TTL_SECONDS,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    SYSTEM_PROMPT
)

# Initialize FastAPI
app = FastAPI(title="CV Chatbot RAG API")

# Add CORS middleware
allowed_origins = CLIENT_APP_ORIGINS or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
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
local_model_path: str | None = None
local_model_lock = threading.Lock()
_session_serializer: Optional[URLSafeTimedSerializer] = None


def get_session_serializer() -> URLSafeTimedSerializer:
    """Lazily initialize the session token serializer."""
    global _session_serializer
    if not SESSION_TOKEN_SECRET:
        raise HTTPException(
            status_code=500,
            detail="SESSION_TOKEN_SECRET is not configured on the server.",
        )
    if _session_serializer is None:
        _session_serializer = URLSafeTimedSerializer(SESSION_TOKEN_SECRET)
    return _session_serializer


def create_session_token() -> str:
    """Create a signed, timestamped session token."""
    serializer = get_session_serializer()
    return serializer.dumps({"issued_at": int(time.time())})


def validate_session_token(token: str) -> None:
    """Validate an incoming session token and enforce expiration."""
    serializer = get_session_serializer()
    try:
        serializer.loads(token, max_age=SESSION_TOKEN_TTL_SECONDS)
    except SignatureExpired as err:
        raise HTTPException(status_code=401, detail="Session token expired") from err
    except BadSignature as err:
        raise HTTPException(status_code=401, detail="Invalid session token") from err


def personalize_question(text: str) -> Tuple[str, bool]:
    """Normalize questions and detect whether the user is addressing the assistant."""

    assistant_patterns = [
        r"\bwho\s+are\s+you\b",
        r"\bwhat\s+are\s+you\b",
        r"\bwho\s+is\s+this\b",
        r"\bare\s+you\s+(real|human)\b",
    ]
    normalized_lower = text.lower()
    if any(re.search(pattern, normalized_lower) for pattern in assistant_patterns):
        return text, True

    def match_case(token: str, replacement: str) -> str:
        if token.isupper():
            return replacement.upper()
        if token[0].isupper():
            return replacement.capitalize()
        return replacement

    def replace_third_person(match: re.Match[str]) -> str:
        token = match.group(0)
        return match_case(token, "Bi")

    def replace_possessive(match: re.Match[str]) -> str:
        token = match.group(0)
        return match_case(token, "Bi's")

    updated = re.sub(r"\bhis\b", replace_possessive, text, flags=re.IGNORECASE)
    updated = re.sub(r"\bhe\b", replace_third_person, updated, flags=re.IGNORECASE)
    updated = re.sub(r"\bhim\b", replace_third_person, updated, flags=re.IGNORECASE)
    return updated, False


def verify_client_access(
    x_api_token: str = Header(default=""),
    x_session_token: str = Header(default=""),
) -> None:
    """Ensure only approved clients can call protected endpoints."""
    if API_ACCESS_TOKEN:
        if not x_api_token:
            raise HTTPException(status_code=401, detail="Missing client token")
        if x_api_token != API_ACCESS_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid client token")
        return

    if SESSION_TOKEN_SECRET:
        if not x_session_token:
            raise HTTPException(status_code=401, detail="Missing session token")
        validate_session_token(x_session_token)
        return

    # If no secrets configured, allow access (useful for local development)
    return


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
    global llm_client, local_model_path

    if LLM_PROVIDER == "huggingface":
        # Will use requests for HF Inference API
        if not HUGGINGFACE_API_KEY:
            raise ValueError("HUGGINGFACE_API_KEY not set in environment variables")
        print(f"Initialized HuggingFace Inference API with model: {HUGGINGFACE_MODEL}")
    elif LLM_PROVIDER == "local":
        try:
            from llama_cpp import Llama  # type: ignore[import]
        except ImportError as import_err:
            raise ValueError(
                "llama-cpp-python is not installed. Ensure requirements are up to date."
            ) from import_err

        auth_token = LOCAL_MODEL_HF_TOKEN or None
        print(
            f"Downloading quantized model {LOCAL_MODEL_REPO}/{LOCAL_MODEL_FILENAME} "
            "via Hugging Face Hub..."
        )
        try:
            local_model_path = hf_hub_download(
                repo_id=LOCAL_MODEL_REPO,
                filename=LOCAL_MODEL_FILENAME,
                token=auth_token,
            )
        except Exception as download_err:
            raise ValueError(
                f"Failed to download local model file: {download_err}"
            ) from download_err

        print(
            "Loading local quantized model with llama.cpp "
            f"(context={LOCAL_MODEL_CONTEXT_LENGTH}, threads={LOCAL_MODEL_THREADS}, "
            f"batch={LOCAL_MODEL_BATCH_SIZE})..."
        )
        try:
            llm_client = Llama(
                model_path=local_model_path,
                n_ctx=LOCAL_MODEL_CONTEXT_LENGTH,
                n_threads=LOCAL_MODEL_THREADS,
                n_batch=LOCAL_MODEL_BATCH_SIZE,
                n_gpu_layers=0,
                chat_format="gemma",  # Works for both Gemma 1 and Gemma 2
                verbose=True,  # Enable to see prompt formatting
            )
        except Exception as load_err:
            raise ValueError(f"Failed to load local model: {load_err}") from load_err
        print("Local quantized model ready for inference.")
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


def generate_response_local(system_prompt: str, user_prompt: str) -> str:
    """Generate response using a locally hosted quantized model."""
    global llm_client

    if llm_client is None:
        raise HTTPException(status_code=500, detail="Local model is not initialized")

    try:
        with local_model_lock:
            if os.getenv("DEBUG_LOCAL_PROMPT", "0") == "1":
                preview = user_prompt if len(user_prompt) < 400 else user_prompt[:400] + "..."
                print("Local prompt =>", preview)
            completion = llm_client.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=LOCAL_MODEL_MAX_OUTPUT_TOKENS,
                temperature=0.5,
                top_p=0.9,
                repeat_penalty=1.2,
                stop=["<end_of_turn>", "</s>"],
            )
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Local model error: {err}") from err

    try:
        choices = completion.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if content:
                return content.strip()
        return str(completion)
    except Exception as parse_err:
        raise HTTPException(
            status_code=500, detail=f"Unexpected local model response format: {parse_err}"
        ) from parse_err

def generate_response(
    context: str,
    question: str,
    original_question: str | None = None,
    assistant_query: bool = False,
) -> str:
    """Generate response using configured LLM provider"""
    if assistant_query:
        persona_instruction = (
            "Respond in first person as Bi's AI assistant. Mention you run locally on a "
            "quantized Google Gemma 2B IT model (Q4_K_M via llama.cpp with MiniLM embeddings and FAISS)."
        )
    else:
        persona_instruction = (
            "Speak directly about Bi by name in a professional, supportive manner - like a knowledgeable secretary. "
            "Use direct references such as 'Bi has experience in...', 'Bi specializes in...', 'Bi worked on...'. "
            "Rely only on the provided context."
        )

    system_prompt = "\n".join(
        [
            SYSTEM_PROMPT.strip(),
            persona_instruction,
            "Provide a direct, concise answer without repeating the context.",
            "If the context lacks the answer, state that politely.",
            "Do not echo or list the context - synthesize it into a clear response.",
        ]
    )

    user_prompt = f"""Context:
{context}

Question: {original_question or question}

Provide a concise, professional answer based only on the context above."""

    combined_prompt = f"{system_prompt}\n\n{user_prompt}"

    if LLM_PROVIDER == "huggingface":
        return generate_response_huggingface(combined_prompt)
    elif LLM_PROVIDER == "local":
        return generate_response_local(system_prompt, user_prompt)
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


@app.get("/session-token")
async def session_token():
    """Issue a short-lived session token for client-side access."""
    if not SESSION_TOKEN_SECRET:
        raise HTTPException(status_code=500, detail="Session tokens are not configured")
    token = create_session_token()
    return {"token": token, "expires_in": SESSION_TOKEN_TTL_SECONDS}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, _: None = Depends(verify_client_access)):
    """Main chat endpoint with RAG"""
    try:
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(request.message)

        # Build context from chunks
        context = "\n\n".join(relevant_chunks)

        # Generate response
        response = generate_response(
            context,
            request.message,
            original_question=request.message,
        )

        return ChatResponse(
            response=response,
            context_used=relevant_chunks
        )
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "rag_initialized": embedding_model is not None,
        "llm_initialized": llm_client is not None or LLM_PROVIDER == "huggingface",
        "chunks_count": len(cv_chunks),
        "llm_provider": LLM_PROVIDER,
        "local_model_path": local_model_path if LLM_PROVIDER == "local" else None,
        "allowed_origins": allowed_origins,
        "token_protected": bool(API_ACCESS_TOKEN),
        "session_tokens_enabled": bool(SESSION_TOKEN_SECRET),
        "session_token_ttl": SESSION_TOKEN_TTL_SECONDS if SESSION_TOKEN_SECRET else None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
