"""
Configuration file for LLM provider
Change LLM_PROVIDER to switch between different models
"""

import os

# Swappable LLM provider (environment configurable)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")  # Options: "huggingface", "local"

# API Keys (set these as environment variables in HuggingFace Space secrets)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# Model configurations
HUGGINGFACE_MODEL = "google/gemma-2-2b-it"

# Local model configuration (for quantized models hosted within the Space)
LOCAL_MODEL_REPO = os.getenv("LOCAL_MODEL_REPO", "tensorblock/gemma-2-2b-it-GGUF")
LOCAL_MODEL_FILENAME = os.getenv("LOCAL_MODEL_FILENAME", "gemma-2-2b-it-Q4_K_M.gguf")
LOCAL_MODEL_CONTEXT_LENGTH = int(os.getenv("LOCAL_MODEL_CONTEXT_LENGTH", "4096"))
LOCAL_MODEL_THREADS = int(os.getenv("LOCAL_MODEL_THREADS", str(os.cpu_count() or 4)))
LOCAL_MODEL_BATCH_SIZE = int(os.getenv("LOCAL_MODEL_BATCH_SIZE", "256"))
LOCAL_MODEL_MAX_OUTPUT_TOKENS = int(os.getenv("LOCAL_MODEL_MAX_OUTPUT_TOKENS", "512"))
LOCAL_MODEL_HF_TOKEN = os.getenv("LOCAL_MODEL_HF_TOKEN", HUGGINGFACE_API_KEY or "")

# Access control configuration
CLIENT_APP_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CLIENT_APP_ORIGINS", "").split(",")
    if origin.strip()
]
API_ACCESS_TOKEN = os.getenv("API_ACCESS_TOKEN", "")

# Session token configuration
SESSION_TOKEN_SECRET = os.getenv("SESSION_TOKEN_SECRET", "")
SESSION_TOKEN_TTL_SECONDS = int(os.getenv("SESSION_TOKEN_TTL_SECONDS", "600"))

# RAG Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, lightweight
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks
TOP_K_RESULTS = 3  # Number of relevant chunks to retrieve

# System prompt for the chatbot
SYSTEM_PROMPT = """You are Bi's professional assistant, helping visitors learn about his background, skills, and experience.

Instructions:
- Refer to Bi directly by name (e.g., "Bi has experience in...", "Bi worked on...")
- Answer questions based ONLY on the provided context about Bi
- Be conversational, friendly, and professional - like a knowledgeable secretary
- If information is not in the context, politely say you don't have that information about Bi
- Keep responses concise but informative
- Speak on Bi's behalf in a supportive, professional manner
"""
