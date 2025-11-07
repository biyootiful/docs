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
LOCAL_MODEL_REPO = os.getenv("LOCAL_MODEL_REPO", "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF")
LOCAL_MODEL_FILENAME = os.getenv("LOCAL_MODEL_FILENAME", "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf")  # Q4_K_M (2.50GB, recommended)
LOCAL_MODEL_CONTEXT_LENGTH = int(os.getenv("LOCAL_MODEL_CONTEXT_LENGTH", "2048"))
LOCAL_MODEL_THREADS = int(os.getenv("LOCAL_MODEL_THREADS", str(os.cpu_count() or 2)))  # HF Spaces has 2 vCPUs
LOCAL_MODEL_BATCH_SIZE = int(os.getenv("LOCAL_MODEL_BATCH_SIZE", "512"))  # Increased for better throughput
LOCAL_MODEL_MAX_OUTPUT_TOKENS = int(os.getenv("LOCAL_MODEL_MAX_OUTPUT_TOKENS", "200"))
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
CHUNK_SIZE = 300  # Characters per chunk (reduced for faster inference)
CHUNK_OVERLAP = 30  # Overlap between chunks
TOP_K_RESULTS = 1  # Fewer chunks lowers prompt size on small CPU tiers

# System prompt for the chatbot
SYSTEM_PROMPT = """Answer questions about Bi using the provided context. Keep answers short and direct. Always refer to Bi by name."""
