"""
Configuration file for LLM provider
Change LLM_PROVIDER to switch between different models
"""

import os

# Swappable LLM provider
LLM_PROVIDER = "huggingface"  # Options: "groq", "huggingface", "openai"

# API Keys (set these as environment variables in HuggingFace Space secrets)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model configurations
GROQ_MODEL = "mixtral-8x7b-32768"  # Fast and good quality
# GROQ_MODEL = "llama3-8b-8192"  # Alternative: faster but slightly lower quality

HUGGINGFACE_MODEL = "google/gemma-2-2b-it"
OPENAI_MODEL = "gpt-3.5-turbo"

# RAG Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, lightweight
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks
TOP_K_RESULTS = 3  # Number of relevant chunks to retrieve

# System prompt for the chatbot
SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a person's background, skills, and experience based on their CV/bio.

Instructions:
- Answer questions based ONLY on the provided context
- Be conversational and friendly
- If information is not in the context, politely say you don't have that information
- Keep responses concise but informative
- Act as if you're representing the person professionally
"""
