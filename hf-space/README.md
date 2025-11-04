---
title: CV Chatbot
sdk: docker
app_port: 7860
---

# CV Chatbot

RAG-based chatbot for answering questions about professional background and experience.

## Configuration

Set environment variables in Space secrets:

- `LLM_PROVIDER` - Set to `local` (default), `groq`, or `huggingface`
- `GROQ_API_KEY` - Required if using Groq
- `HUGGINGFACE_API_KEY` - Required if using HuggingFace Inference API
- `SESSION_TOKEN_SECRET` - Optional, for session auth
- `CLIENT_APP_ORIGINS` - Optional, comma-separated allowed origins
