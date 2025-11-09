"""
Beam vLLM service for Qwen3 4B Instruct
Deploy with: beam deploy beam_vllm_service.py:qwen3_4b
"""

from beam import Volume, Image
from beam.integrations import VLLM, VLLMArgs

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
CACHE_PATH = "./model_cache"

# Create vLLM server with optimized caching
qwen3_4b = VLLM(
    name="qwen3-4b-instruct",
    cpu=4,
    memory="16Gi",
    gpu="RTX4090",  # RTX4090 is cheapest ($0.69/hr vs A10G $1.05/hr), plenty for 4B models
    gpu_count=1,
    workers=1,
    # Persistent volume for model caching - prevents re-downloading model on each cold start
    volumes=[
        Volume(
            name="qwen3-model-cache",
            mount_path=CACHE_PATH,
        )
    ],
    # Optional: Add HF_TOKEN secret for private model access
    # secrets=["HF_TOKEN"],
    vllm_args=VLLMArgs(
        model=MODEL_ID,
        served_model_name=[MODEL_ID],
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        enforce_eager=True,  # Faster cold starts - disables CUDA graphs for quicker initialization
        download_dir=CACHE_PATH,  # vLLM will download models to this cached volume
    )
)
