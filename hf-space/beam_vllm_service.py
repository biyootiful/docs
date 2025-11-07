"""
Beam vLLM service for Qwen3 4B Instruct
Deploy with: beam deploy beam_vllm_service.py:qwen3_4b
"""

from beam.integrations import VLLM, VLLMArgs

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# Create vLLM server
qwen3_4b = VLLM(
    name="qwen3-4b-instruct",
    cpu=4,
    memory="16Gi",
    gpu="RTX4090",  # RTX4090 is cheapest ($0.69/hr vs A10G $1.05/hr), plenty for 4B models
    gpu_count=1,
    workers=1,
    vllm_args=VLLMArgs(
        model=MODEL_ID,
        served_model_name=[MODEL_ID],
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        enforce_eager=True,  # Faster cold starts
    )
)
