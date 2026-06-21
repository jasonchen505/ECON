python -m vllm.entrypoints.openai.api_server \
  --model ~/models/Qwen3-0.6B \
  --served-model-name qwen3-0.6b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 4096
