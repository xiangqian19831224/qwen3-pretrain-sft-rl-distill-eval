# 部署 Qwen 模型（假设模型在 model/ 目录）
model_path="../../model/sft_merge"
echo ${model_path}

python -m vllm.entrypoints.openai.api_server \
    --model ${model_path} \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --temperature 0.7 \
    --top-p 0.9
