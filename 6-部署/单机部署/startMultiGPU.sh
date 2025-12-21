# 使用 4 个 GPU
model_path="../../model/sft_merge"
echo ${model_path}

python -m vllm.entrypoints.openai.api_server \
    --model ${model_path} \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9
