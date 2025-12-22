# 使用 n 个 GPU
# --tensor-parallel-size n
model_path="../../output/sft_merge"
echo ${model_path}

python -m vllm.entrypoints.openai.api_server \
    --model ${model_path} \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port 8801 \
    --gpu-memory-utilization 0.9
