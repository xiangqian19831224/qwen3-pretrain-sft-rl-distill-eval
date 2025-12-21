# 启动 vLLM 服务器
model_path="../../model/sft_merge"
echo ${model_path}

python -m vllm.entrypoints.openai.api_server \
    --model ${model_path} \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096
