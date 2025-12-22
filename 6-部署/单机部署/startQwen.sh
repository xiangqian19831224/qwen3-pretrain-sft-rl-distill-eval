model_path="../../output/sft_merge"
echo ${model_path}

python -m vllm.entrypoints.openai.api_server \
    --model ${model_path} \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8801 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192
