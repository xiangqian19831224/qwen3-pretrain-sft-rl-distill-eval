#!/bin/bash

# VLLM 多机多卡分布式部署 - Master节点
# 适用于 Qwen3-32B 模型

# 配置参数
MODEL_PATH="../../output/sft_merge"
MASTER_IP="192.168.0.107"  # 请替换为实际的主节点IP
MASTER_PORT="29500"
WORLD_SIZE="4"             # 总GPU数量 (根据实际配置修改)
TENSOR_PARALLEL_SIZE="1"    # 每个节点的GPU数量
PIPELINE_PARALLEL_SIZE="1"  # 流水并行大小

# 网络配置
export VLLM_USE_RAY=1
export RAY_BACKEND_LOG_LEVEL=info
export RAY_LOG_DIR=./ray_logs

# 创建日志目录
mkdir -p ray_logs

echo "启动 VLLM Master 节点..."
echo "Master IP: ${MASTER_IP}"
echo "Master Port: ${MASTER_PORT}"
echo "Model Path: ${MODEL_PATH}"
echo "World Size: ${WORLD_SIZE}"
echo "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
echo "Pipeline Parallel Size: ${PIPELINE_PARALLEL_SIZE}"

# 启动 Master 节点
python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
    --worker-address ${MASTER_IP}:${MASTER_PORT} \
    --world-size ${WORLD_SIZE} \
    --rank 0 \
    --distributed-executor-backend ray \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 256 \
    --max-model-len 8192 \
    --dtype float16 \
    --trust-remote-code \
    --enable-prefix-caching \
    --enforce-eager &

# 等待服务启动
sleep 10

echo "Master 节点已启动，监听端口: 8000"
echo "API 端点: http://${MASTER_IP}:8000"
echo "健康检查: curl http://${MASTER_IP}:8000/health"

# 保持进程运行
wait