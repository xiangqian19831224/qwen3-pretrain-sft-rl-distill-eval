#!/bin/bash

# VLLM 多机多卡分布式部署 - Worker节点
# 适用于 Qwen3-32B 模型

# 配置参数 (需要根据实际情况修改)
MODEL_PATH="../../model/sft_merge"
MASTER_IP="192.168.1.100"  # Master节点IP
MASTER_PORT="29500"
WORKER_RANK="1"            # Worker节点rank (从1开始，每个worker不同)
TENSOR_PARALLEL_SIZE="2"   # 每个节点的GPU数量
WORLD_SIZE="4"             # 总GPU数量

# 网络配置
export VLLM_USE_RAY=1
export RAY_BACKEND_LOG_LEVEL=info
export RAY_LOG_DIR=./ray_logs
export RAY_ADDRESS=${MASTER_IP}:6379

# 创建日志目录
mkdir -p ray_logs

echo "启动 VLLM Worker 节点..."
echo "Master IP: ${MASTER_IP}"
echo "Master Port: ${MASTER_PORT}"
echo "Worker Rank: ${WORKER_RANK}"
echo "Model Path: ${MODEL_PATH}"
echo "World Size: ${WORLD_SIZE}"
echo "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"

# 启动 Worker 节点
python -m vllm.worker.worker \
    --model ${MODEL_PATH} \
    --worker-address ${MASTER_IP}:${MASTER_PORT} \
    --world-size ${WORLD_SIZE} \
    --rank ${WORKER_RANK} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --distributed-executor-backend ray \
    --gpu-memory-utilization 0.85 \
    --dtype float16 \
    --trust-remote-code \
    --block-size 16 \
    --max-num-batched-tokens 8192 &

echo "Worker 节点 ${WORKER_RANK} 已启动"

# 保持进程运行
wait