#!/bin/bash

# DPO推理启动脚本
# 使用方法: ./inference.sh [interactive|test]

# 设置默认参数
MODE=${1:-interactive}
MODEL_PATH="../../output/sft_merge"  # SFT合并后的模型路径
ADAPTER_PATH="../../output/dpo_adapter"  # DPO适配器路径
TEST_OUTPUT_FILE="dpo_test_results.json"  # 测试结果输出文件

echo "🚀 启动DPO推理..."
echo "运行模式: $MODE"
echo "基础模型路径: $MODEL_PATH"
echo "DPO适配器路径: $ADAPTER_PATH"

# 检查必要路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 基础模型路径不存在: $MODEL_PATH"
    exit 1
fi

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "❌ 错误: DPO适配器路径不存在: $ADAPTER_PATH"
    echo "请先运行DPO训练脚本生成适配器"
    exit 1
fi

# 运行推理
if [ "$MODE" = "test" ]; then
    echo "🧪 运行批量测试模式..."
    python dpo_inference.py \
        --model_path "$MODEL_PATH" \
        --adapter_path "$ADAPTER_PATH" \
        --mode test \
        --test_output_file "$TEST_OUTPUT_FILE"
else
    echo "💬 运行交互式对话模式..."
    python dpo_inference.py \
        --model_path "$MODEL_PATH" \
        --adapter_path "$ADAPTER_PATH" \
        --mode interactive
fi

echo "✅ DPO推理完成!"