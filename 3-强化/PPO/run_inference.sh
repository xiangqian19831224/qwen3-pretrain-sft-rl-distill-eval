#!/bin/bash

# PPO模型推理启动脚本
# 使用方法: ./run_ppo_inference.sh [mode]

# 默认参数配置
MODEL_PATH="../../output/sft_merge"
ADAPTER_PATH="../../output/ppo_adapter"
MODE=${1:-"interactive"}
TEST_OUTPUT_FILE="ppo_test_results.json"

# 检查必要路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 基础模型路径不存在: $MODEL_PATH"
    exit 1
fi

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "❌ 错误: PPO适配器路径不存在: $ADAPTER_PATH"
    echo "💡 提示: 请先运行训练脚本生成PPO适配器"
    exit 1
fi

echo "🚀 启动PPO模型推理..."
echo "📋 配置信息:"
echo "   - 基础模型路径: $MODEL_PATH"
echo "   - PPO适配器路径: $ADAPTER_PATH"
echo "   - 运行模式: $MODE"
echo "   - 测试输出文件: $TEST_OUTPUT_FILE"
echo ""

# 根据模式选择不同的参数
if [ "$MODE" = "test" ]; then
    echo "🎯 运行批量测试模式..."
    python ppo_inference.py \
        --model_path "$MODEL_PATH" \
        --adapter_path "$ADAPTER_PATH" \
        --mode test \
        --test_output_file "$TEST_OUTPUT_FILE"
else
    echo "💬 运行交互式对话模式..."
    python ppo_inference.py \
        --model_path "$MODEL_PATH" \
        --adapter_path "$ADAPTER_PATH" \
        --mode interactive
fi

echo ""
echo "✅ PPO推理运行完成！"