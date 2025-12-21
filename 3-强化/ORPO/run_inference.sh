#!/bin/bash

# ORPO模型推理启动脚本
# 使用方法: ./inference.sh

echo "=== ORPO模型推理启动脚本 ==="

# 默认参数配置
MODEL_PATH="../../model/Qwen/Qwen3-0.6B"  # SFT合并后的基础模型路径
ADAPTER_PATH="../../output/orpo"  # ORPO适配器路径
MODE="interactive"  # 运行模式: interactive 或 test
TEST_OUTPUT_FILE="orpo_test_results.json"  # 测试结果输出文件

# 检查路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌错误: 基础模型路径不存在: $MODEL_PATH"
    echo "请检查SFT合并后的模型路径是否正确"
    exit 1
fi

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "❌错误: ORPO适配器路径不存在: $ADAPTER_PATH"
    echo "请检查ORPO训练后的适配器路径是否正确"
    exit 1
fi

# 创建输出目录
mkdir -p "$(dirname "$TEST_OUTPUT_FILE")"

echo "🚀 启动ORPO模型推理..."
echo "基础模型路径: $MODEL_PATH"
echo "ORPO适配器路径: $ADAPTER_PATH"
echo "运行模式: $MODE"
echo "测试输出文件: $TEST_OUTPUT_FILE"
echo ""

# 启动推理
python orpo_inference.py \
    --model_path "$MODEL_PATH" \
    --adapter_path "$ADAPTER_PATH" \
    --mode "$MODE" \
    --test_output_file "$TEST_OUTPUT_FILE"

echo ""
echo "✅ ORPO推理完成"