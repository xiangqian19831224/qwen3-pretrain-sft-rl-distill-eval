#!/bin/bash

# GRPO推理启动脚本
# 使用示例:
# ./grpo_inference.sh                    # 交互式模式 (需要修改脚本中的路径)
# ./grpo_inference.sh test               # 批量测试模式 (需要修改脚本中的路径)

# 设置模型路径 - 请根据实际情况修改
SFT_MODEL_PATH="../../output/sft_merge"
GRPO_ADAPTER_PATH="../../output/grpo_adapter"

# 检查路径是否存在
if [ ! -d "$SFT_MODEL_PATH" ]; then
    echo "❌ 错误: SFT模型路径不存在: $SFT_MODEL_PATH"
    echo "请修改脚本中的 SFT_MODEL_PATH 变量"
    exit 1
fi

if [ ! -d "$GRPO_ADAPTER_PATH" ]; then
    echo "❌ 错误: GRPO适配器路径不存在: $GRPO_ADAPTER_PATH"
    echo "请修改脚本中的 GRPO_ADAPTER_PATH 变量"
    exit 1
fi

# 根据参数选择模式
MODE=${1:-interactive}

echo "🚀 启动GRPO推理..."
echo "   SFT模型路径: $SFT_MODEL_PATH"
echo "   GRPO适配器路径: $GRPO_ADAPTER_PATH"
echo "   运行模式: $MODE"
echo ""

if [ "$MODE" = "test" ]; then
    # 批量测试模式
    python grpo_inference.py \
        --model_path "$SFT_MODEL_PATH" \
        --adapter_path "$GRPO_ADAPTER_PATH" \
        --mode test \
        --test_output_file "grpo_test_results.json"
else
    # 交互式模式（默认）
    python grpo_inference.py \
        --model_path "$SFT_MODEL_PATH" \
        --adapter_path "$GRPO_ADAPTER_PATH" \
        --mode interactive
fi

echo ""
echo "✅ GRPO推理完成！"