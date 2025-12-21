python lora_merge.py

python lora_merge.py \
    --base_model_path "../model/Qwen/Qwen3-0.6B" \
    --lora_adapter_path "./output/sft_adapter" \
    --merged_output_path "./output/sft_merge" \
    --device_map "cpu" \
    --torch_dtype "bfloat16"
