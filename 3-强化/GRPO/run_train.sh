CUDA_VISIBLE_DEVICES=0 \
python grpo_train.py \
  --model_path ../../output/sft_merge \
  --output_dir ../../output/grpo_adapter \
  --dataset_path ../../data/dirty_chinese_dpo.json \
  --max_datasize 200 \
  --use_bf16 false
