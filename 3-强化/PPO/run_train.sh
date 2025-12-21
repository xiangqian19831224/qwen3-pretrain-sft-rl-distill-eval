python ppo_train.py \
    --model_path ../../output/sft_merge \
    ---rm_path ../../output/rm_adapter \
    --output_dir ../../output/ppo_adapter \
    --dataset_path ../../data/dirty_chinese_dpo.json \
    --max_datasize 10000
