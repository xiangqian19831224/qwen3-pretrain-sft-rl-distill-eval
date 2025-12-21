# -*- coding: utf-8 -*-
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# =====================
# 基本配置  基于基础模型或sft模型训练
# 最好选择  基于sft模型训练
# =====================
MODEL_NAME = "../../output/sft_merge"
OUTPUT_DIR = "../../output/grpo_full"

MAX_PROMPT_LEN = 512
MAX_COMPLETION_LEN = 256
NUM_GENERATIONS = 4

# =====================
# 数据处理
# =====================
def build_prompt(example):
    """
    构建 prompt，将问题放在模板里
    """
    return (
        "你是一个擅长数学推理的助手，请一步一步思考并给出最终答案。\n"
        f"问题：{example['question']}\n"
        "答案："
    )

def preprocess(example):
    """
    处理原始数据集，生成 prompt 和 reference
    """
    return {
        "prompt": build_prompt(example),
        "reference": example["answer"]
    }

# =====================
# 自定义 Reward 函数
# =====================
def reward_fn(prompts, completions, completion_ids=None, **kwargs):
    """
    prompts: List[str] 输入 prompt
    completions: List[str] 模型生成文本
    completion_ids: token id（可忽略）
    返回 List[float] 奖励值
    """
    rewards = []
    for prompt, out in zip(prompts, completions):
        # 提取参考答案
        if "答案：" in prompt:
            ref = prompt.split("答案：")[-1].strip()
        else:
            ref = ""
        # 奖励：参考答案在生成文本中则奖励 1，否则 0
        rewards.append(1.0 if ref in out else 0.0)
    return rewards

# =====================
# Dataset 包装
# =====================
from torch.utils.data import Dataset

class RewardDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            "prompt": item["prompt"],
            "reference": item["reference"]
        }

# =====================
# 主流程
# =====================
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 1. tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3. dataset
    dataset = load_dataset("gsm8k", "main")
    train_ds = dataset["train"].select(range(2000))  # 取前2000条作为示例
    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    train_dataset = RewardDataset(train_ds)

    # 4. GRPO 配置
    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        bf16=True,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LEN,
        max_completion_length=MAX_COMPLETION_LEN,
    )

    # 5. GRPO Trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_fn]  # 注意这里是 list
    )

    # 6. 开始训练
    trainer.train()

    # 7. 保存模型
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"模型已保存到 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
