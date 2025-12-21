# ============================================================
# Reward Model Training (Pairwise, PPO-aligned) for Qwen3
# transformers >= 4.57.3
# ============================================================

import os
import json
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from peft import LoraConfig, get_peft_model
import swanlab


# =========================
# Arguments
# =========================

@dataclass
class ScriptArguments:
    model_path: str = field(default="../../output/sft_merge")
    dataset_path: str = field(default="../../data/dirty_chinese_dpo.json")
    output_dir: str = field(default="../../output/rm_adapter")
    system_prompt: str = field(
        default="你是一个精通脏话的助手，请用不敬的、攻击性的语言回答问题"
    )
    max_prompt_length: int = field(default=512)  # 【修改】与 PPO prompt length 对齐
    max_response_length: int = field(default=128)  # 【新增】只训练 early continuation
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.1)
    use_swanlab: bool = field(default=True)
    max_datasize: int = field(default=500)  # 最大加载训练数据


# =========================
# SwanLab
# =========================

def setup_swanlab(args):
    if not args.use_swanlab:
        return
    os.environ["SWANLAB_PROJECT"] = "qwen3-rl-rm"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    swanlab.init(
        project="qwen3-sft-rm-ppo-chinese",
        run_name="rm-pairwise-training",
        config=vars(args),
    )


# =========================
# Dataset
# =========================

def load_preference_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = []
    for item in raw:
        if "conversations" not in item:
            continue
        if "chosen" not in item or "rejected" not in item:
            continue

        prompt = "\n".join(
            turn["value"]
            for turn in item["conversations"]
            if turn.get("from") == "human"
        ).strip()

        chosen = item["chosen"].get("value", "").strip()
        rejected = item["rejected"].get("value", "").strip()

        if prompt and chosen and rejected:
            data.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

    return Dataset.from_list(data)


# =========================
# Preprocess (PPO-aligned)
# =========================

def build_preprocess_fn(
        tokenizer,
        system_prompt,
        max_prompt_length,
        max_response_length,
):
    def truncate_by_tokens(text, max_tokens):
        # 【新增】按 token 截断，保证 continuation 分布与 PPO 一致
        ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens,
        )["input_ids"]
        return tokenizer.decode(ids)

    def build_prompt(user_text):
        # 【修改】assistant 留空，模拟 PPO rollout 的 generation 起点
        # 效果： “把 think 暴露在 reward 约束之下”
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": ""},  # 👈 关键
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

    def encode(user, answer):
        prompt_text = build_prompt(user)

        # 【新增】只使用 early continuation，避免 RM 学“整篇作文”
        answer = truncate_by_tokens(answer, max_response_length)

        full_text = prompt_text + answer

        enc = tokenizer(
            full_text,
            truncation=True,
            max_length=max_prompt_length + max_response_length,
        )

        return enc["input_ids"], enc["attention_mask"]

    def preprocess(examples):
        input_ids_chosen, attention_chosen = [], []
        input_ids_rejected, attention_rejected = [], []

        for p, c, r in zip(
                examples["prompt"],
                examples["chosen"],
                examples["rejected"],
        ):
            ids_c, mask_c = encode(p, c)
            ids_r, mask_r = encode(p, r)

            input_ids_chosen.append(ids_c)
            attention_chosen.append(mask_c)
            input_ids_rejected.append(ids_r)
            attention_rejected.append(mask_r)

        return {
            "input_ids_chosen": input_ids_chosen,
            "attention_mask_chosen": attention_chosen,
            "input_ids_rejected": input_ids_rejected,
            "attention_mask_rejected": attention_rejected,
        }

    return preprocess


# =========================
# Data Collator
# =========================

@dataclass
class PairwiseDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict]):
        chosen = [
            {
                "input_ids": f["input_ids_chosen"],
                "attention_mask": f["attention_mask_chosen"],
            }
            for f in features
        ]
        rejected = [
            {
                "input_ids": f["input_ids_rejected"],
                "attention_mask": f["attention_mask_rejected"],
            }
            for f in features
        ]

        batch_chosen = self.tokenizer.pad(
            chosen, padding=True, return_tensors="pt"
        )
        batch_rejected = self.tokenizer.pad(
            rejected, padding=True, return_tensors="pt"
        )

        return {
            "chosen": batch_chosen,
            "rejected": batch_rejected,
        }


# =========================
# Trainer
# =========================

class RewardTrainerHF(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        chosen = inputs["chosen"]
        rejected = inputs["rejected"]

        out_c = model(
            input_ids=chosen["input_ids"],
            attention_mask=chosen["attention_mask"],
        )
        out_r = model(
            input_ids=rejected["input_ids"],
            attention_mask=rejected["attention_mask"],
        )

        score_c = out_c.logits.squeeze(-1)
        score_r = out_r.logits.squeeze(-1)

        loss = -F.logsigmoid(score_c - score_r).mean()

        if return_outputs:
            return loss, {
                "score_chosen": score_c,
                "score_rejected": score_r,
            }

        return loss


# =========================
# Main
# =========================

def main():
    parser = HfArgumentParser(ScriptArguments)
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    setup_swanlab(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_preference_dataset(args.dataset_path)

    # 安全截断数据集，避免超出实际大小
    actual_size = len(dataset)
    target_size = min(args.max_datasize, actual_size)
    dataset = dataset.select(range(target_size))

    dataset = dataset.train_test_split(test_size=0.1)

    preprocess_fn = build_preprocess_fn(
        tokenizer,
        args.system_prompt,
        args.max_prompt_length,  # 【修改】prompt / response 解耦
        args.max_response_length,
    )

    train_ds = dataset["train"].map(
        preprocess_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    eval_ds = dataset["test"].map(
        preprocess_fn,
        batched=True,
        remove_columns=dataset["test"].column_names,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=1,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_cfg)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="swanlab" if args.use_swanlab else "none",
        remove_unused_columns=False,
    )

    trainer = RewardTrainerHF(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PairwiseDataCollator(tokenizer),
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    if args.use_swanlab:
        swanlab.finish()


if __name__ == "__main__":
    main()
