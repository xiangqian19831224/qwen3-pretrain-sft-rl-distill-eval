# 基于lora微调的grpo训练
import json
import os
from dataclasses import dataclass, field

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    HfArgumentParser
)
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer


@dataclass
class ScriptArguments:
    model_path: str = field(default="../../output/sft_merge", metadata={"help": "SFT合并后的模型路径"})
    dataset_path: str = field(default="../../data/dirty_chinese_dpo.json", metadata={"help": "数据集路径"})
    output_dir: str = field(default="../../output/grpo_adapter", metadata={"help": "GRPO LoRA适配器保存目录"})
    system_prompt: str = field(default="你是一个有用的AI助手，请根据用户的问题提供准确、有帮助的回答。",
                               metadata={"help": "系统提示语"})

    # LoRA配置
    lora_r: int = field(default=8, metadata={"help": "LoRA的秩"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA的alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA的dropout"})

    # GRPO配置
    learning_rate: float = field(default=1e-5, metadata={"help": "GRPO学习率"})
    max_prompt_length: int = field(default=512, metadata={"help": "最大提示长度"})
    max_completion_length: int = field(default=64, metadata={"help": "最大生成长度"})
    num_generations: int = field(default=2, metadata={"help": "生成数量"})
    
    # 训练配置
    per_device_train_batch_size: int = field(default=1, metadata={"help": "每个设备的训练批次大小"})
    gradient_accumulation_steps: int = field(default=8, metadata={"help": "梯度累积步数"})
    num_train_epochs: int = field(default=1, metadata={"help": "训练轮数"})
    logging_steps: int = field(default=10, metadata={"help": "日志记录步数"})
    save_steps: int = field(default=500, metadata={"help": "保存步数"})
    max_datasize: int = field(default=100, metadata={"help": "训练使用样本数量"})
    bf16: bool = field(default=True, metadata={"help": "是否使用bf16"})


def load_prompts(dataset_path, tokenizer, system_prompt):
    """加载并处理数据集，返回GRPO所需的prompt格式"""
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 数据集文件未找到 at {dataset_path}")
        exit()

    prompts = []
    for item in data:
        if 'conversations' in item:
            # 提取用户输入
            user_input = ""
            for turn in item['conversations']:
                if turn.get('from') == 'human':
                    user_input = turn['value']
                    break
            
            if user_input:
                # 构建GRPO格式的prompt
                formatted_prompt = f"{system_prompt}\n\n问题：{user_input}\n答案："
                prompts.append({
                    "prompt": formatted_prompt,
                    "reference": item.get("chosen", {}).get("value", "") if "chosen" in item else ""
                })
    return prompts


def reward_fn(prompts, completions, completion_ids=None, **kwargs):
    """
    GRPO的奖励函数
    prompts: List[str] 输入 prompt
    completions: List[str] 模型生成文本
    completion_ids: token id（可忽略）
    返回 List[float] 奖励值
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # 简单的奖励策略：如果生成了内容则奖励1，否则0
        # 可以根据需要添加更复杂的奖励逻辑
        if len(completion.strip()) > 0:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


class GRPODataset(Dataset):
    """GRPO训练数据集包装器"""
    def __init__(self, prompts_data):
        self.prompts_data = prompts_data

    def __len__(self):
        return len(self.prompts_data)

    def __getitem__(self, idx):
        item = self.prompts_data[idx]
        return {
            "prompt": item["prompt"],
            "reference": item["reference"]
        }


def main():
    parser = HfArgumentParser(ScriptArguments)
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # --- 路径检查 ---
    for path in [args.model_path, args.dataset_path]:
        if not os.path.exists(path):
            raise ValueError(f"输入路径不存在: {path}")

    print("1. 设置环境变量")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("2. 加载 Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("3. 加载并处理 GRPO 数据集")
    all_prompts = load_prompts(args.dataset_path, tokenizer, args.system_prompt)
    if len(all_prompts) > args.max_datasize:
        all_prompts = all_prompts[:args.max_datasize]

    train_dataset = GRPODataset(all_prompts)

    print(f"数据集加载完成，共 {len(train_dataset)} 条样本")

    print("4. 加载模型")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.enable_input_require_grads()

    print("5. 配置 LoRA")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj",
            "o_proj", "gate_proj",
            "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.config.use_cache = False  # 训练时禁用缓存
    model.print_trainable_parameters()

    print("6. 配置 GRPO")
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        bf16=args.bf16,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        report_to="none",  # 可以改为 "tensorboard" 或其他
    )

    print("7. 创建 GRPOTrainer")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],  # 注意这里是 list
    )

    print("8. 开始训练")
    trainer.train()

    print(f"9. 保存 GRPO LoRA 到: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("🎉 GRPO 训练完成")


if __name__ == "__main__":
    main()