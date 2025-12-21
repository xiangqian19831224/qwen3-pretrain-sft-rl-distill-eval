# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from trl import GRPOConfig, GRPOTrainer


@dataclass
class ScriptArguments:
    """GRPO训练脚本参数配置"""
    # 模型路径配置
    model_path: str = field(default="../../output/sft_merge", metadata={"help": "基础模型或SFT模型路径"})
    output_dir: str = field(default="../../output/grpo_adapter", metadata={"help": "GRPO LoRA适配器保存目录"})

    # 序列长度配置
    max_prompt_len: int = field(default=256, metadata={"help": "最大提示长度"})
    max_completion_len: int = field(default=128, metadata={"help": "最大生成长度"})
    num_generations: int = field(default=2, metadata={"help": "每个提示生成的候选数量"})

    # LoRA配置
    lora_r: int = field(default=8, metadata={"help": "LoRA的秩"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA的alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA的dropout"})

    # GRPO训练配置
    learning_rate: float = field(default=1e-5, metadata={"help": "学习率"})
    per_device_train_batch_size: int = field(default=1, metadata={"help": "每设备批次大小"})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "梯度累积步数"})
    num_train_epochs: int = field(default=1, metadata={"help": "训练轮数"})
    logging_steps: int = field(default=5, metadata={"help": "日志记录步数"})
    save_steps: int = field(default=50, metadata={"help": "模型保存步数"})

    # 数据集配置
    dataset_path: str = field(default="./data/dirty_chinese_dpo.json", metadata={"help": "数据集路径"})
    dataset_name: str = field(default="gsm8k", metadata={"help": "数据集名称"})
    dataset_subset: str = field(default="main", metadata={"help": "数据集子集"})
    max_train_samples: int = field(default=2000, metadata={"help": "最大训练样本数"})

    # 系统配置
    use_bf16: bool = field(default=True, metadata={"help": "是否使用bfloat16"})

    # 最大训练样本数，grpo训练比较慢，添加该参数方便测试
    max_datasize: int = field(default=100, metadata={"help": "训练使用样本数量"})


def build_prompt(example, system_prompt="你是一个智能助手，请根据用户问题提供准确和有用的回答。"):
    """
    构建 prompt，将问题放在模板里
    """
    # 从conversations中提取human的问题
    question = ""
    if "conversations" in example:
        for conv in example["conversations"]:
            if conv["from"] == "human":
                question = conv["value"]
                break
    else:
        question = example.get("question", "")
    
    return (
        f"{system_prompt}\n"
        f"用户：{question}\n"
        "助手："
    )


def preprocess(example, system_prompt="你是一个智能助手，请根据用户问题提供准确和有用的回答。"):
    """
    处理原始数据集，生成 prompt 和 reference
    """
    # 提取问题
    question = ""
    if "conversations" in example:
        for conv in example["conversations"]:
            if conv["from"] == "human":
                question = conv["value"]
                break
    else:
        question = example.get("question", "")
    
    # 提取参考答案
    reference = ""
    if "chosen" in example:
        reference = example["chosen"].get("value", "")
    elif "answer" in example:
        reference = example["answer"]
    
    return {
        "prompt": build_prompt(example, system_prompt),
        "reference": reference,
        "question": question  # 保存原始问题用于奖励函数
    }


def reward_fn(prompts, completions, completion_ids=None, **kwargs):
    """
    自定义 Reward 函数
    prompts: List[str] 输入 prompt
    completions: List[str] 模型生成文本
    completion_ids: token id（可忽略）
    返回 List[float] 奖励值
    """
    rewards = []
    for prompt, out in zip(prompts, completions):
        # 简单的奖励函数：基于生成长度和内容质量
        reward = 0.0
        
        # 基本奖励：生成内容不为空
        if out and len(out.strip()) > 0:
            reward += 0.5
        
        # 长度奖励：适中的生成长度
        if 10 <= len(out.strip()) <= 200:
            reward += 0.3
        
        # 质量奖励：包含中文内容
        if any('\u4e00' <= char <= '\u9fff' for char in out):
            reward += 0.2
        
        rewards.append(min(reward, 1.0))  # 最大奖励为1.0
    return rewards


# Dataset 包装
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
            "reference": item.get("reference", ""),
            "question": item.get("question", "")
        }


def main():
    """主训练函数"""
    # 解析命令行参数
    parser = HfArgumentParser(ScriptArguments)
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # 设置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 检查模型路径
    if not os.path.exists(args.model_path):
        raise ValueError(f"模型路径不存在: {args.model_path}")

    print("1. 加载 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("2. 加载模型")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if args.use_bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("3. 配置 LoRA")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    print("4. 应用 LoRA")
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False  # 训练时禁用缓存
    model.print_trainable_parameters()

    print("5. 加载和处理数据集")
    # 加载数据集
    import json
    
    try:
        # 尝试加载数据集文件
        with open(args.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 手动创建dataset对象
        from datasets import Dataset
        
        if isinstance(data, list):
            dataset = Dataset.from_list(data)
        else:
            dataset = Dataset.from_dict(data)
        
        print(f"成功加载数据集，共 {len(dataset)} 条数据")
        
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("尝试使用默认数据集...")
        try:
            dataset = load_dataset(args.dataset_name, args.dataset_subset)
        except Exception as e2:
            print(f"加载默认数据集也失败: {e2}")
            raise

    # 使用用户指定的max_datasize参数
    max_samples = min(getattr(args, 'max_datasize', 1000), len(dataset))
    train_ds = dataset.select(range(max_samples))
    print(f"使用 {max_samples} 条训练样本")

    # 定义系统提示
    system_prompt = "你是一个智能助手，请根据用户问题提供准确和有用的回答。"

    train_ds = train_ds.map(
        lambda x: preprocess(x, system_prompt),
        remove_columns=list(train_ds.column_names)
    )
    train_dataset = RewardDataset(train_ds)

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
        bf16=args.use_bf16,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_len,
        max_completion_length=args.max_completion_len,
    )

    print("7. 创建 GRPO Trainer")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_fn]  # 注意这里是 list
    )

    print("8. 开始训练")
    trainer.train()

    print("9. 保存 LoRA 模型")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA模型已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
