import json
import os
from dataclasses import dataclass, field

import swanlab
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, HfArgumentParser
)
from trl import PPOConfig, PPOTrainer


@dataclass
class ScriptArguments:
    model_path: str = field(default="../../output/sft_merge", metadata={"help": "SFT合并后的模型路径"})
    rm_path: str = field(default="../../output/rm_adapter", metadata={"help": "RM LoRA适配器路径"})
    dataset_path: str = field(default="../../data/dirty_chinese_dpo.json", metadata={"help": "数据集路径"})
    output_dir: str = field(default="../../output/ppo_adapter", metadata={"help": "PPO LoRA适配器保存目录"})
    system_prompt: str = field(default="你是一个有用的AI助手，请根据用户的问题提供准确、有帮助的回答。",
                               metadata={"help": "系统提示语"})

    # LoRA配置
    lora_r: int = field(default=8, metadata={"help": "LoRA的秩"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA的alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA的dropout"})

    # PPO配置
    learning_rate: float = field(default=1e-5, metadata={"help": "PPO学习率"})
    kl_coef: float = field(default=0.2, metadata={"help": "KL散度惩罚系数"})
    max_prompt_length: int = field(default=512, metadata={"help": "最大提示长度"})

    # 设备配置
    policy_device: str = field(default="cuda:0", metadata={"help": "策略模型和参考模型所在的设备"})
    reward_device: str = field(default="cuda:0", metadata={"help": "奖励模型和价值模型所在的设备"})

    max_datasize: int = field(default=True, metadata={"help": "训练使用样本数量"})

    use_swanlab: bool = field(default=True, metadata={"help": "是否使用SwanLab"})


def setup_swanlab(args: ScriptArguments):
    if not args.use_swanlab:
        return
    os.environ["SWANLAB_PROJECT"] = "qwen3-rl-ppo"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    swanlab.init(
        project="qwen3-sft-ppo",
        run_name="ppo-training",
        config=vars(args),
        mode="offline"
    )


def load_prompts(dataset_path, tokenizer, system_prompt):
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 数据集文件未找到 at {dataset_path}")
        exit()

    prompts = []
    for item in data:
        if 'conversations' in item:
            human_input = "".join(
                [turn['value'] + "\n" for turn in item['conversations'] if turn.get('from') == 'human']).strip()
            if human_input:
                # dd_generation_prompt=True 又额外插入了一次 assistant 起始 token
                # 实际效果
                #     system
                #     user
                #     assistant
                #     assistant   <-- 重复
                formatted_prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": human_input},
                        {"role": "assistant", "content": ""},  # <--与RM模型对齐{"role": "assistant", "content": ""}
                    ],
                    tokenize=False,
                    add_generation_prompt=False
                )
                prompts.append({"query": formatted_prompt})
    return prompts


def main():
    parser = HfArgumentParser(ScriptArguments)
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # --- 路径检查 ---
    for path in [args.model_path, args.rm_path, args.dataset_path]:
        if not os.path.exists(path):
            raise ValueError(f"输入路径不存在: {path}")

    print("1. 初始化 SwanLab")
    setup_swanlab(args)

    print("2. 加载 Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("3. 加载并处理 PPO 数据集")
    all_prompts = load_prompts(args.dataset_path, tokenizer, args.system_prompt)
    if len(all_prompts) > args.max_datasize:
        all_prompts = all_prompts[:args.max_datasize]

    train_dataset = Dataset.from_list(all_prompts)

    def tokenize_fn(example):
        return tokenizer(
            example["query"],
            truncation=True,
            max_length=args.max_prompt_length,
        )

    train_dataset = train_dataset.map(tokenize_fn, batched=False)
    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
    )

    print("4. 配置 PPOConfig（关键：关闭 sample generation）")
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_ppo_epochs=4,
        num_train_epochs=1,
        output_dir="./Output/ppo_model_temp",
        gradient_checkpointing=True,
        kl_coef=args.kl_coef,
        report_to="swanlab" if args.use_swanlab else "none",

        # ====== 关键修复点 ======
        num_sample_generations=0,

        # ===== 日志相关 =====
        logging_steps=100,  # 每 1 个 PPO step 打一次日志（调试期很爽）
        log_level="info",
    )

    print("5. 构建 Policy Model（LoRA）")
    policy_lora = LoraConfig(
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=args.policy_device,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.enable_input_require_grads()
    model = get_peft_model(model, policy_lora)
    model.config.use_cache = False
    model.print_trainable_parameters()

    print("6. 构建 Reference Model")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=args.policy_device,
    )
    ref_model.config.pad_token_id = tokenizer.pad_token_id
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    print("7. 加载 Reward Model")
    rm_base = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=1,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=args.reward_device,
    )
    rm_base.config.pad_token_id = tokenizer.pad_token_id
    reward_model = PeftModel.from_pretrained(rm_base, args.rm_path)
    reward_model.eval()

    print("8. 构建 Value Model（LoRA）")
    value_lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=[
            "q_proj", "k_proj", "v_proj",
            "o_proj", "gate_proj",
            "up_proj", "down_proj",
        ],
    )

    value_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=1,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=args.policy_device,  # 改为与policy设备一致
    )
    value_model.config.pad_token_id = tokenizer.pad_token_id
    value_model = get_peft_model(value_model, value_lora)
    value_model.print_trainable_parameters()

    print("9. 创建 PPOTrainer 并训练")
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        model=model,
        ref_model=ref_model,  # 修复：添加参考模型
        reward_model=reward_model,
        value_model=value_model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        # eval_dataset=train_dataset.select(range(32))， # 如果需要传评估数据的话
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    ppo_trainer.train()

    print(f"9. 保存 PPO LoRA 到: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    ppo_trainer.save_model(args.output_dir)

    if args.use_swanlab:
        swanlab.finish()

    print("🎉 PPO 训练完成")


if __name__ == "__main__":
    main()
