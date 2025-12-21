import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# ===== 配置 =====
MODEL_NAME = "../../model.Qwen/Qwen3-0.6B"              # 基础模型
DATA_PATH = "data/data.txt"                 # 增量预训练语料
OUTPUT_DIR = "Output/Qwen/pretrain"         # 输出目录

# ===== 加载分词器 =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===== 加载数据集 =====
dataset = load_dataset("text", data_files=DATA_PATH)

# ===== Tokenize 函数 =====
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=1024)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ===== 数据整理器（自回归语言模型） =====
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ===== 加载模型 =====
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# ===== 训练参数 =====
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_steps=100,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    deepspeed="ds_config.json",  # DeepSpeed 配置文件
    report_to="none"
)

# ===== Trainer =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ===== 开始训练 =====
trainer.train()

# ===== 保存模型 =====
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
