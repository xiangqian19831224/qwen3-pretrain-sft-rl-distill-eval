### 作者信息

    联系人： 刘向前  
    微信：   13552482980
    QQ:     1012088761 

### 部分参考(方便的话帮他们点个赞)

- [Qwen3微调演练平台](https://github.com/lijiayi-ai/Qwen3-FineTuning-Playground) — Qwen3微调演练平台
- [Qwen3医学推理项目](https://github.com/18520339/multi-reward-medical-reasoning) — 医学推理多奖励相关代码
- [Qwen3模型架构](https://zhuanlan.zhihu.com/p/1905976602019464591) — Qwen3模型架构
- [Qwen3增量预训练](https://blog.csdn.net/hhhhhhhhhhwwwwwwwwww/article/details/148145089) — Qwen3增量预训练
- [Qwen3大模型微调](https://developer.aliyun.com/article/1663178) — Qwen3大模型微调
- [SFT与DPO训练全流程](https://blog.csdn.net/gitblog_00831/article/details/150752889) — SFT与DPO训练全流程

### 1.配置环境

    调试平台： rtx5060ti

```bash
conda create -n qwen3_ft python=3.10
conda activate qwen3_ft
pip install -r requirements.txt
```

### 2.模型下载

```bash
  modelscope download --model Qwen/Qwen3-0.6B --local_dir ./model/Qwen/Qwen3-0.6B
```

```bash
    modelscope download --model Qwen/Qwen3-1.7B --local_dir ./model/Qwen/Qwen3-1.7B
```

### 3.增量预训练

#### 基于llamafactory进行增量预训练

    进入到目录 1-增量预训练/基于llamafactory/

```bash
# 安装llamafacotry

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

```bash
# 全参数增量预训练
sh full_pretrain.sh
```

```bash
# lora增量预训练
sh lora_pretrain.sh

# 模型合并
sh lora_merge.sh
```

#### 基于transformers全参数增量预训练

```bash
# 全参数增量预训练
sh train.sh
```

### 4.微调

```bash
# lora微调 项目根目录下执行

python 2-微调/lora_train.py \
    --model_path ./model/Qwen/Qwen3-0.6B \
    --output_dir ./output/sft_adapter \
    --dataset_path data/dirty_chinese_dpo.json    
```

```bash
# lora合并

python 2-微调/lora_merge.py \
    --base_model_path ./model/Qwen/Qwen3-0.6B \
    --lora_adapter_path "./output/sft_adapter" \
    --merged_output_path "./output/sft_merge" \
    --device_map "cpu" \
    --torch_dtype "bfloat16"
```

### 5. 强化学习

    需要注意强化学习的运行都是基于”4.微调的执行“

#### 5.1 DPO训练

```bash
# 基于lora结构，进行dpo训练

python 3-强化/DPO/dpo_train.py \
    --model_path ./model/sft_merge \
    --output_dir output/dpo_adapter \
    --dataset_path ./data/dirty_chinese_dpo.json    
```

```bash
# 合并模型

python 2-微调/lora_merge.py \
    --base_model_path ./model/Qwen/Qwen3-0.6B \
    --lora_adapter_path "./output/dpo_adapter" \
    --merged_output_path "./output/dpo_merge" \
    --device_map "cpu" \
    --torch_dtype "bfloat16"
```

### 5.2 ORPO

```bash
# 全参数模式训练 

python 3-强化/ORPO/orpo_train.py \
    --model_path ./output/sft_merge \
    --output_dir output/orpo \
    --dataset_path ./data/dirty_chinese_dpo.json
    
```

### 5.3 PPO

```bash
# 奖励模型训练 

python 3-强化/RM/rm_train.py \
    --model_path ./output/sft_merge \
    --output_dir output/rm_adapter \ 
    --dataset_path ./data/dirty_chinese_dpo.json \
    --max_datasize=500
```

```bash
### PPO训练

python 3-强化/RM/rm_train.py \
    --model_path ./output/sft_merge \
    ---rm_path ./model/rm_adapter \
    --output_dir output/ppo_adapter \
    --dataset_path ./data/dirty_chinese_dpo.json \
    --max_datasize 1000 
```

```bash
# 合并模型

python 2-微调/lora_merge.py \
    --base_model_path ./model/Qwen/Qwen3-0.6B \
    --lora_adapter_path "./output/ppo_adapter" \
    --merged_output_path "./output/ppo_merge" \
    --device_map "cpu" \
    --torch_dtype "bfloat16"
```

### 5.4 GRPO

```bash
# 基于lora结构，进行grpo训练  

python 3-强化/GRPO/grpo_train.py \
    --model_path ./output/sft_merge \
    --output_dir output/grpo_adapter \
    --dataset_path ./data/dirty_chinese_dpo.json  \
    --max_datasize 1000    
```

```bash
# 合并模型

python 2-微调/lora_merge.py \
    --base_model_path ./model/Qwen/Qwen3-0.6B \
    --lora_adapter_path "./output/grpo_adapter" \
    --merged_output_path "./output/grpo_merge" \
    --device_map "cpu" \
    --torch_dtype "bfloat16"
```

6.模型蒸馏

```bash
# 模型蒸馏

python 4-蒸馏/distill.py \
    --teacher_model_path ./model/Qwen/Qwen3-1.7B \
    --student_model_path ./model/Qwen/Qwen3-0.6B \
    --output_dir ./output/distilled_adapter \
    --dataset_path ./data/dirty_chinese_dpo.json \
    --alpha 0.7 \
    --temperature 1
```

```bash
# 合并模型

python 2-微调/lora_merge.py \
    --base_model_path ./model/Qwen/Qwen3-0.6B \
    --lora_adapter_path "./output/distilled_adapter" \
    --merged_output_path "./output/distilled_merge" \
    --device_map "cpu" \
    --torch_dtype "bfloat16"
```
