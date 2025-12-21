import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from dataclasses import dataclass, field
from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    """
    LoRA合并脚本的配置参数
    """
    base_model_path: str = field(metadata={"help": "基础模型的路径"})
    lora_adapter_path: str = field(metadata={"help": "LoRA适配器的路径"})
    merged_output_path: str = field(metadata={"help": "合并后模型的保存路径"})
    device_map: str = field(default="cpu", metadata={"help": "设备映射，默认为CPU以避免显存不足"})
    torch_dtype: str = field(default="bfloat16", metadata={"help": "模型数据类型"})


def get_torch_dtype(dtype_str: str):
    """将字符串转换为torch数据类型"""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def main():
    """
    该脚本用于将训练好的LoRA适配器合并到基础模型中，
    并将其保存为一个独立的模型。
    """
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    print("🚀 开始 LoRA 适配器合并...")
    print(f"📁 基础模型路径: {args.base_model_path}")
    print(f"📁 LoRA适配器路径: {args.lora_adapter_path}")
    print(f"📁 合并输出路径: {args.merged_output_path}")
    print(f"🔧 设备映射: {args.device_map}")
    print(f"🔧 数据类型: {args.torch_dtype}")

    # 检查路径是否存在
    if not os.path.exists(args.base_model_path):
        raise FileNotFoundError(f"基础模型未在 {args.base_model_path} 找到")
    
    if not os.path.exists(args.lora_adapter_path):
        raise FileNotFoundError(f"LoRA适配器未在 {args.lora_adapter_path} 找到")

    # 1. 加载基础模型
    print(f"正在从以下路径加载基础模型: {args.base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        dtype=get_torch_dtype(args.torch_dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )

    # 2. 加载 Tokenizer
    print("正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)

    # 3. 加载 LoRA 适配器
    print(f"正在从以下路径加载 LoRA 适配器: {args.lora_adapter_path}")
    model_to_merge = PeftModel.from_pretrained(base_model, args.lora_adapter_path)

    # 4. 调用 merge_and_unload 将适配器权重合并到基础模型
    print("正在将适配器合并到基础模型中...")
    merged_model = model_to_merge.merge_and_unload()
    print("✅ 合并完成。")

    # 5. 保存合并后的模型和 Tokenizer
    print(f"正在将合并后的模型保存到: {args.merged_output_path}")
    os.makedirs(args.merged_output_path, exist_ok=True)
    merged_model.save_pretrained(args.merged_output_path)
    tokenizer.save_pretrained(args.merged_output_path)

    print("🎉 合并后的模型已成功保存！")


if __name__ == "__main__":
    main()