#!/usr/bin/env python3
"""
Qwen3-0.6B 模型加载与测试程序
支持本地模型加载和基本推理测试
"""

import torch
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List

class Qwen3Loader:
    """Qwen3模型加载器"""
    
    def __init__(self, model_path: str):
        """
        初始化Qwen3模型加载器
        
        Args:
            model_path: 本地模型路径
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """加载模型和分词器"""
        print(f"正在从 {self.model_path} 加载模型...")
        print(f"使用设备: {self.device}")
        
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False
            )
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=True,
                use_cache=True
            )
            
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)
            
            print("✅ 模型和分词器加载成功!")
            self.print_model_info()
            
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            raise
    
    def print_model_info(self):
        """打印模型信息"""
        if self.model is None:
            return
            
        print("\n" + "="*50)
        print("📊 模型信息:")
        print(f"模型类型: {type(self.model).__name__}")
        print(f"参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"可训练参数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"词汇表大小: {self.tokenizer.vocab_size:,}")
        print(f"最大序列长度: {self.tokenizer.model_max_length}")
        print("="*50)
    
    def generate_text(self, prompt: str, max_length: int = 512, temperature: float = 0.7, 
                     top_p: float = 0.9, do_sample: bool = True) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top_p采样参数
            do_sample: 是否采样
            
        Returns:
            生成的文本
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("请先加载模型!")
        
        # 对输入进行编码
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # 生成文本
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        generation_time = time.time() - start_time
        
        # 解码输出
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 只返回生成的部分（去掉输入提示）
        if generated_text.startswith(prompt):
            result = generated_text[len(prompt):].strip()
        else:
            result = generated_text.strip()
        
        print(f"⏱️ 生成耗时: {generation_time:.2f}秒")
        
        return result
    
    def chat(self, user_input: str, system_prompt: str = "你是一个有帮助的AI助手。") -> str:
        """
        聊天对话
        
        Args:
            user_input: 用户输入
            system_prompt: 系统提示
            
        Returns:
            AI回复
        """
        # 构建对话格式
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        
        return self.generate_text(prompt)


def run_tests():
    """运行模型测试"""
    print("🚀 开始Qwen3-0.6B模型测试")
    
    # 模型路径
    model_path = "output/sft_merge"
    
    try:
        # 初始化加载器
        loader = Qwen3Loader(model_path)
        
        # 加载模型
        loader.load_model()
        
        # 测试用例
        test_cases = [
            {
                "name": "基础问答测试",
                "input": "你好，请介绍一下你自己。",
                "type": "chat"
            },
            {
                "name": "知识问答测试", 
                "input": "什么是机器学习？",
                "type": "chat"
            },
            {
                "name": "代码生成测试",
                "input": "请用Python写一个计算斐波那契数列的函数。",
                "type": "chat"
            },
            {
                "name": "创意写作测试",
                "input": "请写一首关于春天的短诗。",
                "type": "chat"
            },
            {
                "name": "文本补全测试",
                "input": "人工智能的未来发展趋势是",
                "type": "generate",
                "params": {"max_length": 200, "temperature": 0.8}
            }
        ]
        
        # 运行测试
        print("\n" + "🧪"*20)
        print("开始执行测试用例...")
        print("🧪"*20)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"测试 {i}/{len(test_cases)}: {test_case['name']}")
            print(f"输入: {test_case['input']}")
            print("-"*60)
            
            if test_case['type'] == 'chat':
                response = loader.chat(test_case['input'])
            else:  # generate
                params = test_case.get('params', {})
                response = loader.generate_text(test_case['input'], **params)
            
            print(f"输出: {response}")
            print(f"{'='*60}")
        
        # 性能测试
        print("\n" + "⚡"*20)
        print("性能测试...")
        print("⚡"*20)
        
        test_prompt = "请简单介绍一下深度学习的基本概念。"
        
        # 测试不同参数
        configs = [
            {"temperature": 0.1, "name": "低温度采样"},
            {"temperature": 0.7, "name": "中等温度采样"},
            {"temperature": 1.0, "name": "高温度采样"}
        ]
        
        for config in configs:
            print(f"\n测试配置: {config['name']} (temperature={config['temperature']})")
            start_time = time.time()
            response = loader.chat(test_prompt)
            end_time = time.time()
            print(f"回复: {response}")
            print(f"总耗时: {end_time - start_time:.2f}秒")
        
        print("\n🎉 所有测试完成!")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def interactive_mode():
    """交互式对话模式"""
    print("💬 进入交互式对话模式")
    print("输入 'quit' 或 'exit' 退出")
    
    model_path = "model/sft"
    loader = Qwen3Loader(model_path)
    loader.load_model()
    
    while True:
        try:
            user_input = input("\n用户: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 再见!")
                break
            
            if not user_input:
                continue
            
            print("AI: ", end="", flush=True)
            response = loader.chat(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 生成回复时出错: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        run_tests()