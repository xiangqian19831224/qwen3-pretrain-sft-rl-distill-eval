#!/usr/bin/env python3
"""
VLLM 分布式部署测试脚本
用于验证多机多卡部署是否正常工作
"""

import requests
import json
import time
import argparse
from typing import Dict, List, Any

class VLLMDistributedTester:
    def __init__(self, api_url: str = "http://192.168.1.100:8000", api_key: str = None):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def test_health(self) -> bool:
        """测试API健康状态"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"健康检查失败: {e}")
            return False
    
    def test_models(self) -> List[Dict]:
        """获取可用模型列表"""
        try:
            response = requests.get(f"{self.api_url}/v1/models", headers=self.headers, timeout=10)
            if response.status_code == 200:
                return response.json().get("data", [])
            else:
                print(f"获取模型列表失败: {response.status_code}")
                return []
        except Exception as e:
            print(f"获取模型列表异常: {e}")
            return []
    
    def test_chat_completion(self, message: str, model: str = None, max_tokens: int = 100) -> Dict:
        """测试聊天完成"""
        if not model:
            models = self.test_models()
            if models:
                model = models[0]["id"]
            else:
                model = "Qwen/Qwen2.5-32B-Instruct"
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/v1/chat/completions", 
                headers=self.headers, 
                json=payload, 
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"聊天请求失败: {response.status_code}")
                return {"error": response.text}
        except Exception as e:
            print(f"聊天请求异常: {e}")
            return {"error": str(e)}
    
    def test_concurrent_requests(self, num_requests: int = 10, message: str = "你好，请介绍一下你自己。") -> List[Dict]:
        """测试并发请求"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request(request_id):
            start_time = time.time()
            result = self.test_chat_completion(f"{message} (请求ID: {request_id})")
            end_time = time.time()
            
            results.put({
                "request_id": request_id,
                "response_time": end_time - start_time,
                "result": result,
                "success": "error" not in result
            })
        
        # 启动并发线程
        threads = []
        start_time = time.time()
        
        for i in range(num_requests):
            thread = threading.Thread(target=make_request, args=(i+1,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # 收集结果
        all_results = []
        while not results.empty():
            all_results.append(results.get())
        
        # 计算统计信息
        successful = [r for r in all_results if r["success"]]
        avg_response_time = sum(r["response_time"] for r in successful) / len(successful) if successful else 0
        
        print(f"并发测试结果:")
        print(f"  总请求数: {num_requests}")
        print(f"  成功请求数: {len(successful)}")
        print(f"  成功率: {len(successful)/num_requests*100:.1f}%")
        print(f"  平均响应时间: {avg_response_time:.2f}s")
        print(f"  总耗时: {total_time:.2f}s")
        print(f"  QPS: {num_requests/total_time:.2f}")
        
        return all_results
    
    def test_streaming(self, message: str = "请写一首关于AI的诗。") -> bool:
        """测试流式响应"""
        if not model:
            models = self.test_models()
            if models:
                model = models[0]["id"]
            else:
                model = "Qwen/Qwen2.5-32B-Instruct"
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": 200,
            "stream": True
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=30
            )
            
            if response.status_code == 200:
                print("流式响应测试:")
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]
                            if data != '[DONE]':
                                try:
                                    chunk = json.loads(data)
                                    if chunk['choices'][0]['delta'].get('content'):
                                        print(chunk['choices'][0]['delta']['content'], end='', flush=True)
                                except:
                                    pass
                print("\n✅ 流式响应测试成功")
                return True
            else:
                print(f"流式请求失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"流式请求异常: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="VLLM分布式部署测试")
    parser.add_argument("--url", default="http://192.168.1.100:8000", help="API URL")
    parser.add_argument("--api-key", help="API Key")
    parser.add_argument("--test", choices=["health", "models", "chat", "concurrent", "streaming", "all"], 
                        default="all", help="测试类型")
    parser.add_argument("--concurrent-requests", type=int, default=10, help="并发请求数")
    
    args = parser.parse_args()
    
    tester = VLLMDistributedTester(args.url, args.api_key)
    
    print(f"测试VLLM分布式部署: {args.url}")
    print("=" * 50)
    
    if args.test in ["health", "all"]:
        print("1. 健康检查...")
        if tester.test_health():
            print("✅ 健康检查通过")
        else:
            print("❌ 健康检查失败")
            return
    
    if args.test in ["models", "all"]:
        print("\n2. 获取模型列表...")
        models = tester.test_models()
        if models:
            print(f"✅ 发现 {len(models)} 个模型:")
            for model in models:
                print(f"  - {model['id']}")
        else:
            print("❌ 未发现可用模型")
    
    if args.test in ["chat", "all"]:
        print("\n3. 测试聊天完成...")
        result = tester.test_chat_completion("你好，请简单介绍一下Qwen3模型的特点。")
        if "error" not in result:
            print("✅ 聊天测试成功")
            print(f"回复: {result['choices'][0]['message']['content'][:100]}...")
        else:
            print(f"❌ 聊天测试失败: {result['error']}")
    
    if args.test in ["streaming", "all"]:
        print("\n4. 测试流式响应...")
        tester.test_streaming()
    
    if args.test in ["concurrent", "all"]:
        print("\n5. 测试并发请求...")
        tester.test_concurrent_requests(args.concurrent_requests)
    
    print("\n" + "=" * 50)
    print("测试完成!")

if __name__ == "__main__":
    main()