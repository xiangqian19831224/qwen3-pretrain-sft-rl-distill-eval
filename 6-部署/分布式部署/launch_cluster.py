#!/usr/bin/env python3
"""
VLLM 多机多卡分布式部署启动脚本
适用于 Qwen3-32B 模型
"""

import os
import argparse
import subprocess
import time
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="启动VLLM分布式集群")
    parser.add_argument("--config", type=str, required=True,
                        help="集群配置文件路径")
    parser.add_argument("--mode", type=str, choices=["master", "worker", "all"], 
                        default="all", help="启动模式")
    return parser.parse_args()

def load_config(config_path):
    """加载集群配置"""
    with open(config_path, 'r') as f:
        return json.load(f)

def start_master(config):
    """启动Master节点"""
    master_script = Path(__file__).parent / "vllm_distributed_master.sh"
    
    # 设置环境变量
    env = os.environ.copy()
    env.update({
        'MODEL_PATH': config['model_path'],
        'MASTER_IP': config['master']['ip'],
        'MASTER_PORT': str(config['master']['port']),
        'WORLD_SIZE': str(config['world_size']),
        'TENSOR_PARALLEL_SIZE': str(config['tensor_parallel_size']),
        'PIPELINE_PARALLEL_SIZE': str(config['pipeline_parallel_size'])
    })
    
    print(f"启动Master节点: {config['master']['ip']}")
    cmd = ["bash", str(master_script)]
    
    process = subprocess.Popen(cmd, env=env)
    return process

def start_worker(worker_config, config, rank):
    """启动Worker节点"""
    worker_script = Path(__file__).parent / "vllm_distributed_worker.sh"
    
    # 设置环境变量
    env = os.environ.copy()
    env.update({
        'MODEL_PATH': config['model_path'],
        'MASTER_IP': config['master']['ip'],
        'MASTER_PORT': str(config['master']['port']),
        'WORKER_RANK': str(rank),
        'TENSOR_PARALLEL_SIZE': str(config['tensor_parallel_size']),
        'WORLD_SIZE': str(config['world_size'])
    })
    
    print(f"启动Worker节点 {rank}: {worker_config['ip']}")
    cmd = f"ssh {worker_config['user']}@{worker_config['ip']} 'cd {worker_config['work_dir']} && bash {worker_script}'"
    
    process = subprocess.Popen(cmd, shell=True, env=env)
    return process

def check_cluster_health(master_ip):
    """检查集群健康状态"""
    import requests
    
    try:
        response = requests.get(f"http://{master_ip}:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    print(f"加载配置: {args.config}")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    
    processes = []
    
    try:
        if args.mode in ["master", "all"]:
            # 启动Master节点
            master_process = start_master(config)
            processes.append(master_process)
            time.sleep(10)  # 等待Master启动
            
        if args.mode in ["worker", "all"]:
            # 启动Worker节点
            for i, worker_config in enumerate(config['workers'], 1):
                worker_process = start_worker(worker_config, config, i)
                processes.append(worker_process)
                time.sleep(5)  # 间隔启动
        
        # 检查集群状态
        if args.mode == "all":
            print("等待集群启动...")
            time.sleep(30)
            
            if check_cluster_health(config['master']['ip']):
                print("✅ 集群启动成功!")
                print(f"API 端点: http://{config['master']['ip']}:8000")
                print(f"API Key: {config.get('api_key', 'Not set')}")
            else:
                print("❌ 集群启动失败")
        
        # 保持进程运行
        if args.mode != "all":
            for process in processes:
                process.wait()
        else:
            print("集群运行中... 按Ctrl+C停止")
            while True:
                time.sleep(10)
                if not check_cluster_health(config['master']['ip']):
                    print("⚠️  集群健康检查失败")
                    
    except KeyboardInterrupt:
        print("\n停止集群...")
        for process in processes:
            process.terminate()

if __name__ == "__main__":
    main()