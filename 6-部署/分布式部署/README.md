# Qwen3-32B VLLM 多机多卡分布式部署

本目录包含了基于VLLM框架部署Qwen3-32B模型的多机多卡分布式方案。

## 🏗️ 架构概述

- **模型**: Qwen3-32B (32B参数)
- **框架**: VLLM + Ray
- **并行策略**: Tensor Parallel + Pipeline Parallel
- **部署方式**: 多机多卡分布式推理

## 📋 环境要求

### 硬件要求
- **最小配置**: 2台服务器，每台至少2张GPU (建议A100 40GB或H100 80GB)
- **推荐配置**: 4台服务器，每台2-4张GPU
- **网络**: 万兆以太网或InfiniBand
- **存储**: 共享存储或每个节点本地存储模型文件

### 软件要求
- CUDA >= 12.1
- Python >= 3.9
- Docker (可选)
- Kubernetes (可选)

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
./deploy_commands.sh install

# 检查环境
./deploy_commands.sh check
```

### 2. 配置集群

编辑 `cluster_config.json` 文件，配置你的集群信息：

```json
{
    "model_path": "../../model/sft_merge",
    "world_size": 4,
    "tensor_parallel_size": 2,
    "pipeline_parallel_size": 2,
    "master": {
        "ip": "192.168.1.100",
        "port": 29500,
        "user": "username",
        "work_dir": "/path/to/deployment"
    },
    "workers": [
        {
            "ip": "192.168.1.101",
            "user": "username", 
            "work_dir": "/path/to/deployment"
        },
        {
            "ip": "192.168.1.102",
            "user": "username",
            "work_dir": "/path/to/deployment"
        }
    ]
}
```

### 3. 部署方式

#### 方式1: Python脚本部署 (推荐)

```bash
# 启动整个集群
python launch_cluster.py --config cluster_config.json --mode all

# 仅启动Master节点
python launch_cluster.py --config cluster_config.json --mode master

# 仅启动Worker节点
python launch_cluster.py --config cluster_config.json --mode worker
```

#### 方式2: 手动部署

```bash
# Master节点
bash vllm_distributed_master.sh

# Worker节点1
WORKER_RANK=1 bash vllm_distributed_worker.sh

# Worker节点2  
WORKER_RANK=2 bash vllm_distributed_worker.sh
```

#### 方式3: Docker部署

```bash
# 构建镜像
docker build -t vllm-qwen3:latest .

# 启动Master
docker run -d --gpus all -p 8000:8000 -p 8265:8265 \
  -v $(pwd)/model:/app/model \
  -e MASTER_IP=192.168.1.100 \
  --name vllm-master vllm-qwen3:latest

# 启动Worker
docker run -d --gpus all \
  -v $(pwd)/model:/app/model \
  -e MASTER_IP=192.168.1.100 \
  -e WORKER_RANK=1 \
  --name vllm-worker1 vllm-qwen3:latest
```

#### 方式4: Kubernetes部署

```bash
kubectl apply -f k8s-deployment.yaml
```

## 📊 部署命令速查

```bash
# 查看所有部署选项
./deploy_commands.sh

# 环境检查
./deploy_commands.sh check

# 监控集群
./deploy_commands.sh monitor

# 故障排除
./deploy_commands.sh troubleshoot

# 性能调优
./deploy_commands.sh tuning
```

## 🧪 测试验证

部署完成后，运行测试脚本验证功能：

```bash
# 完整测试
python test_distributed.py --url http://192.168.1.100:8000

# 单独测试
python test_distributed.py --test health
python test_distributed.py --test chat
python test_distributed.py --test concurrent --concurrent-requests 20
python test_distributed.py --test streaming
```

## 📈 性能调优

### 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--gpu-memory-utilization` | 0.85-0.95 | GPU内存利用率 |
| `--max-num-seqs` | 256-512 | 最大并发序列数 |
| `--max-model-len` | 8192-16384 | 最大序列长度 |
| `--block-size` | 16-32 | 块大小 |
| `--tensor-parallel-size` | 2-4 | 张量并行度 |
| `--pipeline-parallel-size` | 2-4 | 流水并行度 |

### 优化技巧

1. **启用前缀缓存**: `--enable-prefix-caching`
2. **使用量化**: `--quantization fp8` 或 `--quantization int4`
3. **投机解码**: `--speculative-model path/to/draft_model`
4. **批处理优化**: 调整 `--max-num-batched-tokens`

## 🔍 监控指标

### Ray Dashboard
- URL: `http://master-ip:8265`
- 监控集群状态、资源使用、任务执行情况

### API监控
```bash
# 健康检查
curl http://master-ip:8000/health

# 模型信息
curl http://master-ip:8000/v1/models

# 性能指标
curl http://master-ip:8000/metrics
```

## 🛠️ 故障排除

### 常见问题

1. **网络连接失败**
   ```bash
   ping worker-ip
   telnet worker-ip 29500
   ```

2. **GPU内存不足**
   ```bash
   nvidia-smi
   # 减少 --max-num-seqs 或 --gpu-memory-utilization
   ```

3. **模型加载失败**
   ```bash
   # 检查模型路径
   ls -la ../../model/sft_merge
   # 确保所有节点都有模型文件访问权限
   ```

4. **Ray集群异常**
   ```bash
   ray stop
   ray start --head --port=6379
   ```

### 日志查看

```bash
# VLLM日志
tail -f ray_logs/worker*.out

# Ray日志
tail -f ray_logs/raylet.out

# 系统日志
journalctl -u vllm -f
```

## 📚 API使用示例

### OpenAI兼容API

```bash
curl -X POST http://master-ip:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-32B-Instruct",
    "messages": [{"role": "user", "content": "你好！"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Python客户端

```python
import openai

client = openai.OpenAI(
    base_url="http://master-ip:8000/v1",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-32B-Instruct",
    messages=[{"role": "user", "content": "你好！"}],
    max_tokens=100
)

print(response.choices[0].message.content)
```

## 🔐 安全配置

### API认证

```bash
# 设置API密钥
export VLLM_API_KEY="your-secret-key"

# 在请求中使用
curl -H "Authorization: Bearer your-secret-key" \
     http://master-ip:8000/v1/models
```

### 网络安全

- 使用防火墙限制访问端口
- 配置SSL/TLS加密
- 设置访问控制列表(ACL)

## 📞 技术支持

如遇到问题，请：

1. 查看本文档的故障排除部分
2. 检查日志文件获取详细错误信息
3. 运行 `./deploy_commands.sh troubleshoot`
4. 提交Issue并附上环境信息和错误日志

---

## 📄 文件说明

- `cluster_config.json` - 集群配置文件
- `launch_cluster.py` - 集群启动脚本
- `vllm_distributed_master.sh` - Master节点启动脚本
- `vllm_distributed_worker.sh` - Worker节点启动脚本
- `deploy_commands.sh` - 部署命令集合
- `test_distributed.py` - 分布式测试脚本
- `Dockerfile` - Docker镜像构建文件
- `k8s-deployment.yaml` - Kubernetes部署配置
- `requirements.txt` - Python依赖列表