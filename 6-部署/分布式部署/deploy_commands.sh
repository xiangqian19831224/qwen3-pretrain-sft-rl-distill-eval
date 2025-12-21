#!/bin/bash

# Qwen3-32B VLLM 分布式部署命令集合

# =============================================================================
# 部署前准备
# =============================================================================

echo "=== Qwen3-32B VLLM 分布式部署指南 ==="

# 检查环境
check_environment() {
    echo "检查部署环境..."
    
    # 检查Python版本
    python --version
    
    # 检查VLLM安装
    pip show vllm
    
    # 检查GPU
    nvidia-smi
    
    # 检查网络连接
    echo "检查节点间网络连接..."
}

# 安装依赖
install_dependencies() {
    echo "安装依赖包..."
    pip install vllm>=0.4.0
    pip install ray>=2.8.0
    pip install accelerate>=0.24.0
    pip install transformers>=4.35.0
    pip install fastapi>=0.104.0
    pip install uvicorn>=0.24.0
}

# =============================================================================
# 快速部署命令
# =============================================================================

# 方式1: 使用Python脚本部署
deploy_with_script() {
    echo "方式1: 使用Python脚本部署"
    echo "启动整个集群:"
    echo "python launch_cluster.py --config cluster_config.json --mode all"
    echo ""
    echo "仅启动Master:"
    echo "python launch_cluster.py --config cluster_config.json --mode master"
    echo ""
    echo "仅启动Worker:"
    echo "python launch_cluster.py --config cluster_config.json --mode worker"
}

# 方式2: 手动部署
deploy_manually() {
    echo "方式2: 手动部署"
    echo ""
    echo "Step 1: 在Master节点 (192.168.1.100) 执行:"
    echo "bash vllm_distributed_master.sh"
    echo ""
    echo "Step 2: 在Worker节点1 (192.168.1.101) 执行:"
    echo "WORKER_RANK=1 bash vllm_distributed_worker.sh"
    echo ""
    echo "Step 3: 在Worker节点2 (192.168.1.102) 执行:"
    echo "WORKER_RANK=2 bash vllm_distributed_worker.sh"
}

# 方式3: Docker部署
deploy_with_docker() {
    echo "方式3: Docker部署"
    echo ""
    echo "构建镜像:"
    echo "docker build -t vllm-qwen3:latest ."
    echo ""
    echo "启动Master:"
    echo "docker run -d --gpus all -p 8000:8000 -p 8265:8265 \\"
    echo "  -v \$(pwd)/model:/app/model \\"
    echo "  -e MASTER_IP=192.168.1.100 \\"
    echo "  -e WORLD_SIZE=4 \\"
    echo "  -e TENSOR_PARALLEL_SIZE=2 \\"
    echo "  --name vllm-master vllm-qwen3:latest"
    echo ""
    echo "启动Worker:"
    echo "docker run -d --gpus all \\"
    echo "  -v \$(pwd)/model:/app/model \\"
    echo "  -e MASTER_IP=192.168.1.100 \\"
    echo "  -e WORKER_RANK=1 \\"
    echo "  -e WORLD_SIZE=4 \\"
    echo "  --name vllm-worker1 vllm-qwen3:latest"
}

# 方式4: Kubernetes部署
deploy_with_k8s() {
    echo "方式4: Kubernetes部署"
    echo "kubectl apply -f k8s-deployment.yaml"
}

# =============================================================================
# 配置命令
# =============================================================================

# Ray集群配置
setup_ray_cluster() {
    echo "配置Ray集群:"
    echo "ray start --head --port=6379 --dashboard-host=0.0.0.0"
    echo "ray start --address='192.168.1.100:6379'"
}

# =============================================================================
# 监控命令
# =============================================================================

monitor_cluster() {
    echo "监控命令:"
    echo "1. 查看Ray Dashboard: http://192.168.1.100:8265"
    echo "2. 检查API健康状态: curl http://192.168.1.100:8000/health"
    echo "3. 查看模型信息: curl http://192.168.1.100:8000/v1/models"
    echo "4. 测试推理: curl -X POST http://192.168.1.100:8000/v1/chat/completions \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -d '{\"model\": \"Qwen/Qwen2.5-32B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
}

# =============================================================================
# 故障排除
# =============================================================================

troubleshooting() {
    echo "常见问题解决:"
    echo "1. 检查GPU内存: nvidia-smi"
    echo "2. 检查网络: ping 192.168.1.100"
    echo "3. 检查端口: netstat -tlnp | grep :8000"
    echo "4. 查看日志: tail -f ray_logs/worker*.out"
    echo "5. 重启Ray: ray stop && ray start --head"
}

# =============================================================================
# 性能调优
# =============================================================================

performance_tuning() {
    echo "性能调优参数:"
    echo "1. GPU内存利用率: --gpu-memory-utilization 0.95"
    echo "2. 最大序列数: --max-num-seqs 512"
    echo "3. 块大小: --block-size 32"
    echo "4. 启用前缀缓存: --enable-prefix-caching"
    echo "5. 使用FP8: --quantization fp8"
    echo "6. 启用投机解码: --speculative-model path/to/draft_model"
}

# =============================================================================
# 主函数
# =============================================================================

case "$1" in
    "check")
        check_environment
        ;;
    "install")
        install_dependencies
        ;;
    "script")
        deploy_with_script
        ;;
    "manual")
        deploy_manually
        ;;
    "docker")
        deploy_with_docker
        ;;
    "k8s")
        deploy_with_k8s
        ;;
    "ray")
        setup_ray_cluster
        ;;
    "monitor")
        monitor_cluster
        ;;
    "troubleshoot")
        troubleshooting
        ;;
    "tuning")
        performance_tuning
        ;;
    *)
        echo "用法: $0 {check|install|script|manual|docker|k8s|ray|monitor|troubleshoot|tuning}"
        echo ""
        echo "示例:"
        echo "  $0 check      # 检查环境"
        echo "  $0 install    # 安装依赖"
        echo "  $0 script     # Python脚本部署"
        echo "  $0 manual     # 手动部署"
        echo "  $0 docker     # Docker部署"
        echo "  $0 monitor    # 监控命令"
        echo ""
        deploy_with_script
        ;;
esac