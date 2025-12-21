#!/bin/bash

# Qwen3-32B VLLM 分布式部署快速启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 默认配置
DEFAULT_MODEL_PATH="../../model/sft_merge"
DEFAULT_MASTER_IP="192.168.1.100"
DEFAULT_MASTER_PORT="29500"
DEFAULT_WORLD_SIZE="4"
DEFAULT_TENSOR_PARALLEL_SIZE="2"
DEFAULT_PIPELINE_PARALLEL_SIZE="2"

# 显示帮助信息
show_help() {
    cat << EOF
Qwen3-32B VLLM 分布式部署快速启动脚本

用法:
    $0 [选项] <模式>

模式:
    master         启动Master节点
    worker         启动Worker节点  
    all            启动完整集群
    test           测试部署
    stop           停止所有服务
    status         查看集群状态

选项:
    --model-path PATH          模型路径 (默认: $DEFAULT_MODEL_PATH)
    --master-ip IP             Master节点IP (默认: $DEFAULT_MASTER_IP)
    --master-port PORT         Master节点端口 (默认: $DEFAULT_MASTER_PORT)
    --world-size SIZE          总GPU数量 (默认: $DEFAULT_WORLD_SIZE)
    --tensor-parallel SIZE     张量并行大小 (默认: $DEFAULT_TENSOR_PARALLEL_SIZE)
    --pipeline-parallel SIZE   流水并行大小 (默认: $DEFAULT_PIPELINE_PARALLEL_SIZE)
    --worker-rank RANK         Worker节点rank (仅worker模式)
    --gpu-memory UTIL          GPU内存利用率 (默认: 0.85)
    --max-seqs NUM             最大序列数 (默认: 256)
    --max-length LEN           最大序列长度 (默认: 8192)

示例:
    # 启动Master节点
    $0 master --master-ip 192.168.1.100 --model-path ./model

    # 启动Worker节点1
    $0 worker --master-ip 192.168.1.100 --worker-rank 1

    # 启动完整集群 (单机测试)
    $0 all --world-size 2 --tensor-parallel-size 2

    # 测试部署
    $0 test --master-ip 192.168.1.100

EOF
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model-path)
                MODEL_PATH="$2"
                shift 2
                ;;
            --master-ip)
                MASTER_IP="$2"
                shift 2
                ;;
            --master-port)
                MASTER_PORT="$2"
                shift 2
                ;;
            --world-size)
                WORLD_SIZE="$2"
                shift 2
                ;;
            --tensor-parallel)
                TENSOR_PARALLEL_SIZE="$2"
                shift 2
                ;;
            --pipeline-parallel)
                PIPELINE_PARALLEL_SIZE="$2"
                shift 2
                ;;
            --worker-rank)
                WORKER_RANK="$2"
                shift 2
                ;;
            --gpu-memory)
                GPU_MEMORY_UTILIZATION="$2"
                shift 2
                ;;
            --max-seqs)
                MAX_NUM_SEQS="$2"
                shift 2
                ;;
            --max-length)
                MAX_MODEL_LEN="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                MODE="$1"
                shift
                ;;
        esac
    done
}

# 设置默认值
set_defaults() {
    MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
    MASTER_IP="${MASTER_IP:-$DEFAULT_MASTER_IP}"
    MASTER_PORT="${MASTER_PORT:-$DEFAULT_MASTER_PORT}"
    WORLD_SIZE="${WORLD_SIZE:-$DEFAULT_WORLD_SIZE}"
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$DEFAULT_TENSOR_PARALLEL_SIZE}"
    PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-$DEFAULT_PIPELINE_PARALLEL_SIZE}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
}

# 检查环境
check_environment() {
    log_info "检查部署环境..."
    
    # 检查Python
    if ! command -v python &> /dev/null; then
        log_error "Python未安装"
        exit 1
    fi
    
    # 检查VLLM
    if ! python -c "import vllm" &> /dev/null; then
        log_error "VLLM未安装，请运行: pip install vllm"
        exit 1
    fi
    
    # 检查GPU
    if ! command -v nvidia-smi &> /dev/null; then
        log_warning "nvidia-smi不可用，请确保NVIDIA驱动已安装"
    else
        log_info "GPU信息:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    fi
    
    # 检查模型路径
    if [[ ! -d "$MODEL_PATH" ]]; then
        log_error "模型路径不存在: $MODEL_PATH"
        exit 1
    fi
    
    log_success "环境检查通过"
}

# 启动Master节点
start_master() {
    log_info "启动Master节点..."
    log_info "Master IP: $MASTER_IP"
    log_info "模型路径: $MODEL_PATH"
    log_info "配置: world_size=$WORLD_SIZE, tensor_parallel=$TENSOR_PARALLEL_SIZE, pipeline_parallel=$PIPELINE_PARALLEL_SIZE"
    
    # 设置环境变量
    export VLLM_USE_RAY=1
    export RAY_BACKEND_LOG_LEVEL=info
    export RAY_LOG_DIR=./ray_logs
    export RAY_ADDRESS="${MASTER_IP}:6379"
    
    # 创建日志目录
    mkdir -p ray_logs
    
    # 启动Ray head节点
    log_info "启动Ray head节点..."
    ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 || true
    
    # 启动VLLM Master
    log_info "启动VLLM Master服务..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" \
        --world-size "$WORLD_SIZE" \
        --rank 0 \
        --distributed-executor-backend ray \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-model-len "$MAX_MODEL_LEN" \
        --dtype float16 \
        --trust-remote-code \
        --enable-prefix-caching \
        --block-size 16 \
        --max-num-batched-tokens 8192 > ray_logs/vllm_master.out 2>&1 &
    
    MASTER_PID=$!
    echo $MASTER_PID > ray_logs/master.pid
    
    log_success "Master节点已启动 (PID: $MASTER_PID)"
    log_info "API端点: http://$MASTER_IP:8000"
    log_info "Ray Dashboard: http://$MASTER_IP:8265"
    
    # 等待服务启动
    sleep 10
    
    # 健康检查
    if curl -s http://localhost:8000/health > /dev/null; then
        log_success "Master服务健康检查通过"
    else
        log_warning "Master服务健康检查失败，请查看日志"
    fi
}

# 启动Worker节点
start_worker() {
    if [[ -z "$WORKER_RANK" ]]; then
        log_error "Worker模式需要指定 --worker-rank"
        exit 1
    fi
    
    log_info "启动Worker节点 $WORKER_RANK..."
    log_info "Master IP: $MASTER_IP"
    log_info "Worker Rank: $WORKER_RANK"
    
    # 设置环境变量
    export VLLM_USE_RAY=1
    export RAY_BACKEND_LOG_LEVEL=info
    export RAY_LOG_DIR=./ray_logs
    export RAY_ADDRESS="${MASTER_IP}:6379"
    
    # 创建日志目录
    mkdir -p ray_logs
    
    # 启动Ray worker节点
    log_info "连接到Ray集群..."
    ray start --address="$MASTER_IP:6379" || true
    
    # 启动VLLM Worker
    log_info "启动VLLM Worker进程..."
    python -m vllm.worker.worker \
        --model "$MODEL_PATH" \
        --world-size "$WORLD_SIZE" \
        --rank "$WORKER_RANK" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --distributed-executor-backend ray \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --dtype float16 \
        --trust-remote-code \
        --block-size 16 \
        --max-num-batched-tokens 8192 > ray_logs/vllm_worker_$WORKER_RANK.out 2>&1 &
    
    WORKER_PID=$!
    echo $WORKER_PID > ray_logs/worker_$WORKER_RANK.pid
    
    log_success "Worker节点 $WORKER_RANK 已启动 (PID: $WORKER_PID)"
}

# 测试部署
test_deployment() {
    log_info "测试部署..."
    
    # 健康检查
    log_info "检查API健康状态..."
    if curl -s http://$MASTER_IP:8000/health | grep -q "OK"; then
        log_success "API健康检查通过"
    else
        log_error "API健康检查失败"
        return 1
    fi
    
    # 模型列表
    log_info "获取可用模型..."
    curl -s http://$MASTER_IP:8000/v1/models | python -m json.tool || true
    
    # 聊天测试
    log_info "测试聊天功能..."
    curl -s -X POST http://$MASTER_IP:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "Qwen/Qwen2.5-32B-Instruct",
            "messages": [{"role": "user", "content": "你好，请简单介绍一下自己。"}],
            "max_tokens": 50,
            "temperature": 0.7
        }' | python -m json.tool || true
    
    log_success "测试完成"
}

# 停止服务
stop_services() {
    log_info "停止所有服务..."
    
    # 停止VLLM进程
    if [[ -f ray_logs/master.pid ]]; then
        MASTER_PID=$(cat ray_logs/master.pid)
        if kill -0 $MASTER_PID 2>/dev/null; then
            kill $MASTER_PID
            log_info "已停止Master进程 (PID: $MASTER_PID)"
        fi
        rm -f ray_logs/master.pid
    fi
    
    # 停止Worker进程
    for pid_file in ray_logs/worker_*.pid; do
        if [[ -f "$pid_file" ]]; then
            WORKER_PID=$(cat "$pid_file")
            if kill -0 $WORKER_PID 2>/dev/null; then
                kill $WORKER_PID
                log_info "已停止Worker进程 (PID: $WORKER_PID)"
            fi
            rm -f "$pid_file"
        fi
    done
    
    # 停止Ray
    ray stop || true
    
    log_success "所有服务已停止"
}

# 查看状态
show_status() {
    log_info "集群状态:"
    
    # 检查进程
    if [[ -f ray_logs/master.pid ]]; then
        MASTER_PID=$(cat ray_logs/master.pid)
        if kill -0 $MASTER_PID 2>/dev/null; then
            log_success "Master节点运行中 (PID: $MASTER_PID)"
        else
            log_error "Master节点未运行"
        fi
    else
        log_warning "Master节点未启动"
    fi
    
    # 检查Worker
    for pid_file in ray_logs/worker_*.pid; do
        if [[ -f "$pid_file" ]]; then
            WORKER_PID=$(cat "$pid_file")
            WORKER_RANK=$(basename "$pid_file" | sed 's/worker_\(.*\)\.pid/\1/')
            if kill -0 $WORKER_PID 2>/dev/null; then
                log_success "Worker节点 $WORKER_RANK 运行中 (PID: $WORKER_PID)"
            else
                log_error "Worker节点 $WORKER_RANK 未运行"
            fi
        fi
    done
    
    # API状态
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        log_success "API服务正常"
    else
        log_warning "API服务不可用"
    fi
    
    # Ray状态
    if ray status 2>/dev/null | grep -q "cluster"; then
        log_success "Ray集群正常"
        ray status
    else
        log_warning "Ray集群未运行"
    fi
}

# 主函数
main() {
    # 解析参数
    parse_args "$@"
    
    # 设置默认值
    set_defaults
    
    # 显示配置信息
    log_info "Qwen3-32B VLLM 分布式部署"
    log_info "模式: ${MODE:-未指定}"
    log_info "配置:"
    log_info "  模型路径: $MODEL_PATH"
    log_info "  Master IP: $MASTER_IP"
    log_info "  World Size: $WORLD_SIZE"
    log_info "  Tensor Parallel: $TENSOR_PARALLEL_SIZE"
    log_info "  Pipeline Parallel: $PIPELINE_PARALLEL_SIZE"
    
    case "${MODE:-help}" in
        master)
            check_environment
            start_master
            ;;
        worker)
            check_environment
            start_worker
            ;;
        all)
            check_environment
            start_master
            log_info "启动完成，使用 '$0 test' 测试部署"
            ;;
        test)
            test_deployment
            ;;
        stop)
            stop_services
            ;;
        status)
            show_status
            ;;
        *)
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"