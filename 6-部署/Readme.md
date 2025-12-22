## 硬件与算例选型

    | 模型规模 |         推荐硬件           | 说明                                              |
    | ------- | ------------------------ | ------------------------------------------------- |
    | 0B~1B   | GPU单机/CPU               | 可用单卡推理，成本低                                 |
    | 3B~7B   | 单机多卡或GPU集群           | 可以用数据并行或TensorParallel                      |
    | 13B~70B | 多机多卡                   | 需要分布式推理（TensorParallel + PipelineParallel）  |
    | 100B+   | 专用集群（DGX A100/H100）   | 高并发和低延迟需要结合并行策略                         |

## 部署架构

    2.1 单机服务模式
        场景：小规模用户、开发测试。    
        特点：直接在GPU上加载模型，提供HTTP/gRPC接口。    
        技术栈：    
            FastAPI / Flask + Uvicorn    
            TorchServe / vLLM / text-generation-inference (TGI)    
        优势：简单快速，成本低。    
        缺点：难以横向扩展，单机GPU成为瓶颈。
    
    2.2 分布式服务模式    
        场景：高并发、超大模型。    
        特点：    
            推理并行：TensorParallel + PipelineParallel + Model Sharding    
            请求调度：前置负载均衡 + 请求排队/批处理    
            异步推理：队列 + Worker 模型实例
        技术栈：    
            NVIDIA Triton Inference Server    
            Hugging Face Accelerate + DeepSpeed Inference    
            vLLM + Ray 或 Celery 进行任务分发    
        优势：可以支撑上万QPS，支持大模型。    
        缺点：系统复杂，调试和运维成本高。

    2.3 Serverless/云原生部署    
        场景：多租户、动态伸缩。    
        特点：    
            Kubernetes + GPU Operator + Pod 弹性伸缩    
            利用云GPU（AWS, GCP, Azure, Lambda GPU）    
        优势：按需弹性伸缩，管理成本低。    
        缺点：延迟不可控，对大模型分布式优化复杂。

## 部署说明  采用codebuddy生成

    单机部署     验证通过
    分布式部署   需要验证？？？
    
