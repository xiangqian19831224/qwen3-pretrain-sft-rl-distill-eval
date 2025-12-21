### 参考代码

    https://github.com/Zeyi-Lin/Qwen3-Medical-SFT

### 参考文档

    https://developer.aliyun.com/article/1663178

### 全参数微调&继续预训练

    | 维度             | 全参数微调（Full Fine-Tuning） | 继续预训练（Continual / Further Pretraining）  |
    | --------------- | ---------------------------- | ------------------------------------------- |
    | 是否更新全部参数   | ✅ 是                        | ✅ 是                                       |
    | 是否从已有权重开始 | ✅ 是                        | ✅ 是                                        |
    | 数据类型         | 有标注 / 指令 / SFT 数据        | 无标注语料                                    |
    | 训练目标         | 任务 / 对齐 / 指令能力           | 语言建模能力 / 领域知识                        |
    | Loss            | SFT loss / 对话 loss          | CLM / MLM loss                              |
    | 业界是否区分      | ✅ 明确区分                    | ✅ 明确区分                                  |

### 代码说明

    基于llamafactory
        支持lora增量预训练     这里保留lora增量预训练，只是说明有这种增量预训练，其实这种增量预训练理论上价值不大
        支持full增量预训练     增量预训练一般采用full模式
    基于transformers  
        支持full增量预训练     
        
    