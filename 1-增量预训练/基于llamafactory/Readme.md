## 参考文档

    https://blog.csdn.net/hhhhhhhhhhwwwwwwwwww/article/details/148145089

## llamafactory

    不是可安装的软件包，必须保留
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory
    pip install -e ".[torch,metrics]"

## lora增量预训练

    sh lora_pretrain.sh

## 模型合并

    sh lora_merge.sh


## 全参数增量预训练
    sh full_pretrain.sh
