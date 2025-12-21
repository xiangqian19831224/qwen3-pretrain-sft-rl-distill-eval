# 使用 Python API 方式
from vllm import LLM, SamplingParams

llm = LLM(model="../../model/sft_merge", tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(["你的问题"], sampling_params)
