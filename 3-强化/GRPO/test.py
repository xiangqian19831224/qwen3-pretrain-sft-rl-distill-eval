import torch
from transformers import AutoModelForCausalLM

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("../../output/sft_merge", trust_remote_code=True)
print("Tokenizer loaded")

