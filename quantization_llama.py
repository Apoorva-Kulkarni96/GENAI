import torch
import os
import gc
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig

hf_token = os.getenv("HF_TOKEN")
login(hf_token)

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"


messages = [
    {"role":"user", "content": "tell me a joke"}
]


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer.apply_chat_template(messages, return_tensors = "pt").to("cuda")
print(inputs)

model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map = "auto", quantization_config = quant_config)

memory = model.get_memory_footprint() / 1e6
print(f"Memory Footprint : {memory:,.1f} MB")
