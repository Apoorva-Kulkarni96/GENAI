import torch
import os
import gc
from huggingface_hub import login
#from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
# Instead of AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaConfig, BitsAndBytesConfig
hf_token = os.getenv("HF_TOKEN")
login(hf_token)


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
config = LlamaConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", 
    config=config,
    quantization_config=quant_config,
    device_map="auto"
)
memory = model.get_memory_footprint() / 1e6
print(f"Memory Footprint : {memory:,.1f} MB")

