"""
Run this script to load initial pretrained model, tokens and weights
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# First, download the pretrained model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


print(f"Model loaded: {model}")
print(f"Tokenizer: {tokenizer}")