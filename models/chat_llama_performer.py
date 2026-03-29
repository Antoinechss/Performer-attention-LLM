"""
Chat interface for TinyLlama model with Performer attention (FAVOR+)
Uses LlamaPerformerAttention instead of standard softmax attention
"""
import sys
import os
import importlib.util
import torch
from transformers import AutoTokenizer, TextStreamer

# Add local transformers to path so relative imports resolve correctly
_base = os.path.join(os.path.dirname(__file__), '..', 'transformers', 'src')
sys.path.insert(0, _base)

# Import parent packages first so relative imports in the performer file work
import transformers
import transformers.models
import transformers.models.llama

# Load performer file as a proper submodule of transformers.models.llama
_module_path = os.path.join(_base, 'transformers', 'models', 'llama', 'modeling_llama_performer.py')
_full_name = 'transformers.models.llama.modeling_llama_performer'
_spec = importlib.util.spec_from_file_location(_full_name, _module_path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_full_name] = _module
_spec.loader.exec_module(_module)

LlamaForCausalLM = _module.LlamaForCausalLM


def chat():
    print("Loading TinyLlama model with Performer attention...")
    model = LlamaForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("\n" + "="*50)
    print("TinyLlama Performer Chat (type 'quit' to exit)")
    print("="*50 + "\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        prompt = f"<|user|>\n{user_input}</s>\n<|assistant|>\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        print("Assistant: ", end="", flush=True)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer
            )

        print()


if __name__ == "__main__":
    chat()
