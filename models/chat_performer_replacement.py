"""
Performer Chat Interface - Option 1: On-the-fly attention replacement

This approach:
1. Loads standard TinyLlama model
2. Replaces each standard attention layer with LlamaPerformerAttention
3. Copies weights from standard attention to Performer attention
4. Uses the modified model for generation

This is ideal for:
- Comparing Performer vs Standard attention on same weights
- Computing metrics (perplexity, latency, memory, etc.)
- Studying approximation impact without modifying training pipeline
"""

import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# Add performer directory to path
performer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../performer'))
sys.path.insert(0, performer_path)

from performer_attention import PerformerAttentionCore


class LlamaPerformerAttention(torch.nn.Module):
    """LLaMA attention wrapper using Performer causal kernel attention"""

    def __init__(self, config, layer_idx: int, dtype=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype

        # Grouped query attention parameters
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # Q, K, V, O projections (will be initialized with standard attention weights)
        self.q_proj = torch.nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias, dtype=dtype
        )
        self.k_proj = torch.nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias, dtype=dtype
        )
        self.v_proj = torch.nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias, dtype=dtype
        )
        self.o_proj = torch.nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias, dtype=dtype
        )

        # Performer attention core engine with FAVOR+
        self.performer_att = PerformerAttentionCore(
            head_dim=self.head_dim,
            num_features=256
        )
        
        # Convert performer attention to correct dtype
        if dtype is not None:
            self.performer_att = self.performer_att.to(dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple = None,
        attention_mask: torch.Tensor = None,
        past_key_values: object = None,
        **kwargs,
    ) -> tuple:
        """
        Forward pass using Performer attention instead of standard softmax attention.
        
        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
            position_embeddings: (cos, sin) from RoPE
            attention_mask: causal mask
            past_key_valuesatt: KV cache
            
        Returns:
            (attn_output, attn_weights) - weights are None for Performer
        """
        batch_size, seq_length, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)  # (B, H, N, D)

        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # (B, num_kv_heads, N, D)
        
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # (B, num_kv_heads, N, D)
        
        # Apply RoPE position embeddings
        cos, sin = position_embeddings
        query_states, key_states = self._apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # Handle KV cache if present
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        # Expand K/V to match Q heads (for Grouped Query Attention)
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Performer attention computation
        attn_output = self.performer_att(query_states, key_states, value_states)
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None

    @staticmethod
    def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        """Apply rotary position embeddings to Q and K"""
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (LlamaPerformerAttention._rotate_half(q) * sin)
        k_embed = (k * cos) + (LlamaPerformerAttention._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


def replace_attention_with_performer(model):
    """
    Replace all standard attention layers with Performer attention.
    Copies weights from standard attention to Performer attention.
    
    Args:
        model: LlamaForCausalLM model
        
    Returns:
        Modified model with Performer attention
    """
    print(f"\n{'='*60}")
    print("Replacing attention layers with Performer attention...")
    print(f"{'='*60}\n")
    
    # Detect dtype from model weights
    model_dtype = next(model.parameters()).dtype
    
    for layer_idx, layer in enumerate(model.model.layers):
        # Get the standard attention module to extract weights
        standard_attn = layer.self_attn
        
        # Create new Performer attention with correct dtype
        performer_attn = LlamaPerformerAttention(
            model.config, layer_idx, dtype=model_dtype
        )
        
        # Move to same device and dtype as model
        performer_attn = performer_attn.to(dtype=model_dtype, device=model.device)
        
        # Copy weights from standard attention to Performer attention
        # These projections have identical dimensions, so weight transfer is direct
        with torch.no_grad():
            performer_attn.q_proj.weight.copy_(
                standard_attn.q_proj.weight
            )
            performer_attn.k_proj.weight.copy_(
                standard_attn.k_proj.weight
            )
            performer_attn.v_proj.weight.copy_(
                standard_attn.v_proj.weight
            )
            performer_attn.o_proj.weight.copy_(
                standard_attn.o_proj.weight
            )
            
            # Copy biases if present
            if standard_attn.q_proj.bias is not None:
                performer_attn.q_proj.bias.copy_(
                    standard_attn.q_proj.bias
                )
                performer_attn.k_proj.bias.copy_(
                    standard_attn.k_proj.bias
                )
                performer_attn.v_proj.bias.copy_(
                    standard_attn.v_proj.bias
                )
                performer_attn.o_proj.bias.copy_(
                    standard_attn.o_proj.bias
                )
        
        # Replace the attention module
        layer.self_attn = performer_attn
        
        if layer_idx == 0:
            print(f"✓ Layer {layer_idx}: Replaced attention "
                  f"(dtype={model_dtype}, Performer num_features=256)")
        elif (layer_idx + 1) % 5 == 0:
            print(f"✓ Layers {layer_idx - 4} to {layer_idx}: "
                  f"Replaced attention")
    
    print(f"\n✓ Successfully replaced all {len(model.model.layers)} "
          f"attention layers")
    print(f"{'='*60}\n")
    
    return model


def chat():
    """Interactive chat interface for Performer-based TinyLlama"""
    print("\n" + "="*60)
    print("Loading TinyLlama model...")
    print("="*60)
    
    # Load standard model first
    # Use device_map=None to avoid disk offloading issues
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype=torch.float16,
        device_map=None,  # Disable automatic device mapping
        low_cpu_mem_usage=True
    )
    
    # Move to GPU/MPS if available
    if torch.cuda.is_available():
        model = model.cuda()
    elif torch.backends.mps.is_available():
        model = model.to("mps")
    else:
        print("⚠️  No GPU available, using CPU (slow)")
        model = model.to("cpu")
    
    # Replace attention layers with Performer
    model = replace_attention_with_performer(model)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Create streamer for real-time output
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    print("\n" + "="*60)
    print("Performer Chat Interface")
    print("="*60)
    print("Model: TinyLlama-1.1B with Performer Attention")
    print("Attention Type: FAVOR+ (Orthogonal Random Features)")
    print("Type 'quit' or 'exit' to stop\n")
    
    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            # Format prompt for TinyLlama chat template
            prompt = f"<|user|>\n{user_input}</s>\n<|assistant|>\n"
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate with streaming
            print("Assistant: ", end="", flush=True)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer
                )
            
            print()  # New line after generation
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


if __name__ == "__main__":
    chat()
