"""
Interactive generation and perplexity comparison across checkpoints.

Usage:
    python finetune/generate.py --ckpt_dir /workspace/checkpoints
    python finetune/generate.py --ckpt hf  # download from HuggingFace
"""

import sys, os, argparse, math
import torch
import torch.nn.functional as F

_REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PERFORMER_DIR = os.path.join(_REPO_ROOT, "performer")
if _PERFORMER_DIR not in sys.path:
    sys.path.insert(0, _PERFORMER_DIR)

from performer_attention import PerformerAttentionCore
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE   = "cuda"
DTYPE    = torch.bfloat16


# ── MixedPerformerAttention (same as finetune.py) ─────────────────────────────

class MixedPerformerAttention(torch.nn.Module):
    def __init__(self, original_attn, num_performer_heads):
        super().__init__()
        self.head_dim             = original_attn.head_dim
        self.num_heads            = original_attn.config.num_attention_heads
        self.num_key_value_heads  = original_attn.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling              = self.head_dim ** -0.5
        self.num_performer_heads  = num_performer_heads
        self.num_standard_heads   = self.num_heads - num_performer_heads
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj
        self.performer_core = PerformerAttentionCore(head_dim=self.head_dim, num_features=256)
        self.config    = original_attn.config
        self.layer_idx = original_attn.layer_idx
        self.is_causal = True

    def _rotate_half(self, x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary(self, q, k, cos, sin):
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        return (q * cos) + (self._rotate_half(q) * sin), (k * cos) + (self._rotate_half(k) * sin)

    def forward(self, hidden_states, position_embeddings=None,
                attention_mask=None, past_key_values=None, **kwargs):
        B, N, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, N, self.num_heads,           self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, N, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, N, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = self._apply_rotary(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        if attention_mask is None and N > 1 and self.num_standard_heads > 0:
            key_len = k.shape[2]
            causal  = torch.full((N, key_len), torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype)
            attention_mask = torch.triu(causal, diagonal=key_len - N + 1)[None, None]
        if self.num_standard_heads == 0:
            attn_out = self.performer_core(q, k, v)
        elif self.num_performer_heads == 0:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
            if attention_mask is not None:
                scores = scores + attention_mask
            w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_out = torch.matmul(w, v)
        else:
            Kp = self.num_performer_heads
            out_p = self.performer_core(q[:, :Kp], k[:, :Kp], v[:, :Kp])
            q_s, k_s, v_s = q[:, Kp:], k[:, Kp:], v[:, Kp:]
            scores = torch.matmul(q_s, k_s.transpose(-2, -1)) * self.scaling
            if attention_mask is not None:
                scores = scores + attention_mask
            w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q_s.dtype)
            out_s = torch.matmul(w, v_s)
            attn_out = torch.cat([out_p, out_s], dim=1)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, N, -1)
        return self.o_proj(attn_out), None, None


def patch_model(model, num_performer_heads):
    for layer in model.model.layers:
        layer.self_attn = MixedPerformerAttention(layer.self_attn, num_performer_heads)
    return model


def load_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    phase = ckpt["phase"]
    n_heads = int(phase.split("_K")[1].split("_")[0])
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    model = patch_model(base, n_heads)
    for i, layer in enumerate(model.model.layers):
        key = f"layer_{i}"
        if key in ckpt["omegas"] and hasattr(layer.self_attn, "performer_core"):
            layer.self_attn.performer_core.omega.copy_(ckpt["omegas"][key].to(DEVICE))
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"  Loaded {os.path.basename(ckpt_path)} — {n_heads}/32 performer heads")
    return model, n_heads, phase


# ── Generation ────────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ── Perplexity ────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_ppl(model, tokenizer, texts, seq_len=512):
    total_ce, n_tokens = 0.0, 0
    for text in texts:
        ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
        if ids.shape[1] < 10:
            continue
        ids = ids[:, :seq_len]
        with torch.amp.autocast("cuda", dtype=DTYPE):
            out = model(input_ids=ids, use_cache=False)
        logits = out.logits[:, :-1].float()
        tgt    = ids[:, 1:].contiguous()
        ce     = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="sum")
        total_ce += ce.item()
        n_tokens += tgt.numel()
    return math.exp(min(total_ce / max(n_tokens, 1), 20))


# ── Main ──────────────────────────────────────────────────────────────────────

PROMPTS = [
    "The history of artificial intelligence begins",
    "In mathematics, a prime number is",
    "The capital of France is Paris. The Eiffel Tower",
    "Once upon a time in a land far away,",
    "The process of photosynthesis involves",
]

VAL_TEXTS = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "The French Revolution began in 1789 and fundamentally transformed French society and government.",
    "Quantum mechanics describes the behavior of matter and energy at the atomic and subatomic levels.",
    "Climate change refers to long-term shifts in global temperatures and weather patterns.",
    "The human genome contains approximately three billion base pairs of DNA.",
] * 20  # 100 short texts for quick perplexity estimate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="/workspace/checkpoints")
    parser.add_argument("--phase", default=None, help="Specific phase to load, e.g. best_phase2_K8_QKVO")
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine which checkpoints to load
    if args.phase:
        ckpt_files = [os.path.join(args.ckpt_dir, f"{args.phase}.pt")]
    else:
        ckpt_files = [
            os.path.join(args.ckpt_dir, "best_phase1_K4_QK.pt"),
            os.path.join(args.ckpt_dir, "best_phase2_K8_QKVO.pt"),
            os.path.join(args.ckpt_dir, "best_phase3_K16_QKVO.pt"),
            os.path.join(args.ckpt_dir, "best_phase4_K32_QKVO.pt"),
        ]
        ckpt_files = [f for f in ckpt_files if os.path.exists(f)]

    # Teacher baseline
    print("\nLoading teacher (softmax baseline)...")
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    teacher.eval()

    print("\n" + "="*70)
    print("TEACHER (0/32 performer heads — pure softmax)")
    print("="*70)
    teacher_ppl = compute_ppl(teacher, tokenizer, VAL_TEXTS)
    print(f"Perplexity: {teacher_ppl:.2f}")
    print("\nGenerations:")
    for prompt in PROMPTS[:3]:
        print(f"\n  Prompt: {prompt!r}")
        print(f"  Output: {generate(teacher, tokenizer, prompt, args.max_new_tokens, args.temperature)}")

    del teacher
    torch.cuda.empty_cache()

    # Per-checkpoint eval
    for ckpt_path in ckpt_files:
        print(f"\n{'='*70}")
        model, n_heads, phase = load_checkpoint(ckpt_path)
        print(f"CHECKPOINT: {phase} ({n_heads}/32 performer heads)")
        print("="*70)

        ppl = compute_ppl(model, tokenizer, VAL_TEXTS)
        print(f"Perplexity: {ppl:.2f}")

        print("\nGenerations:")
        for prompt in PROMPTS[:3]:
            print(f"\n  Prompt: {prompt!r}")
            print(f"  Output: {generate(model, tokenizer, prompt, args.max_new_tokens, args.temperature)}")

        del model
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
