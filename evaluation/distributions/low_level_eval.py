"""
Low-level evaluation: student (finetuned performer) vs teacher (softmax).

Metrics per phase (4/32, 8/32, 16/32, 32/32 performer heads):
  1. KL divergence  — token-level, averaged over sequence
  2. Top-1 agreement — fraction of tokens where student == teacher argmax
  3. Hidden state MSE — per transformer layer, averaged over tokens

Usage:
    python finetune/low_level_eval.py --ckpt_dir /workspace/checkpoints
    python finetune/low_level_eval.py --ckpt_dir /workspace/checkpoints --out docs/low_level_results.json
"""

import sys, os, argparse, json
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PERFORMER_DIR = os.path.join(_REPO_ROOT, "performer")
if _PERFORMER_DIR not in sys.path:
    sys.path.insert(0, _PERFORMER_DIR)

from performer_attention import PerformerAttentionCore
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE   = "cuda"
DTYPE    = torch.bfloat16

# Evaluation texts — diverse domains to avoid training distribution bias
EVAL_TEXTS = [
    "The history of artificial intelligence begins with the development of the first computers in the 1940s.",
    "In mathematics, a prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.",
    "The French Revolution began in 1789 and led to the rise of Napoleon Bonaparte as Emperor of France.",
    "Quantum mechanics is a fundamental theory in physics that describes nature at the smallest scales of energy levels.",
    "Climate change refers to long-term shifts in global temperatures and weather patterns caused by human activities.",
    "The human genome contains approximately three billion base pairs encoding around twenty thousand protein-coding genes.",
    "Shakespeare wrote thirty-seven plays and one hundred fifty-four sonnets during his lifetime in the sixteenth century.",
    "The theory of relativity introduced by Einstein revolutionized our understanding of space, time, and gravity.",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy stored as glucose molecules.",
    "The industrial revolution transformed manufacturing processes and led to urbanization across Europe and North America.",
]


# ── MixedPerformerAttention ───────────────────────────────────────────────────

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
    ckpt    = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    phase   = ckpt["phase"]
    n_heads = int(phase.split("_K")[1].split("_")[0])
    base    = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    model   = patch_model(base, n_heads)
    for i, layer in enumerate(model.model.layers):
        key = f"layer_{i}"
        if key in ckpt["omegas"] and hasattr(layer.self_attn, "performer_core"):
            layer.self_attn.performer_core.omega.copy_(ckpt["omegas"][key].to(DEVICE))
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, n_heads, phase


# ── Hook-based hidden state extraction ───────────────────────────────────────

def register_hidden_state_hooks(model):
    """Attach forward hooks to collect hidden states after each attention layer."""
    hidden_states = {}
    hooks = []

    for i, layer in enumerate(model.model.layers):
        def make_hook(idx):
            def hook(module, inp, out):
                # out is (hidden_state, ...) for LlamaDecoderLayer
                hs = out[0] if isinstance(out, tuple) else out
                hidden_states[idx] = hs.detach().float()
            return hook
        hooks.append(layer.register_forward_hook(make_hook(i)))

    return hidden_states, hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ── Core metrics ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_pair(teacher, student, tokenizer, texts):
    """
    Compare teacher and student on a list of texts.
    Returns:
        kl_mean        : mean KL divergence per token (nats)
        top1_agreement : fraction of tokens with same argmax
        hs_mse_per_layer: list of MSE per transformer layer
    """
    num_layers = len(teacher.model.layers)

    total_kl      = 0.0
    total_top1    = 0.0
    total_tokens  = 0
    layer_mse_sum = [0.0] * num_layers
    layer_tokens  = 0

    for text in texts:
        ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
        if ids.shape[1] < 4:
            continue
        ids = ids[:, :512]  # cap at 512 to stay within model limits

        # — Teacher forward with hidden state hooks —
        t_hidden, t_hooks = register_hidden_state_hooks(teacher)
        t_out  = teacher(input_ids=ids, use_cache=False)
        remove_hooks(t_hooks)
        t_logits = t_out.logits[:, :-1].float()   # [1, N-1, V]

        # — Student forward with hidden state hooks —
        s_hidden, s_hooks = register_hidden_state_hooks(student)
        s_out  = student(input_ids=ids, use_cache=False)
        remove_hooks(s_hooks)
        s_logits = s_out.logits[:, :-1].float()   # [1, N-1, V]

        N = t_logits.shape[1]

        # KL divergence: KL(teacher || student) per token
        t_log_probs = F.log_softmax(t_logits, dim=-1)
        s_log_probs = F.log_softmax(s_logits, dim=-1)
        t_probs     = t_log_probs.exp()
        # KL(P||Q) = sum P * (log P - log Q)
        kl = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)  # [1, N-1]
        total_kl   += kl.sum().item()

        # Top-1 agreement
        t_top1 = t_logits.argmax(dim=-1)   # [1, N-1]
        s_top1 = s_logits.argmax(dim=-1)
        total_top1   += (t_top1 == s_top1).float().sum().item()
        total_tokens += N

        # Hidden state MSE per layer (on full sequence including last token)
        for layer_idx in range(num_layers):
            if layer_idx in t_hidden and layer_idx in s_hidden:
                th = t_hidden[layer_idx]   # [1, N, D]
                sh = s_hidden[layer_idx]
                mse = ((th - sh) ** 2).mean().item()
                layer_mse_sum[layer_idx] += mse
        layer_tokens += 1  # average over texts, not tokens (already mean over N, D)

    kl_mean         = total_kl / max(total_tokens, 1)
    top1_agreement  = total_top1 / max(total_tokens, 1)
    hs_mse_per_layer = [s / max(layer_tokens, 1) for s in layer_mse_sum]

    return kl_mean, top1_agreement, hs_mse_per_layer


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    phases      = [r["n_heads"] for r in results]
    kl_vals     = [r["kl_mean"] for r in results]
    top1_vals   = [r["top1_agreement"] * 100 for r in results]  # as %
    labels      = [f"{k}/32" for k in phases]

    # ── Plot 1: KL divergence per phase ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, kl_vals, color="tab:blue", alpha=0.8)
    ax.set_xlabel("Performer heads replaced")
    ax.set_ylabel("Mean KL divergence (nats/token)")
    ax.set_title("KL divergence — student vs teacher")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    for i, v in enumerate(kl_vals):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "kl_divergence.png"), dpi=150)
    plt.close(fig)
    print("  Saved kl_divergence.png")

    # ── Plot 2: Top-1 agreement per phase ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, top1_vals, color="tab:green", alpha=0.8)
    ax.set_xlabel("Performer heads replaced")
    ax.set_ylabel("Top-1 agreement with teacher (%)")
    ax.set_title("Token-level agreement — student vs teacher")
    ax.set_ylim(0, 105)
    for i, v in enumerate(top1_vals):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "top1_agreement.png"), dpi=150)
    plt.close(fig)
    print("  Saved top1_agreement.png")

    # ── Plot 3: Hidden state MSE per layer for each phase ────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tab:blue", "tab:orange", "tab:red", "tab:purple"]
    for i, r in enumerate(results):
        ax.plot(r["hs_mse_per_layer"], label=f"{r['n_heads']}/32 heads",
                color=colors[i], linewidth=2)
    ax.set_xlabel("Transformer layer index")
    ax.set_ylabel("Hidden state MSE (teacher vs student)")
    ax.set_title("Hidden state divergence per layer — all phases")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hidden_state_mse.png"), dpi=150)
    plt.close(fig)
    print("  Saved hidden_state_mse.png")

    # ── Plot 4: Summary — KL and top-1 on same figure ────────────────────────
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    x = range(len(phases))
    ax1.plot(list(x), kl_vals,   "o-", color="tab:blue",  linewidth=2, label="KL divergence")
    ax2.plot(list(x), top1_vals, "s--", color="tab:green", linewidth=2, label="Top-1 agreement (%)")
    ax1.set_xlabel("Performer heads replaced")
    ax1.set_ylabel("KL divergence (nats/token)", color="tab:blue")
    ax2.set_ylabel("Top-1 agreement (%)", color="tab:green")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="center right")
    ax1.set_title("Low-level quality metrics vs number of performer heads")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary.png"), dpi=150)
    plt.close(fig)
    print("  Saved summary.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="/workspace/checkpoints")
    parser.add_argument("--out",      default="docs/low_level_results.json")
    parser.add_argument("--plots",    default="docs/plots")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ckpt_files = [
        ("best_phase1_K4_QK.pt",    4),
        ("best_phase2_K8_QKVO.pt",  8),
        ("best_phase3_K16_QKVO.pt", 16),
        ("best_phase4_K32_QKVO.pt", 32),
    ]
    ckpt_files = [(os.path.join(args.ckpt_dir, f), k)
                  for f, k in ckpt_files
                  if os.path.exists(os.path.join(args.ckpt_dir, f))]

    if not ckpt_files:
        print("No checkpoints found. Check --ckpt_dir.")
        return

    # Load teacher once
    print("Loading teacher (softmax)...")
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    teacher.eval()

    print(f"\n{'Phase':<20} {'KL (nats/tok)':>14} {'Top-1 agree':>12} {'Avg HS MSE':>12}")
    print("-" * 62)

    all_results = []

    for ckpt_path, n_heads in ckpt_files:
        print(f"\nEvaluating phase {n_heads}/32...")
        student, _, phase = load_checkpoint(ckpt_path)

        kl, top1, hs_mse = evaluate_pair(teacher, student, tokenizer, EVAL_TEXTS)
        avg_hs_mse = sum(hs_mse) / len(hs_mse)

        print(f"  Phase {n_heads}/32  |  KL={kl:.4f}  |  Top-1={top1*100:.1f}%  |  HS MSE={avg_hs_mse:.6f}")

        all_results.append({
            "phase":            phase,
            "n_heads":          n_heads,
            "kl_mean":          round(kl, 6),
            "top1_agreement":   round(top1, 6),
            "hs_mse_per_layer": [round(v, 8) for v in hs_mse],
            "hs_mse_avg":       round(avg_hs_mse, 8),
        })

        del student
        torch.cuda.empty_cache()

    del teacher
    torch.cuda.empty_cache()

    # Save JSON
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.out}")

    # Plots
    make_plots(all_results, args.plots)
    print(f"Plots saved to {args.plots}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
