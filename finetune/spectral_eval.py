"""
Spectral analysis: softmax vs FAVOR+ kernel matrices, before and after finetuning.

For each transformer layer (0, 5, 10, 15, 21), computes singular value spectra of:
  1. Softmax kernel matrix (teacher)
  2. FAVOR+ kernel matrix — no finetuning (base model, patched weights)
  3. FAVOR+ kernel matrix — phase 1 finetuned (4/32 heads)
  4. FAVOR+ kernel matrix — phase 2 finetuned (8/32 heads)
  5. FAVOR+ kernel matrix — phase 3 finetuned (16/32 heads)
  6. FAVOR+ kernel matrix — phase 4 finetuned (32/32 heads)

Metrics per layer per model:
  - Top-64 singular values
  - Effective rank = exp(Shannon entropy of normalised singular value distribution)
  - Top-1 coverage = sigma_1 / sum(sigma_i)

Usage:
    python finetune/spectral_eval.py --ckpt_dir /workspace/checkpoints
    python finetune/spectral_eval.py --ckpt_dir /workspace/checkpoints --out docs/spectral_results.json
"""

import sys, os, argparse, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PERFORMER_DIR = os.path.join(_REPO_ROOT, "performer")
if _PERFORMER_DIR not in sys.path:
    sys.path.insert(0, _PERFORMER_DIR)

from performer_attention import PerformerAttentionCore
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID      = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE        = "cuda"
DTYPE         = torch.bfloat16
TARGET_LAYERS = [0, 5, 10, 15, 21]
N_SAMPLES     = 10
SEQ_LEN       = 256
TOP_K_SINGULAR = 64

PROMPT = "The quick brown fox jumps over the lazy dog. " * 20


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


def load_checkpoint(ckpt_path, base_model):
    ckpt    = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    phase   = ckpt["phase"]
    n_heads = int(phase.split("_K")[1].split("_")[0])
    model   = patch_model(base_model, n_heads)
    for i, layer in enumerate(model.model.layers):
        key = f"layer_{i}"
        if key in ckpt["omegas"] and hasattr(layer.self_attn, "performer_core"):
            layer.self_attn.performer_core.omega.copy_(ckpt["omegas"][key].to(DEVICE))
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, n_heads, phase


# ── Spectral metrics ──────────────────────────────────────────────────────────

def effective_rank(singular_values):
    s = np.array(singular_values, dtype=np.float64)
    s = s[s > 0]
    s = s / s.sum()
    entropy = -np.sum(s * np.log(s + 1e-12))
    return float(np.exp(entropy))


def top1_coverage(singular_values):
    s = np.array(singular_values, dtype=np.float64)
    s = s[s > 0]
    return float(s[0] / s.sum()) if len(s) > 0 else 0.0


# ── Q/K extraction and kernel matrix ─────────────────────────────────────────

def get_qk_at_layer(model, input_ids, layer_idx):
    """Run model forward and capture Q, K at a given layer via hook."""
    captured = {}

    def hook(module, inp, kwargs_h, out):
        hs = kwargs_h.get("hidden_states")
        if hs is None and inp:
            hs = inp[0]
        if hs is not None:
            captured["hs"] = hs.detach()

    layer  = model.model.layers[layer_idx]
    handle = layer.self_attn.register_forward_hook(hook, with_kwargs=True)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=DTYPE):
            model(input_ids=input_ids, use_cache=False)

    handle.remove()

    hs       = captured["hs"].to(DTYPE)
    attn     = layer.self_attn
    B, N, _  = hs.shape
    num_heads = attn.config.num_attention_heads
    num_kv    = attn.config.num_key_value_heads
    num_groups = num_heads // num_kv
    head_dim  = attn.head_dim

    q = attn.q_proj(hs).view(B, N, num_heads, head_dim).transpose(1, 2).float()
    k = attn.k_proj(hs).view(B, N, num_kv,    head_dim).transpose(1, 2).float()
    if num_groups > 1:
        k = k.repeat_interleave(num_groups, dim=1)

    return q, k, attn  # [1, H, N, D]


def softmax_kernel_matrix(q, k, head_dim):
    """K_ij = exp(q_i . k_j / sqrt(d)), averaged over heads."""
    scale  = head_dim ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale  # [1, H, N, N]
    K      = torch.exp(scores).mean(dim=1).squeeze(0)                       # [N, N]
    return K.cpu().numpy()


def favor_kernel_matrix(q, k, attn_module, n_performer_heads):
    """K_ij = phi(q_i) . phi(k_j), averaged over performer heads."""
    core  = attn_module.performer_core
    scale = attn_module.head_dim ** -0.25
    phi_q = core.phi(q[:, :n_performer_heads].float() * scale, is_query=True)   # [1, K, N, M]
    phi_k = core.phi(k[:, :n_performer_heads].float() * scale, is_query=False)  # [1, K, N, M]
    # Average over performer heads
    pq = phi_q.squeeze(0).mean(0)  # [N, M]
    pk = phi_k.squeeze(0).mean(0)  # [N, M]
    K  = torch.mm(pq, pk.transpose(0, 1))  # [N, N]
    return K.cpu().numpy()


def compute_spectrum(K_matrix):
    """Singular values of K (top TOP_K_SINGULAR), sorted descending."""
    svd = np.linalg.svd(K_matrix, compute_uv=False)
    return sorted(svd[:TOP_K_SINGULAR].tolist(), reverse=True)


# ── Main extraction ───────────────────────────────────────────────────────────

@torch.no_grad()
def extract_spectra(model, input_ids, layer_idx, mode, n_performer_heads=None):
    """
    mode: 'softmax' or 'favor'
    Returns averaged singular values over N_SAMPLES sequence lengths.
    """
    all_spectra = []

    for i in range(N_SAMPLES):
        length = max(32, SEQ_LEN - i * 4)
        ids    = input_ids[:, :length]

        q, k, attn = get_qk_at_layer(model, ids, layer_idx)

        if mode == "softmax":
            K = softmax_kernel_matrix(q, k, attn.head_dim)
        else:
            K = favor_kernel_matrix(q, k, attn, n_performer_heads)

        all_spectra.append(compute_spectrum(K))

    return np.array(all_spectra).mean(axis=0).tolist()


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    model_keys   = list(results[str(TARGET_LAYERS[0])].keys())
    layer_labels = [f"Layer {l}" for l in TARGET_LAYERS]

    colors = {
        "softmax":        ("tab:blue",   "-",  "Softmax (teacher)"),
        "no_finetune":    ("black",      "--", "FAVOR+ (no finetune)"),
        "phase1_4heads":  ("tab:green",  "-",  "FAVOR+ phase 1 (4/32)"),
        "phase2_8heads":  ("tab:orange", "-",  "FAVOR+ phase 2 (8/32)"),
        "phase3_16heads": ("tab:red",    "-",  "FAVOR+ phase 3 (16/32)"),
        "phase4_32heads": ("tab:purple", "-",  "FAVOR+ phase 4 (32/32)"),
    }

    # ── Plot 1: spectrum shape per layer (one subplot per layer) ──────────────
    fig, axes = plt.subplots(1, len(TARGET_LAYERS), figsize=(18, 4), sharey=False)
    for ax, layer_idx in zip(axes, TARGET_LAYERS):
        layer_data = results[str(layer_idx)]
        for key, (color, ls, label) in colors.items():
            if key not in layer_data:
                continue
            svs = layer_data[key]["singular_values"][:20]
            ax.plot(svs, color=color, linestyle=ls, linewidth=1.8, label=label)
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlabel("Singular value index")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Singular value (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.15))
    fig.suptitle("Kernel singular value spectra — softmax vs FAVOR+ (before/after finetuning)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "spectral_spectra.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved spectral_spectra.png")

    # ── Plot 2: effective rank per layer ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(len(TARGET_LAYERS)))
    for key, (color, ls, label) in colors.items():
        ranks = []
        for layer_idx in TARGET_LAYERS:
            ld = results[str(layer_idx)]
            if key not in ld:
                continue
            ranks.append(ld[key]["effective_rank"])
        if ranks:
            ax.plot(x, ranks, color=color, linestyle=ls, linewidth=2,
                    marker="o", label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.set_ylabel("Effective rank")
    ax.set_title("Effective rank per layer — softmax vs FAVOR+")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "spectral_effective_rank.png"), dpi=150)
    plt.close(fig)
    print("  Saved spectral_effective_rank.png")

    # ── Plot 3: top-1 coverage per layer ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, (color, ls, label) in colors.items():
        coverages = []
        for layer_idx in TARGET_LAYERS:
            ld = results[str(layer_idx)]
            if key not in ld:
                continue
            coverages.append(ld[key]["top1_coverage"] * 100)
        if coverages:
            ax.plot(x, coverages, color=color, linestyle=ls, linewidth=2,
                    marker="o", label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.set_ylabel("Top-1 singular value coverage (%)")
    ax.set_title("Top-1 coverage per layer — softmax vs FAVOR+")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "spectral_top1_coverage.png"), dpi=150)
    plt.close(fig)
    print("  Saved spectral_top1_coverage.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="/workspace/checkpoints")
    parser.add_argument("--out",      default="docs/spectral_results.json")
    parser.add_argument("--plots",    default="docs/plots")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer(PROMPT, return_tensors="pt",
                          max_length=SEQ_LEN, truncation=True)["input_ids"].to(DEVICE)

    ckpt_files = {
        "phase1_4heads":  "best_phase1_K4_QK.pt",
        "phase2_8heads":  "best_phase2_K8_QKVO.pt",
        "phase3_16heads": "best_phase3_K16_QKVO.pt",
        "phase4_32heads": "best_phase4_K32_QKVO.pt",
    }

    results = {str(l): {} for l in TARGET_LAYERS}

    # ── 1. Softmax teacher ────────────────────────────────────────────────────
    print("\nLoading teacher (softmax)...")
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    teacher.eval()

    for layer_idx in TARGET_LAYERS:
        print(f"  Softmax — layer {layer_idx}...")
        svs = extract_spectra(teacher, input_ids, layer_idx, mode="softmax")
        results[str(layer_idx)]["softmax"] = {
            "singular_values": svs,
            "effective_rank":  effective_rank(svs),
            "top1_coverage":   top1_coverage(svs),
        }

    del teacher
    torch.cuda.empty_cache()

    # ── 2. FAVOR+ without finetuning ─────────────────────────────────────────
    print("\nLoading base model (no finetuning) + patching 4 performer heads...")
    base_noft = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    base_noft = patch_model(base_noft, num_performer_heads=4)
    base_noft.eval()

    for layer_idx in TARGET_LAYERS:
        print(f"  No finetune — layer {layer_idx}...")
        svs = extract_spectra(base_noft, input_ids, layer_idx, mode="favor", n_performer_heads=4)
        results[str(layer_idx)]["no_finetune"] = {
            "singular_values": svs,
            "effective_rank":  effective_rank(svs),
            "top1_coverage":   top1_coverage(svs),
        }

    del base_noft
    torch.cuda.empty_cache()

    # ── 3-6. Finetuned phases ─────────────────────────────────────────────────
    for result_key, fname in ckpt_files.items():
        ckpt_path = os.path.join(args.ckpt_dir, fname)
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {fname} (not found)")
            continue

        print(f"\nLoading {fname}...")
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
        model, n_heads, phase = load_checkpoint(ckpt_path, base)

        for layer_idx in TARGET_LAYERS:
            print(f"  {result_key} — layer {layer_idx}...")
            svs = extract_spectra(model, input_ids, layer_idx, mode="favor",
                                  n_performer_heads=n_heads)
            results[str(layer_idx)][result_key] = {
                "singular_values": svs,
                "effective_rank":  effective_rank(svs),
                "top1_coverage":   top1_coverage(svs),
            }

        del model
        torch.cuda.empty_cache()

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'Layer':<8} {'Model':<20} {'Eff. rank':>10} {'Top-1 cov':>10}")
    print("-" * 52)
    for layer_idx in TARGET_LAYERS:
        for key, val in results[str(layer_idx)].items():
            print(f"{layer_idx:<8} {key:<20} {val['effective_rank']:>10.3f} {val['top1_coverage']*100:>9.1f}%")
        print()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.out}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    make_plots(results, args.plots)
    print(f"Plots saved to {args.plots}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
