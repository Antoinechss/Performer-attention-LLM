"""
Kernel spectrum comparison: eigenvalues of (1/n) * K where K_ij = k(x_i, x_j).

Computes empirical eigenvalues of the Gram matrix (normalized by n) for:
  - Softmax kernel:  k(q_i, k_j) = exp(q_i . k_j / sqrt(d))
  - FAVOR+ kernel:   k(q_i, k_j) = phi(q_i) . phi(k_j)

For each transformer layer in TARGET_LAYERS, averaged over N_SAMPLES prompts.
Results saved to kernel_spectrum_results.json, plots to docs/plots/.

Usage:
    python finetune/kernel_spectrum.py --ckpt_dir /workspace/checkpoints
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
N_SAMPLES     = 5
SEQ_LEN       = 128   # n tokens = size of Gram matrix
TOP_K_EIG     = 64

PROMPTS = [
    "The quick brown fox jumps over the lazy dog. " * 10,
    "Artificial intelligence is transforming the way we interact with technology. " * 6,
    "In the beginning, there was nothing. Then, slowly, the universe came into being. " * 5,
    "Mathematics is the language in which God has written the universe. " * 7,
    "Climate change poses significant risks to global ecosystems and human societies. " * 5,
]


# ── MixedPerformerAttention (identical to spectral_eval.py) ──────────────────

class MixedPerformerAttention(torch.nn.Module):
    def __init__(self, original_attn, num_performer_heads):
        super().__init__()
        self.head_dim             = original_attn.head_dim
        self.num_heads            = original_attn.config.num_attention_heads
        self.num_key_value_heads  = original_attn.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling              = self.head_dim ** -0.5
        self.num_performer_heads  = num_performer_heads
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

    @property
    def num_standard_heads(self):
        return self.num_heads - self.num_performer_heads


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


# ── Q/K extraction ────────────────────────────────────────────────────────────

def get_qk(model, input_ids, layer_idx):
    """Capture Q, K at a given layer via hook. Returns [1, H, N, D] float32."""
    captured = {}

    def hook(module, inp, kwargs_h, out):
        hs = kwargs_h.get("hidden_states")
        if hs is None and inp:
            hs = inp[0]
        if hs is not None:
            captured["hs"] = hs.detach()

    handle = model.model.layers[layer_idx].self_attn.register_forward_hook(hook, with_kwargs=True)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=DTYPE):
            model(input_ids=input_ids, use_cache=False)

    handle.remove()

    attn = model.model.layers[layer_idx].self_attn
    hs   = captured["hs"].to(DTYPE)
    B, N, _ = hs.shape

    num_heads  = attn.config.num_attention_heads
    num_kv     = attn.config.num_key_value_heads
    num_groups = num_heads // num_kv
    head_dim   = attn.head_dim

    q = attn.q_proj(hs).view(B, N, num_heads, head_dim).transpose(1, 2).float()  # [1, H, N, D]
    k = attn.k_proj(hs).view(B, N, num_kv,    head_dim).transpose(1, 2).float()
    if num_groups > 1:
        k = k.repeat_interleave(num_groups, dim=1)

    return q, k, attn


# ── Gram matrices ─────────────────────────────────────────────────────────────

def gram_softmax(q, k, head_dim):
    """
    K_ij = exp(q_i . k_j / sqrt(d)), averaged over heads, then normalized by n.
    Returns symmetric-ish [N, N] numpy array (averaged over heads).
    """
    scale = head_dim ** -0.5
    # [1, H, N, N]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    K = torch.exp(scores).mean(dim=1).squeeze(0)  # [N, N]
    n = K.shape[0]
    return (K / n).cpu().numpy()


def gram_favor(q, k, attn, n_performer_heads):
    """
    K_ij = phi(q_i) . phi(k_j), averaged over performer heads, normalized by n.
    """
    core  = attn.performer_core
    scale = attn.head_dim ** -0.25
    phi_q = core.phi(q[:, :n_performer_heads] * scale, is_query=True)   # [1, K, N, M]
    phi_k = core.phi(k[:, :n_performer_heads] * scale, is_query=False)  # [1, K, N, M]
    pq = phi_q.squeeze(0).mean(0)   # [N, M]
    pk = phi_k.squeeze(0).mean(0)   # [N, M]
    K  = torch.mm(pq, pk.t())       # [N, N]  — rank-M factorization
    n  = K.shape[0]
    return (K / n).cpu().numpy()


# ── Eigenvalues ───────────────────────────────────────────────────────────────

def eigenvalues(K_np):
    """
    Eigenvalues of a (approximately) symmetric matrix, sorted descending.
    We symmetrize K to handle floating-point asymmetry before eigh.
    """
    K_sym = (K_np + K_np.T) / 2.0
    eigs  = np.linalg.eigvalsh(K_sym)          # real eigenvalues, ascending
    eigs  = np.sort(eigs)[::-1]                # descending
    return eigs[:TOP_K_EIG].tolist()


# ── Per-model extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_eigenvalues(model, tokenizer, layer_idx, mode, n_performer_heads=None):
    """Average eigenvalue spectrum over N_SAMPLES prompts."""
    all_eigs = []
    for prompt in PROMPTS[:N_SAMPLES]:
        ids = tokenizer(prompt, return_tensors="pt",
                        max_length=SEQ_LEN, truncation=True)["input_ids"].to(DEVICE)
        q, k, attn = get_qk(model, ids, layer_idx)
        if mode == "softmax":
            K = gram_softmax(q, k, attn.head_dim)
        else:
            K = gram_favor(q, k, attn, n_performer_heads)
        all_eigs.append(eigenvalues(K))
    return np.array(all_eigs).mean(axis=0).tolist()


# ── Plots ─────────────────────────────────────────────────────────────────────

STYLE = {
    "softmax":        ("#1f77b4", "-",  2.0, "Softmax"),
    "no_finetune":    ("#333333", "--", 1.5, "FAVOR+ (no FT)"),
    "phase1_4heads":  ("#2ca02c", "-",  1.5, "FAVOR+ phase 1 (4h)"),
    "phase2_8heads":  ("#ff7f0e", "-",  1.5, "FAVOR+ phase 2 (8h)"),
    "phase3_16heads": ("#d62728", "-",  1.5, "FAVOR+ phase 3 (16h)"),
    "phase4_32heads": ("#9467bd", "-",  1.5, "FAVOR+ phase 4 (32h)"),
}


def make_plots(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    n_layers = len(TARGET_LAYERS)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4), sharey=False)

    for ax, layer_idx in zip(axes, TARGET_LAYERS):
        layer_data = results[str(layer_idx)]
        for key, (color, ls, lw, label) in STYLE.items():
            if key not in layer_data:
                continue
            eigs = layer_data[key]["eigenvalues"]
            # only plot positive eigenvalues on semilogy
            eigs_pos = [max(e, 1e-20) for e in eigs]
            ax.semilogy(range(1, len(eigs_pos) + 1), eigs_pos,
                        color=color, linestyle=ls, linewidth=lw, label=label)
        ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.set_xlabel("Eigenvalue index")
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel("Eigenvalue of (1/n)·K  [log scale]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.18), fontsize=9)
    fig.suptitle("Empirical kernel spectrum: eigenvalues of (1/n)·K\n"
                 "Softmax vs FAVOR+ (before/after distillation)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "kernel_spectrum.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="/workspace/checkpoints")
    parser.add_argument("--out",      default="kernel_spectrum_results.json")
    parser.add_argument("--plots",    default="docs/plots")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ckpt_files = {
        "phase1_4heads":  "best_phase1_K4_QK.pt",
        "phase2_8heads":  "best_phase2_K8_QKVO.pt",
        "phase3_16heads": "best_phase3_K16_QKVO.pt",
        "phase4_32heads": "best_phase4_K32_QKVO.pt",
    }

    results = {str(l): {} for l in TARGET_LAYERS}

    # 1. Softmax teacher
    print("\n[1/6] Softmax teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    teacher.eval()
    for layer_idx in TARGET_LAYERS:
        print(f"      layer {layer_idx}...")
        eigs = extract_eigenvalues(teacher, tokenizer, layer_idx, mode="softmax")
        results[str(layer_idx)]["softmax"] = {"eigenvalues": eigs}
    del teacher; torch.cuda.empty_cache()

    # 2. FAVOR+ no finetuning
    print("\n[2/6] FAVOR+ no finetuning...")
    base_noft = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    base_noft = patch_model(base_noft, num_performer_heads=4)
    base_noft.eval()
    for layer_idx in TARGET_LAYERS:
        print(f"      layer {layer_idx}...")
        eigs = extract_eigenvalues(base_noft, tokenizer, layer_idx, mode="favor", n_performer_heads=4)
        results[str(layer_idx)]["no_finetune"] = {"eigenvalues": eigs}
    del base_noft; torch.cuda.empty_cache()

    # 3-6. Finetuned phases
    for i, (result_key, fname) in enumerate(ckpt_files.items(), start=3):
        ckpt_path = os.path.join(args.ckpt_dir, fname)
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {fname} (not found)")
            continue
        print(f"\n[{i}/6] {result_key}...")
        base  = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
        model, n_heads, _ = load_checkpoint(ckpt_path, base)
        for layer_idx in TARGET_LAYERS:
            print(f"      layer {layer_idx}...")
            eigs = extract_eigenvalues(model, tokenizer, layer_idx, mode="favor",
                                       n_performer_heads=n_heads)
            results[str(layer_idx)][result_key] = {"eigenvalues": eigs}
        del model; torch.cuda.empty_cache()

    # Save JSON
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")

    # Plots
    make_plots(results, args.plots)
    print("Done.")


if __name__ == "__main__":
    main()
