"""
Full spectral analysis of softmax vs FAVOR+ kernel Gram matrices.

For each layer and each model phase, computes:
  - Eigenvalues of (1/n)(K + K^T)/2          [integral operator theory]
  - Singular values of (1/n)K                 [asymmetric structure]
  - Frobenius norm ||K_softmax - K_favor||_F  [Hoffmann-Wielandt bound]
  - Asymmetry ||K - K^T||_F                   [Q/K alignment]
  - Effective rank r_eff = exp(H(p))           [spectral concentration]
  - LHS of Hoffmann-Wielandt: sum|λi - μi|^2  [bound verification]

Usage:
    python evaluation/spectral/kernel_spectrum.py --ckpt_dir /workspace/checkpoints
"""

import sys, os, argparse, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
SEQ_LEN       = 128
TOP_K_EIG     = 64
TOP_K_SV      = 64

PROMPTS = [
    "The quick brown fox jumps over the lazy dog. " * 10,
    "Artificial intelligence is transforming the way we interact with technology. " * 6,
    "In the beginning, there was nothing. Then, slowly, the universe came into being. " * 5,
    "Mathematics is the language in which God has written the universe. " * 7,
    "Climate change poses significant risks to global ecosystems and human societies. " * 5,
]

SEP = "=" * 70


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

    q = attn.q_proj(hs).view(B, N, num_heads, head_dim).transpose(1, 2).float()
    k = attn.k_proj(hs).view(B, N, num_kv,    head_dim).transpose(1, 2).float()
    if num_groups > 1:
        k = k.repeat_interleave(num_groups, dim=1)

    return q, k, attn


# ── Gram matrices (raw, not symmetrized) ──────────────────────────────────────

def gram_softmax_raw(q, k, head_dim):
    """Returns (1/n) * K where K_ij = exp(q_i.k_j/sqrt(d)), averaged over heads. [N,N] numpy."""
    scale  = head_dim ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    K      = torch.exp(scores).mean(dim=1).squeeze(0)
    return (K / K.shape[0]).cpu().numpy()


def gram_favor_raw(q, k, attn, n_performer_heads):
    """Returns (1/n) * K where K_ij = phi(q_i).phi(k_j), averaged over performer heads. [N,N] numpy."""
    core  = attn.performer_core
    scale = attn.head_dim ** -0.25
    phi_q = core.phi(q[:, :n_performer_heads] * scale, is_query=True)
    phi_k = core.phi(k[:, :n_performer_heads] * scale, is_query=False)
    pq = phi_q.squeeze(0).mean(0)
    pk = phi_k.squeeze(0).mean(0)
    K  = torch.mm(pq, pk.t())
    return (K / K.shape[0]).cpu().numpy()


# ── Spectral metrics ──────────────────────────────────────────────────────────

def compute_eigenvalues(K_np):
    """Eigenvalues of symmetrized (K+K^T)/2, sorted descending."""
    K_sym = (K_np + K_np.T) / 2.0
    eigs  = np.linalg.eigvalsh(K_sym)
    return np.sort(eigs)[::-1][:TOP_K_EIG]


def compute_singular_values(K_np):
    """Singular values of K (no symmetrization), sorted descending."""
    svs = np.linalg.svd(K_np, compute_uv=False)
    return svs[:TOP_K_SV]


def compute_asymmetry(K_np):
    """||K - K^T||_F / ||K||_F  (relative asymmetry)."""
    diff = K_np - K_np.T
    fro_k    = np.linalg.norm(K_np,  'fro')
    fro_diff = np.linalg.norm(diff,   'fro')
    return float(fro_diff), float(fro_diff / (fro_k + 1e-12))


def compute_effective_rank(eigs):
    """r_eff = exp(Shannon entropy of normalized |eigenvalue| distribution)."""
    vals = np.abs(eigs)
    vals = vals[vals > 1e-15]
    if len(vals) == 0:
        return 0.0
    p = vals / vals.sum()
    H = -np.sum(p * np.log(p + 1e-30))
    return float(np.exp(H))


def compute_frobenius_norm(K_np):
    return float(np.linalg.norm(K_np, 'fro'))


def compute_hw_lhs(eigs_a, eigs_b):
    """Hoffmann-Wielandt LHS: sum_i |lambda_i(A) - lambda_i(B)|^2."""
    n = min(len(eigs_a), len(eigs_b))
    return float(np.sum((np.array(eigs_a[:n]) - np.array(eigs_b[:n])) ** 2))


# ── Per-prompt extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_all_metrics(model, tokenizer, layer_idx, mode, n_performer_heads=None):
    """
    Returns averaged metrics over N_SAMPLES prompts.
    Also returns raw Gram matrices (averaged) for cross-model comparisons.
    """
    all_eigs, all_svs, all_asym_abs, all_asym_rel, all_fro, all_reff = [], [], [], [], [], []
    K_avg = None  # averaged raw Gram matrix for cross-model Frobenius distance

    for prompt in PROMPTS[:N_SAMPLES]:
        ids = tokenizer(prompt, return_tensors="pt",
                        max_length=SEQ_LEN, truncation=True)["input_ids"].to(DEVICE)
        q, k, attn = get_qk(model, ids, layer_idx)

        if mode == "softmax":
            K = gram_softmax_raw(q, k, attn.head_dim)
        else:
            K = gram_favor_raw(q, k, attn, n_performer_heads)

        K_avg = K if K_avg is None else K_avg + K

        eigs = compute_eigenvalues(K)
        svs  = compute_singular_values(K)
        asym_abs, asym_rel = compute_asymmetry(K)
        fro  = compute_frobenius_norm(K)
        reff = compute_effective_rank(eigs)

        all_eigs.append(eigs)
        all_svs.append(svs)
        all_asym_abs.append(asym_abs)
        all_asym_rel.append(asym_rel)
        all_fro.append(fro)
        all_reff.append(reff)

    K_avg /= N_SAMPLES

    return {
        "eigenvalues":     np.array(all_eigs).mean(axis=0).tolist(),
        "singular_values": np.array(all_svs).mean(axis=0).tolist(),
        "asymmetry_abs":   float(np.mean(all_asym_abs)),
        "asymmetry_rel":   float(np.mean(all_asym_rel)),
        "frobenius_norm":  float(np.mean(all_fro)),
        "effective_rank":  float(np.mean(all_reff)),
    }, K_avg


# ── Pretty printing ───────────────────────────────────────────────────────────

def print_model_summary(label, layer_results):
    print(f"\n  {label}")
    print(f"  {'Layer':<8} {'λ₁':>10} {'λ₂':>10} {'σ₁':>10} {'Asym%':>8} {'r_eff':>8} {'||K||_F':>10}")
    print(f"  {'-'*64}")
    for layer_idx in TARGET_LAYERS:
        d = layer_results[str(layer_idx)]
        eigs = d["eigenvalues"]
        svs  = d["singular_values"]
        print(f"  layer {layer_idx:<3} "
              f"{eigs[0]:>10.4f} "
              f"{eigs[1]:>10.4f} "
              f"{svs[0]:>10.4f} "
              f"{d['asymmetry_rel']*100:>7.2f}% "
              f"{d['effective_rank']:>8.2f} "
              f"{d['frobenius_norm']:>10.4f}")


def print_hw_table(results):
    print(f"\n{SEP}")
    print("  Hoffmann-Wielandt verification: sum|λi(softmax) - λi(favor)|²")
    print(f"  (bound: ||K_softmax - K_favor||²_F)")
    print(f"  {'Layer':<8} {'HW_LHS':>12} {'||Ks-Kf||²_F':>16} {'Bound tight?':>14}")
    print(f"  {'-'*54}")
    for layer_idx in TARGET_LAYERS:
        l = str(layer_idx)
        for key in ["no_finetune", "phase1_4heads", "phase2_8heads", "phase3_16heads", "phase4_32heads"]:
            if key not in results[l]:
                continue
            eigs_s = results[l]["softmax"]["eigenvalues"]
            eigs_f = results[l][key]["eigenvalues"]
            lhs    = compute_hw_lhs(eigs_s, eigs_f)
            rhs    = results[l][key].get("frobenius_dist_to_softmax", float("nan"))
            ratio  = lhs / (rhs + 1e-30) if not np.isnan(rhs) else float("nan")
            print(f"  layer {layer_idx:<3} [{key:<18}]  LHS={lhs:>10.4f}  ||Ks-Kf||²={rhs:>10.4f}  ratio={ratio:.3f}")


# ── Plots ─────────────────────────────────────────────────────────────────────

STYLE = {
    "softmax":         ("#1f77b4", "-",  2.0, "Softmax"),
    "no_finetune":     ("#333333", "--", 1.5, "FAVOR+ (no FT)"),
    "phase1_4heads":   ("#2ca02c", "-",  1.5, "FAVOR+ phase 1 (4h)"),
    "phase2_8heads":   ("#ff7f0e", "-",  1.5, "FAVOR+ phase 2 (8h)"),
    "phase3_16heads":  ("#d62728", "-",  1.5, "FAVOR+ phase 3 (16h)"),
    "phase4_32heads":  ("#9467bd", "-",  1.5, "FAVOR+ phase 4 (32h)"),
}


def make_plots(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # One combined plot
    n_layers = len(TARGET_LAYERS)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4), sharey=False)
    for ax, layer_idx in zip(axes, TARGET_LAYERS):
        layer_data = results[str(layer_idx)]
        for key, (color, ls, lw, label) in STYLE.items():
            if key not in layer_data:
                continue
            eigs = [max(e, 1e-20) for e in layer_data[key]["eigenvalues"]]
            ax.semilogy(range(1, len(eigs) + 1), eigs, color=color, linestyle=ls, linewidth=lw, label=label)
        ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.set_xlabel("Eigenvalue index")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Eigenvalue of (1/n)·K  [log scale]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.18), fontsize=9)
    fig.suptitle("Empirical kernel spectrum: eigenvalues of (1/n)·K\nSoftmax vs FAVOR+", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "kernel_spectrum_eigenvalues.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # Per-layer individual plots (eigenvalues)
    for layer_idx in TARGET_LAYERS:
        fig, ax = plt.subplots(figsize=(6, 4))
        layer_data = results[str(layer_idx)]
        for key, (color, ls, lw, label) in STYLE.items():
            if key not in layer_data:
                continue
            eigs = [max(e, 1e-20) for e in layer_data[key]["eigenvalues"]]
            ax.semilogy(range(1, len(eigs) + 1), eigs, color=color, linestyle=ls, linewidth=lw, label=label)
        ax.set_title(f"Kernel spectrum — Layer {layer_idx}", fontsize=12)
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("Eigenvalue of (1/n)·K  [log scale]")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"kernel_spectrum_layer{layer_idx}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")

    # Singular values combined plot
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4), sharey=False)
    for ax, layer_idx in zip(axes, TARGET_LAYERS):
        layer_data = results[str(layer_idx)]
        for key, (color, ls, lw, label) in STYLE.items():
            if key not in layer_data:
                continue
            svs = [max(s, 1e-20) for s in layer_data[key]["singular_values"]]
            ax.semilogy(range(1, len(svs) + 1), svs, color=color, linestyle=ls, linewidth=lw, label=label)
        ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.set_xlabel("Singular value index")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Singular value of (1/n)·K  [log scale]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.18), fontsize=9)
    fig.suptitle("Singular value spectrum of (1/n)·K\nSoftmax vs FAVOR+", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "kernel_spectrum_singular_values.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # Effective rank bar chart
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
    for ax, layer_idx in zip(axes, TARGET_LAYERS):
        layer_data = results[str(layer_idx)]
        keys   = [k for k in STYLE if k in layer_data]
        reffs  = [layer_data[k]["effective_rank"] for k in keys]
        colors = [STYLE[k][0] for k in keys]
        labels = [STYLE[k][3] for k in keys]
        bars = ax.bar(range(len(keys)), reffs, color=colors)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
        ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.set_ylabel("Effective rank")
    fig.suptitle("Effective rank by layer and model", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "kernel_spectrum_effective_rank.png")
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

    results   = {str(l): {} for l in TARGET_LAYERS}
    K_softmax = {str(l): None for l in TARGET_LAYERS}  # store raw Gram for cross-model distance

    # ── 1. Softmax teacher ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("[1/6] Softmax teacher")
    print(SEP)
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    teacher.eval()
    layer_res = {}
    for layer_idx in TARGET_LAYERS:
        print(f"  layer {layer_idx}...", end=" ", flush=True)
        metrics, K_avg = extract_all_metrics(teacher, tokenizer, layer_idx, mode="softmax")
        results[str(layer_idx)]["softmax"] = metrics
        K_softmax[str(layer_idx)] = K_avg
        layer_res[str(layer_idx)] = metrics
        print(f"λ₁={metrics['eigenvalues'][0]:.4f}  r_eff={metrics['effective_rank']:.2f}  asym={metrics['asymmetry_rel']*100:.2f}%")
    del teacher; torch.cuda.empty_cache()

    # ── 2. FAVOR+ no finetuning ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print("[2/6] FAVOR+ no finetuning (4 heads)")
    print(SEP)
    base_noft = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    base_noft = patch_model(base_noft, num_performer_heads=4)
    base_noft.eval()
    for layer_idx in TARGET_LAYERS:
        print(f"  layer {layer_idx}...", end=" ", flush=True)
        metrics, K_avg = extract_all_metrics(base_noft, tokenizer, layer_idx, mode="favor", n_performer_heads=4)
        fro_dist = float(np.linalg.norm(K_softmax[str(layer_idx)] - K_avg, 'fro'))
        metrics["frobenius_dist_to_softmax"] = fro_dist
        results[str(layer_idx)]["no_finetune"] = metrics
        print(f"λ₁={metrics['eigenvalues'][0]:.4e}  r_eff={metrics['effective_rank']:.2f}  ||Ks-Kf||_F={fro_dist:.4f}")
    del base_noft; torch.cuda.empty_cache()

    # ── 3-6. Finetuned phases ─────────────────────────────────────────────────
    for i, (result_key, fname) in enumerate(ckpt_files.items(), start=3):
        ckpt_path = os.path.join(args.ckpt_dir, fname)
        if not os.path.exists(ckpt_path):
            print(f"\n  [SKIP] {fname} not found")
            continue
        n_heads_label = result_key.split("_")[1]
        print(f"\n{SEP}")
        print(f"[{i}/6] {result_key}  ({n_heads_label} performer heads)")
        print(SEP)
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
        model, n_heads, _ = load_checkpoint(ckpt_path, base)
        for layer_idx in TARGET_LAYERS:
            print(f"  layer {layer_idx}...", end=" ", flush=True)
            metrics, K_avg = extract_all_metrics(model, tokenizer, layer_idx, mode="favor", n_performer_heads=n_heads)
            fro_dist = float(np.linalg.norm(K_softmax[str(layer_idx)] - K_avg, 'fro'))
            metrics["frobenius_dist_to_softmax"] = fro_dist
            results[str(layer_idx)][result_key] = metrics
            print(f"λ₁={metrics['eigenvalues'][0]:.4e}  r_eff={metrics['effective_rank']:.2f}  ||Ks-Kf||_F={fro_dist:.4f}  asym={metrics['asymmetry_rel']*100:.2f}%")
        del model; torch.cuda.empty_cache()

    # ── Summary tables ────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("SUMMARY — Softmax")
    print(SEP)
    print_model_summary("Softmax teacher", {str(l): results[str(l)] for l in TARGET_LAYERS})

    for key in ["no_finetune", "phase1_4heads", "phase2_8heads", "phase3_16heads", "phase4_32heads"]:
        if any(key in results[str(l)] for l in TARGET_LAYERS):
            print(f"\n{SEP}")
            print(f"SUMMARY — {key}")
            print(SEP)
            print_model_summary(key, {str(l): results[str(l)] for l in TARGET_LAYERS})

    # ── Hoffmann-Wielandt table ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print_hw_table(results)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{SEP}")
    print(f"Results saved to {args.out}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\nGenerating plots → {args.plots}")
    make_plots(results, args.plots)
    print("\nDone.")


if __name__ == "__main__":
    main()
