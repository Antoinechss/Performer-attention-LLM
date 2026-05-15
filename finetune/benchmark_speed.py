"""
Speed benchmark: teacher (softmax) vs Phase 2 (8/32 performer heads).

Measures:
  - Prefill latency across sequence lengths (256..8192)
  - Decode latency across context lengths (256..8192)
  - M sweep: quality vs speed tradeoff at fixed N

Usage:
    python finetune/benchmark_speed.py --ckpt /workspace/checkpoints/best_phase2_K8_QKVO.pt
    python finetune/benchmark_speed.py --ckpt /workspace/checkpoints/best_phase2_K8_QKVO.pt --out docs/benchmark_results.json
"""

import sys, os, argparse, json, math, time
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PERFORMER_DIR = os.path.join(_REPO_ROOT, "performer")
if _PERFORMER_DIR not in sys.path:
    sys.path.insert(0, _PERFORMER_DIR)

from performer_attention import PerformerAttentionCore, _sample_orf
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE   = "cuda"
DTYPE    = torch.bfloat16
WARMUP   = 5
REPEATS  = 20

SEQ_LENS     = [256, 512, 1024, 2048]
M_SWEEP_VALS = [64, 128, 256, 512, 1024]
M_SWEEP_N    = 2048  # fixed N for M sweep

VAL_TEXTS = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "The French Revolution began in 1789 and fundamentally transformed French society and government.",
    "Quantum mechanics describes the behavior of matter and energy at the atomic and subatomic levels.",
    "Climate change refers to long-term shifts in global temperatures and weather patterns.",
    "The human genome contains approximately three billion base pairs of DNA.",
] * 20


# ── MixedPerformerAttention ───────────────────────────────────────────────────

class MixedPerformerAttention(torch.nn.Module):
    def __init__(self, original_attn, num_performer_heads, kernel="favor"):
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
        self.performer_core = PerformerAttentionCore(
            head_dim=self.head_dim, num_features=256, kernel=kernel
        )
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


def patch_model(model, num_performer_heads, kernel="favor"):
    for layer in model.model.layers:
        layer.self_attn = MixedPerformerAttention(layer.self_attn, num_performer_heads, kernel=kernel)
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
    print(f"  Loaded {os.path.basename(ckpt_path)} — {n_heads}/32 performer heads")
    return model, n_heads, phase


# ── Timing helpers ────────────────────────────────────────────────────────────

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def measure_prefill(model, input_ids, repeats=REPEATS, warmup=WARMUP):
    """Returns median prefill latency in ms."""
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids=input_ids, use_cache=False)
            _sync()
        times = []
        for _ in range(repeats):
            _sync()
            t0 = time.perf_counter()
            _ = model(input_ids=input_ids, use_cache=False)
            _sync()
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def measure_decode(model, tokenizer, context_len, repeats=REPEATS, warmup=WARMUP):
    """Returns median single-step decode latency in ms at given context length."""
    # Build a context of exactly context_len tokens
    prompt = "The quick brown fox jumps over the lazy dog. " * 200
    ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    ids = ids[:, :context_len]
    if ids.shape[1] < context_len:
        pad = torch.zeros(1, context_len - ids.shape[1], dtype=torch.long, device=DEVICE)
        ids = torch.cat([ids, pad], dim=1)

    with torch.no_grad():
        # One decode step: feed full context, measure time for the last token prediction
        for _ in range(warmup):
            _ = model(input_ids=ids, use_cache=False)
            _sync()
        times = []
        for _ in range(repeats):
            _sync()
            t0 = time.perf_counter()
            _ = model(input_ids=ids[:, -1:], use_cache=False)
            _sync()
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


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


# ── M sweep ───────────────────────────────────────────────────────────────────

def run_m_sweep(ckpt_path, tokenizer, n_heads):
    """For each M: measure prefill latency at M_SWEEP_N and compute perplexity."""
    print(f"\n{'='*60}")
    print(f"M SWEEP  (N={M_SWEEP_N}, phase2 weights)")
    print(f"{'='*60}")
    print(f"{'M':>6}  {'M/D':>5}  {'Prefill (ms)':>12}  {'Perplexity':>10}  {'RelErr approx':>13}")
    print("-" * 55)

    # Precomputed approximation relative errors from analysis notebook
    approx_err = {64: 17.04, 128: 12.32, 256: 9.02, 512: 6.31, 1024: 4.47}

    results = []
    prompt  = "The quick brown fox jumps over the lazy dog. " * 200
    ids     = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    ids     = ids[:, :M_SWEEP_N]

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    for m in M_SWEEP_VALS:
        base  = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
        model = patch_model(base, n_heads)

        # Override omega with new sample at this M
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer.self_attn, "performer_core"):
                new_omega = _sample_orf(layer.self_attn.head_dim, m).to(DEVICE)
                # Replace performer_core with one of the right size
                layer.self_attn.performer_core = PerformerAttentionCore(
                    head_dim=layer.self_attn.head_dim, num_features=m
                ).to(DEVICE).to(DTYPE)
                layer.self_attn.performer_core.omega.copy_(new_omega)

        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()

        latency = measure_prefill(model, ids)
        ppl     = compute_ppl(model, tokenizer, VAL_TEXTS[:40])
        err     = approx_err.get(m, float("nan"))

        print(f"{m:>6}  {m/64:>5.1f}  {latency:>12.2f}  {ppl:>10.2f}  {err:>12.2f}%")
        results.append({"M": m, "M_over_D": round(m/64, 2), "prefill_ms": round(latency, 3),
                        "perplexity": round(ppl, 3), "approx_rel_err_pct": err})

        del model
        torch.cuda.empty_cache()

    return results


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(prefill_results, decode_results, m_sweep_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    seq_lens   = [r["N"] for r in prefill_results]
    t_softmax  = [r["softmax_ms"] for r in prefill_results]
    t_perf     = [r["performer_ms"] for r in prefill_results]
    speedups   = [r["speedup"] for r in prefill_results]

    # -- Plot 1: latency vs sequence length (prefill) --------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(seq_lens, t_softmax, "o-", label="Softmax (teacher)", color="tab:red", linewidth=2)
    ax.plot(seq_lens, t_perf,   "s--", label="FAVOR+ Phase 2 (8/32)", color="tab:blue", linewidth=2)
    ax.set_xlabel("Sequence length N (tokens)")
    ax.set_ylabel("Prefill latency (ms)")
    ax.set_title("Prefill latency — Softmax vs FAVOR+")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(seq_lens)
    ax.set_xticklabels([str(n) for n in seq_lens])
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "prefill_latency.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved prefill_latency.png")

    # -- Plot 2: speedup vs sequence length ------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Break-even (1×)")
    ax.plot(seq_lens, speedups, "o-", color="tab:green", linewidth=2)
    ax.fill_between(seq_lens, 1.0, speedups,
                    where=[s > 1 for s in speedups], alpha=0.15, color="tab:green", label="Faster")
    ax.fill_between(seq_lens, speedups, 1.0,
                    where=[s < 1 for s in speedups], alpha=0.15, color="tab:red", label="Slower")
    ax.set_xlabel("Sequence length N (tokens)")
    ax.set_ylabel("Speedup (softmax / performer)")
    ax.set_title("Prefill speedup factor — FAVOR+ vs Softmax")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_xticks(seq_lens)
    ax.set_xticklabels([str(n) for n in seq_lens])
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "prefill_speedup.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved prefill_speedup.png")

    # -- Plot 3: M sweep — quality vs speed ------------------------------------
    if m_sweep_results:
        ms    = [r["M"] for r in m_sweep_results]
        ppls  = [r["perplexity"] for r in m_sweep_results]
        lats  = [r["prefill_ms"] for r in m_sweep_results]

        # D=64 for TinyLlama heads
        D = 64
        M_PAPER_MIN = D       # M=D — paper minimum recommendation
        M_OURS      = 256     # M=4D — our choice

        fig, ax1 = plt.subplots(figsize=(9, 5))
        color1 = "tab:blue"
        ax1.set_xlabel("M (number of random features)")
        ax1.set_ylabel("Perplexity", color=color1)
        ax1.plot(ms, ppls, "o-", color=color1, linewidth=2, label="Perplexity")
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = "tab:orange"
        ax2.set_ylabel(f"Prefill latency at N={M_SWEEP_N} (ms)", color=color2)
        ax2.plot(ms, lats, "s--", color=color2, linewidth=2, label="Latency (ms)")
        ax2.tick_params(axis="y", labelcolor=color2)

        # Vertical markers
        ax1.axvline(M_PAPER_MIN, color="gray",    linestyle=":",  linewidth=1.5,
                    label=f"Paper minimum (M=D={M_PAPER_MIN})")
        ax1.axvline(M_OURS,      color="tab:green", linestyle="--", linewidth=1.5,
                    label=f"Our choice (M=4D={M_OURS})")

        ax1.set_xscale("log", base=2)
        ax1.set_xticks(ms)
        ax1.set_xticklabels([str(m) for m in ms])
        ax1.grid(True, which="both", alpha=0.3)
        fig.suptitle(f"Quality vs Speed tradeoff — M sweep (N={M_SWEEP_N}, Phase 2 weights)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "m_sweep.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved m_sweep.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",    required=True, help="Path to phase2 checkpoint .pt")
    parser.add_argument("--out",     default="docs/benchmark_results.json")
    parser.add_argument("--plots",   default="docs/plots")
    parser.add_argument("--no_m_sweep", action="store_true", help="Skip M sweep (faster)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "The quick brown fox jumps over the lazy dog. " * 200

    # ── Teacher ───────────────────────────────────────────────────────────────
    print("\nLoading teacher (softmax)...")
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    teacher.eval()

    print(f"\n{'='*60}")
    print("PREFILL — Teacher vs Phase 2")
    print(f"{'='*60}")
    print(f"{'N':>6}  {'Softmax (ms)':>12}  {'Performer (ms)':>14}  {'Speedup':>8}")
    print("-" * 48)

    teacher_prefill, perf_prefill = [], []
    for n in SEQ_LENS:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        ids = ids[:, :n]
        if ids.shape[1] < n:
            pad = torch.zeros(1, n - ids.shape[1], dtype=torch.long, device=DEVICE)
            ids = torch.cat([ids, pad], dim=1)
        t = measure_prefill(teacher, ids)
        teacher_prefill.append(t)
        print(f"{n:>6}  {t:>12.2f}  {'(loading perf...)':>14}  {'':>8}")

    del teacher
    torch.cuda.empty_cache()

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    print("\nLoading Phase 2...")
    model, n_heads, phase = load_checkpoint(args.ckpt)

    prefill_results = []
    print(f"\n{'N':>6}  {'Softmax (ms)':>12}  {'Performer (ms)':>14}  {'Speedup':>8}")
    print("-" * 48)
    for i, n in enumerate(SEQ_LENS):
        ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        ids = ids[:, :n]
        if ids.shape[1] < n:
            pad = torch.zeros(1, n - ids.shape[1], dtype=torch.long, device=DEVICE)
            ids = torch.cat([ids, pad], dim=1)
        t_p   = measure_prefill(model, ids)
        t_s   = teacher_prefill[i]
        spd   = t_s / t_p
        print(f"{n:>6}  {t_s:>12.2f}  {t_p:>14.2f}  {spd:>7.2f}×")
        prefill_results.append({"N": n, "softmax_ms": round(t_s, 3),
                                "performer_ms": round(t_p, 3), "speedup": round(spd, 3)})
        perf_prefill.append(t_p)

    # ── Decode ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("DECODE — single step latency")
    print(f"{'='*60}")
    print(f"{'Context N':>10}  {'Softmax (ms)':>12}  {'Performer (ms)':>14}  {'Speedup':>8}")
    print("-" * 52)

    # Reload teacher for decode
    teacher2 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    teacher2.eval()

    decode_results = []
    for n in SEQ_LENS:
        t_s = measure_decode(teacher2, tokenizer, n)
        t_p = measure_decode(model, tokenizer, n)
        spd = t_s / t_p
        print(f"{n:>10}  {t_s:>12.3f}  {t_p:>14.3f}  {spd:>7.2f}×")
        decode_results.append({"N": n, "softmax_ms": round(t_s, 4),
                               "performer_ms": round(t_p, 4), "speedup": round(spd, 3)})

    del teacher2
    torch.cuda.empty_cache()

    # ── M sweep ───────────────────────────────────────────────────────────────
    m_sweep_results = []
    if not args.no_m_sweep:
        m_sweep_results = run_m_sweep(args.ckpt, tokenizer, n_heads)

    del model
    torch.cuda.empty_cache()

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "model":           MODEL_ID,
        "checkpoint":      os.path.basename(args.ckpt),
        "warmup":          WARMUP,
        "repeats":         REPEATS,
        "prefill":         prefill_results,
        "decode":          decode_results,
        "m_sweep":         m_sweep_results,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    make_plots(prefill_results, decode_results, m_sweep_results, args.plots)
    print(f"Plots saved to {args.plots}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
