"""
Three-section comparison of standard vs performer attention.

  A — Per-token live generation: quality side-by-side + probability alignment
  B — Prefill speed scaling: O(N²) vs O(N·M) across sequence lengths
  C — Mixed-head quality sweep: KL / top-5 overlap as num_performer_heads grows
"""
import sys
import os
import importlib.util
import time
import torch
import torch.nn.functional as F

# ── Import order: venv transformers FIRST, then local path ───────────────────
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.activations   # cache in sys.modules before local path shadows it

_base = os.path.join(os.path.dirname(__file__), '..', 'transformers', 'src')
sys.path.insert(0, _base)
import transformers.models
import transformers.models.llama


def _load_performer_module():
    path = os.path.join(_base, 'transformers', 'models', 'llama', 'modeling_llama_performer.py')
    name = 'transformers.models.llama.modeling_llama_performer'
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_perf_mod = _load_performer_module()
PerformerLlamaForCausalLM = _perf_mod.LlamaForCausalLM

# ── Config ───────────────────────────────────────────────────────────────────
MODEL          = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT         = "<|user|>\nHow do I get a good night's sleep?</s>\n<|assistant|>\n"
MAX_NEW_TOKENS = 20
DTYPE          = torch.float32

RUN_A = False   # per-token live generation (slow — needs full model, token by token)
RUN_B = True    # attention kernel speed benchmark (fast — no model load needed)
RUN_C = False    # mixed-head quality sweep (moderate — one forward pass per K value)
# ─────────────────────────────────────────────────────────────────────────────

# Models only needed for sections A and C
if RUN_A or RUN_C:
    print("Loading standard model...")
    std_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="cpu")
    std_model.eval()

    # Change SECTION_A_PERFORMER_HEADS to try different K values in the live comparison
    # 32 = all performer (worst quality) | 0 = pure softmax | 1,2,4... = mixed
    SECTION_A_PERFORMER_HEADS = 4

    print("Loading performer model (all heads, will patch for Section A)...")
    perf_model = PerformerLlamaForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="cpu")
    perf_model.eval()

    tokenizer  = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    num_heads  = std_model.config.num_attention_heads  # 32 for TinyLlama

# ════════════════════════════════════════════════════════════════════════════
# SECTION A — per-token live comparison
# ════════════════════════════════════════════════════════════════════════════
if RUN_A:
    for layer in perf_model.model.layers:
        layer.self_attn.num_performer_heads = SECTION_A_PERFORMER_HEADS
        layer.self_attn.num_standard_heads  = num_heads - SECTION_A_PERFORMER_HEADS

    print(f"\n{'═'*70}")
    print(f"SECTION A — Per-token generation  [{SECTION_A_PERFORMER_HEADS}/{num_heads} performer heads]")
    print(f"{'═'*70}\n")

    W = 14
    header = (f"{'Step':>4}  {'Classic':.<{W}}  {'Performer':.<{W}}"
              f"  {'Classic p':>9}  {'Perf p(classic)':>15}  {'KL':>7}")
    print(header)
    print("─" * len(header))

    current_ids   = prompt_ids.clone()
    classic_tokens, perf_tokens = [], []
    kl_per_step, perf_p_track  = [], []

    with torch.no_grad():
        for step in range(1, MAX_NEW_TOKENS + 1):
            std_out  = std_model(input_ids=current_ids,  use_cache=False)
            perf_out = perf_model(input_ids=current_ids, use_cache=False)

            std_logits  = std_out.logits[0, -1].float()
            perf_logits = perf_out.logits[0, -1].float()

            std_probs  = F.softmax(std_logits,  dim=-1)
            perf_probs = F.softmax(perf_logits, dim=-1)

            classic_id = std_logits.argmax().item()
            perf_id    = perf_logits.argmax().item()

            classic_p  = std_probs[classic_id].item()
            perf_p_cls = perf_probs[classic_id].item()
            kl         = F.kl_div(perf_probs.log(), std_probs, reduction='sum').item()

            c_tok = repr(tokenizer.decode([classic_id]))[1:-1]
            p_tok = repr(tokenizer.decode([perf_id]))[1:-1]

            print(f"{step:>4}  {c_tok:<{W}}  {p_tok:<{W}}"
                  f"  {classic_p:>8.2%}  {perf_p_cls:>15.2%}  {kl:>7.3f}")

            classic_tokens.append(classic_id)
            perf_tokens.append(perf_id)
            kl_per_step.append(kl)
            perf_p_track.append(perf_p_cls)

            current_ids = torch.cat([current_ids, torch.tensor([[classic_id]])], dim=-1)
            if classic_id == tokenizer.eos_token_id:
                break

    n = len(classic_tokens)
    print(f"\n  Classic:   {tokenizer.decode(classic_tokens, skip_special_tokens=True)}")
    print(f"  Performer: {tokenizer.decode(perf_tokens,    skip_special_tokens=True)}")
    match = sum(c == p for c, p in zip(classic_tokens, perf_tokens))
    print(f"\n  Token match: {match}/{n}  |  Avg KL: {sum(kl_per_step)/n:.4f}"
          f"  |  Avg perf p(classic): {sum(perf_p_track)/n:.2%}")

# ════════════════════════════════════════════════════════════════════════════
# SECTION B — Attention-kernel-only speed scaling
#
# Why isolate the kernel?
#   Full model forward is dominated by MLP + LayerNorm at small N.
#   To see O(N²) vs O(N·M) we must time ONLY the attention operation.
#
# Two sub-benchmarks:
#   B1 — Prefill: process N tokens at once  (performer: O(N·M), std: O(N²))
#   B2 — Decoding step: one new token, growing KV cache
#          Standard:           O(N·D) per step  (grows with history)
#          Performer (naive):  O(N·M·D) per step (recomputes kv_sum — SLOW)
#          Performer (state):  O(M·D) per step   (incremental update — FAST)
# ════════════════════════════════════════════════════════════════════════════
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'performer'))
from performer_attention import PerformerAttentionCore, _HAS_TRITON

try:
    from triton_scan import triton_scan_forward as _triton_scan_raw
except ImportError:
    _triton_scan_raw = None

_CUDA = torch.cuda.is_available()
_TRITON_BENCH = _HAS_TRITON and _CUDA   # True only on a CUDA machine with triton installed

H, D, M = 32, 64, 256   # heads, head_dim, num_features
REPEATS  = 3
performer_core = PerformerAttentionCore(head_dim=D, num_features=M)

def time_fn(fn, repeats=REPEATS):
    # one warmup, then average
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats * 1000

# ── B1: Prefill scaling ───────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("SECTION B1 — Prefill attention kernel  O(N²·D) vs O(N·M·D)")
print(f"             Crossover at N = M = {M}  |  H={H}, D={D}, M={M}")
if _TRITON_BENCH:
    print("             Triton kernel: ACTIVE  (single fused GPU launch)")
else:
    print("             Triton kernel: INACTIVE  (requires CUDA GPU + pip install triton)")
    print("             Performer column uses Python scan loop (CPU fallback)")
print(f"{'═'*70}\n")

SEQ_LENS = [64, 128, 256, 512]

if _TRITON_BENCH:
    hdr = f"{'N':>6}  {'Std (ms)':>10}  {'Py-scan (ms)':>13}  {'Triton (ms)':>12}  {'Speedup vs std':>15}  {'Theory':>7}"
    print(hdr)
    print("─" * len(hdr))
else:
    print(f"{'N':>6}  {'Std attn (ms)':>14}  {'Perf attn (ms)':>15}  {'Speedup':>8}  {'Theory ratio':>13}")
    print("─" * 63)

scale = D ** -0.25
_dev = torch.device("cuda" if _CUDA else "cpu")

with torch.no_grad():
    for N in SEQ_LENS:
        q = torch.randn(1, H, N, D, device=_dev)
        k = torch.randn(1, H, N, D, device=_dev)
        v = torch.randn(1, H, N, D, device=_dev)

        # Standard: scaled dot-product attention
        def std_attn():
            scores = torch.matmul(q, k.transpose(-2, -1)) * (D ** -0.5)
            w = torch.softmax(scores, dim=-1)
            return torch.matmul(w, v)

        # Performer: FAVOR+ kernel (Triton on CUDA, Python scan on CPU)
        def perf_attn():
            return performer_core(q * scale, k * scale, v)

        std_ms  = time_fn(std_attn)
        perf_ms = time_fn(perf_attn)

        if _TRITON_BENCH and _triton_scan_raw is not None:
            # Also time the raw Triton kernel (scan only, phi pre-computed)
            phi_q_b = performer_core.phi(q * scale).float()
            phi_k_b = performer_core.phi(k * scale).float()
            v_b     = v.float()
            def triton_only():
                return _triton_scan_raw(phi_q_b, phi_k_b, v_b)
            triton_ms = time_fn(triton_only)
            speedup   = std_ms / triton_ms
            theory    = N / M
            print(f"{N:>6}  {std_ms:>10.2f}  {perf_ms:>13.2f}  {triton_ms:>12.2f}  {speedup:>14.2f}x  {theory:>6.2f}x")
        else:
            speedup = std_ms / perf_ms
            theory  = N / M
            print(f"{N:>6}  {std_ms:>13.2f}  {perf_ms:>14.2f}  {speedup:>7.2f}x  {theory:>12.2f}x")

# ── B2: Decoding step scaling ─────────────────────────────────────────────
print(f"\n{'═'*70}")
print("SECTION B2 — Decoding step  (one new token, growing KV cache)")
print(f"             Standard O(N·D) | Performer naive O(N·M·D) | Performer state O(M·D)")
print(f"{'═'*70}\n")

CACHE_SIZES = [64, 128, 256, 512, 1024]

print(f"{'Cache N':>8}  {'Std (ms)':>10}  {'Perf naive (ms)':>16}  {'Perf state (ms)':>16}  {'State speedup':>14}")
print("─" * 72)

with torch.no_grad():
    for N in CACHE_SIZES:
        q_new = torch.randn(1, H, 1, D)        # single new query token
        k_all = torch.randn(1, H, N, D)        # full cached keys
        v_all = torch.randn(1, H, N, D)        # full cached values

        # Standard: new query attends to all N cached keys
        def std_decode():
            scores = torch.matmul(q_new, k_all.transpose(-2, -1)) * (D ** -0.5)
            w = torch.softmax(scores, dim=-1)
            return torch.matmul(w, v_all)

        # Performer naive: recomputes kv_sum from full cache each step
        def perf_naive():
            return performer_core(q_new * scale, k_all * scale, v_all)

        # Performer with incremental state: O(M·D) per step
        # Simulates maintaining running sums updated one token at a time
        phi_k_all = performer_core.phi(k_all * scale)                    # [1,H,N,M]
        kv_state  = torch.einsum("bhnm,bhnd->bhmd", phi_k_all, v_all)   # [1,H,M,D]
        k_state   = phi_k_all.sum(dim=2)                                  # [1,H,M]

        def perf_state():
            phi_q = performer_core.phi(q_new * scale)                    # [1,H,1,M]
            out   = torch.einsum("bhnm,bhmd->bhnd", phi_q, kv_state)    # [1,H,1,D]
            denom = torch.einsum("bhnm,bhm->bhn",   phi_q, k_state) + 1e-6
            return out / denom.unsqueeze(-1)

        std_ms        = time_fn(std_decode)
        perf_naive_ms = time_fn(perf_naive)
        perf_state_ms = time_fn(perf_state)
        state_speedup = std_ms / perf_state_ms

        print(f"{N:>8}  {std_ms:>9.3f}  {perf_naive_ms:>15.3f}  {perf_state_ms:>15.3f}  {state_speedup:>13.2f}x")

# ════════════════════════════════════════════════════════════════════════════
# SECTION C — Mixed-head quality sweep
# Load model once, change num_performer_heads at runtime (no weight reload)
# ════════════════════════════════════════════════════════════════════════════
if RUN_C:
    print(f"\n{'═'*70}")
    print("SECTION C — Mixed-head sweep  [quality as performer heads increase]")
    print(f"{'═'*70}\n")

    with torch.no_grad():
        std_ref = std_model(input_ids=prompt_ids, use_cache=False)
    std_logits_ref = std_ref.logits[0, -1].float()
    std_probs_ref  = F.softmax(std_logits_ref, dim=-1)
    std_top5_ref   = set(std_logits_ref.topk(5).indices.tolist())

    SWEEP = [0, 1, 2, 4, 8, 16, 32]

    print(f"{'K heads':>8}  {'KL div':>8}  {'Top-5 overlap':>14}  {'Perf p(std top1)':>17}  {'Note'}")
    print("─" * 65)

    for k in SWEEP:
        for layer in perf_model.model.layers:
            layer.self_attn.num_performer_heads = k
            layer.self_attn.num_standard_heads  = num_heads - k

        with torch.no_grad():
            out = perf_model(input_ids=prompt_ids, use_cache=False)

        logits  = out.logits[0, -1].float()
        probs   = F.softmax(logits, dim=-1)
        kl      = F.kl_div(probs.log(), std_probs_ref, reduction='sum').item()
        top5    = set(logits.topk(5).indices.tolist())
        overlap = len(top5 & std_top5_ref)
        p_top1  = probs[std_logits_ref.argmax().item()].item()

        note = ""
        if k == 0:       note = "← pure softmax (baseline)"
        elif k == num_heads: note = "← all performer"

        print(f"{k:>8}  {kl:>8.4f}  {overlap:>14}/5  {p_top1:>17.2%}  {note}")
