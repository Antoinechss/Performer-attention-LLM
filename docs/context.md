# Project Context

M2 research project (CERMICS / Advansight). Goal: reduce LLM attention computation from O(N²) to O(N) by replacing a fraction of softmax attention heads in TinyLlama 1.1B with FAVOR+ (Performer) linear attention, then recovering output quality via knowledge distillation fine-tuning.

---

## Repository Structure

```
performer/
  performer_attention.py   # Core FAVOR+ implementation: PerformerAttentionCore, PerformerAttention
  triton_scan.py           # Triton CUDA kernels for prefill scan and decode step
models/
  analysis.py              # Standalone analysis script (Sections A/B/C, no notebooks needed)
  performer_attention.py   # (same source used via sys.path in notebooks)
analysis.ipynb             # Main analysis notebook (Colab): quality + speed benchmarks
finetune.ipynb             # Knowledge distillation training notebook (Colab)
Notebook - spectre approximations (2).ipynb  # Kernel spectrum theory notebook
docs/
  context.md               # This file
  theory/                  # PDF paper summaries (see below)
```

---

## What Is Done

### Core implementation
- `PerformerAttentionCore` (`performer/performer_attention.py`): standalone FAVOR+ attention module, no Q/K/V projections, plugs into any architecture. Takes pre-projected Q, K, V tensors.
- `PerformerAttention`: full module with projections, for isolated testing.
- `_sample_orf`: orthogonal random feature sampling (FAVOR+ spec, chi(d) norm scaling).
- `_phi`: FAVOR+ positive feature map with numerical stability (per-query max stabilizer for queries, global max for keys).
- `_python_scan`: causal sequential scan fallback (CPU/MPS, supports autograd for training).
- `triton_scan.py`: Triton CUDA kernels for (1) prefill causal scan and (2) fused decode step. Auto-detected and used when CUDA + Triton available.
- `MixedPerformerAttention` (in both notebooks and `models/analysis.py`): wraps HuggingFace `LlamaAttention`, routes the first `num_performer_heads` heads through FAVOR+ and the rest through standard scaled dot-product attention. Handles GQA (TinyLlama uses 4 KV heads / 32 query heads), RoPE application, KV-cache compatibility.

### Analysis notebook (`analysis.ipynb`, Colab)
- Section 0: setup, correctness checkpoint (0/32 performer heads must equal standard model exactly).
- Section 1.1: layerwise replacement — first n layers fully replaced; shows error accumulates rapidly across layers.
- Section 1.2: per-token generation comparison (4/32 heads replaced), token-level KL divergence and probability tracking.
- Section 2: prefill scaling benchmarks (O(N²) softmax vs O(N·M) performer) and decode step benchmarks. Sequence lengths 256–4096.
- Section C: KL divergence and top-5 overlap sweep across 0–32 performer heads.

### Fine-tuning notebook (`finetune.ipynb`, Colab)
- Knowledge distillation: frozen teacher (softmax TinyLlama) guides student (performer-patched TinyLlama).
- Loss: `alpha * CE_loss + (1-alpha) * KL_distillation_loss`.
- Dataset: WikiText-103, fixed-length pre-tokenized chunks.
- 4-phase curriculum: 4 heads → 8 heads → 16 heads → 32 heads, progressively unfreezing Q/K then Q/K/V/O.
- Checkpoint save/resume support for Colab disconnects. Optional Google Drive export.

### Kernel spectrum notebook (`Notebook - spectre approximations (2).ipynb`)
- Numerical study of the eigenvalue spectrum of the softmax kernel and its three approximations (trigonometric, positive PRF, hyperbolic PRF).
- Studies effect of normalization conventions (1/√d, 1/d^0.25) on spectrum alignment between approximations and exact kernel.
- Variance analysis across multiple omega draws.

---

## Key Design Decisions

- `num_features=256` random features per head (M). Trade-off between approximation quality and speed.
- First K heads (indices 0..K-1) are replaced with performer; remaining heads stay softmax. This is a configurable parameter, not a learned selection.
- During prefill (N > 1 tokens), causal scan is used (sequential O(N·M·D) ops). During decode (single new token), accumulated KV state allows O(M·D) step.
- `omega` buffer is registered as `persistent=True` to survive model save/load.
- Triton kernels are disabled during training (Python scan supports autograd; Triton scan does not).
- Causal mask is built manually in `MixedPerformerAttention` because SDPA's `is_causal=True` is bypassed in the mixed path.

---

## Next Steps

1. **Fine-tuning execution**: run the 4-phase distillation in `finetune.ipynb` on Colab (GPU required). Target: recover perplexity close to baseline after full (32/32) head replacement.

2. **Theory-code link**: compare empirical eigenvalue spectra of the attention kernel matrix (from the codebase) against the theoretical spectral decay predicted by the integral operator analysis. The `Notebook - spectre approximations (2).ipynb` sets up the numerical tools; the next step is to run this comparison on actual TinyLlama Q/K activations.

3. **Industry cost analysis**: model and compare inference costs (FLOPs, memory bandwidth, latency) for softmax vs. linear attention at production scale (long contexts, large batch sizes). Use benchmark data from Section B of `analysis.ipynb` as input.

4. **Potential implementation extensions** (theory-driven, see paper summaries in `docs/theory/`):
   - Hyperbolic feature map (`phi_hyp`) as a lower-variance alternative to the current `phi_pos` used in FAVOR+.
   - Orthogonal random features with block-diagonal Hadamard structure (Fastfood / SORF) for O(d log d) omega sampling instead of O(d²).
   - Extend kernel approximation beyond softmax (e.g., Gaussian kernel attention variants).

---

## Theory Papers

Summaries in `docs/theory/`:
- [performers.md](theory/performers.md) — FAVOR+ algorithm, positive random features, unbiasedness guarantees
- [kernel_approximation.md](theory/kernel_approximation.md) — Bochner's theorem, RFF, PRF, structured methods
- [integral_operator.md](theory/integral_operator.md) — Spectral theory for kernel operators, Hoffman-Wielandt inequality
