# Performer Attention LLM

M2 research project — CERMICS / Advansight.

Replacing a fraction of softmax attention heads in TinyLlama 1.1B with FAVOR+ (Performer) linear attention, then recovering output quality via knowledge distillation. The goal is to reduce attention complexity from O(N²) to O(N) while preserving generation quality.

---

## The Triangle

The project evaluates the softmax → FAVOR+ substitution across three independent axes:

```
                        QUALITY
                       (generation)
                           ▲
                          / \
                         /   \
                        /     \
                       /       \
                      /         \
          SPEED ◄────────────────► APPROXIMATION
       (prefill/decode)            (kernel convergence
        latency, FLOPs)             eigenvalues, M/D ratio)
```

Each vertex is a separate concern with its own code and outputs:

- **Quality** — does the model still generate coherent text? How does perplexity and token-level KL divergence degrade as performer heads increase? Does fine-tuning recover it?
- **Speed** — what is the actual latency gain at various sequence lengths? Where does the O(N·M) performer beat O(N²) softmax in practice?
- **Approximation** — how well does FAVOR+ approximate the softmax kernel mathematically? Eigenvalue spectrum comparison, convergence as M/D ratio increases, variance across omega draws.

---

## Results

Fine-tuning via knowledge distillation on WikiText-103 (50k samples, 4-phase curriculum).

| Model | Performer heads | WT103 ppl | C4 ppl |
|-------|----------------|-----------|--------|
| Teacher (softmax) | 0/32 | 11.52 | 11.87 |
| Baseline (no FT) | 4/32 | 14.02 | 14.34 |
| Phase 1 (fine-tuned) | 4/32 | 12.28 | — |
| Phase 2 (fine-tuned) | 8/32 | 12.14 | — |
| Phase 3 (fine-tuned) | 16/32 | 15.01 | — |

**Phase 2 (8/32 performer heads)** is the sweet spot: linear attention on 25% of heads with perplexity within 0.6 points of the softmax teacher, and sub-quadratic prefill scaling.

---

## Checkpoints

Fine-tuned checkpoints are available on HuggingFace: [antoinechss/performer-checkpoints](https://huggingface.co/antoinechss/performer-checkpoints)

| File | Description |
|------|-------------|
| `best_phase1_K4_QK.pt` | 4/32 performer heads, Q/K projections fine-tuned |
| `best_phase2_K8_QKVO.pt` | 8/32 performer heads, Q/K/V/O fine-tuned |
| `best_phase3_K16_QKVO.pt` | 16/32 performer heads |
| `best_phase4_K32_QKVO.pt` | 32/32 performer heads (collapsed — for reference) |

Loading a checkpoint:

```python
from huggingface_hub import hf_hub_download
import torch

path = hf_hub_download("antoinechss/performer-checkpoints", "best_phase2_K8_QKVO.pt")
ckpt = torch.load(path, map_location="cpu", weights_only=False)
```

---

## Repository Structure

```
performer/                        # Shared kernel implementation (imported by everything)
  performer_attention.py          # PerformerAttentionCore, _phi, _sample_orf, _python_scan,
                                  # _python_scan_checkpointed (training path, additive)
  triton_scan.py                  # Triton CUDA kernels (prefill scan + decode step)

analysis/                         # The three triangle vertices
  analysis.ipynb                  # Main notebook (Colab): all three sections in one place
  analysis.py                     # Standalone script version (no Colab needed)
  quality/                        # Vertex 1 — generation quality
  speed/                          # Vertex 2 — prefill/decode benchmarks
  approximation/                  # Vertex 3 — kernel convergence, eigenvalues
    Notebook - spectre approximations (2).ipynb

finetune/                         # Fine-tuning pipeline (run on RunPod, not locally)
  finetune.py                     # Main training script
  eval_post_training.py           # Perplexity, lm-eval, speed, spectral analysis post-FT
  generate.py                     # Generation comparison across checkpoints
  requirements_finetune.txt       # pip dependencies

docs/
  finetune_plan.md                # Fine-tuning decisions, phase table, rationale
  finetune_runpod_setup.md        # RunPod step-by-step setup guide
  theory/                         # PDF summaries
    performers.md
    kernel_approximation.md
    integral_operator.md
```

---

## Quick Start

**Run analysis (Colab or local GPU):**
```bash
pip install torch transformers datasets
python analysis/analysis.py
```

**Generate text with a fine-tuned checkpoint:**
```bash
pip install -r finetune/requirements_finetune.txt
pip install "transformers==4.46.3"
python finetune/generate.py --ckpt_dir /path/to/checkpoints
```

**Re-run fine-tuning on RunPod:**

See [docs/finetune_runpod_setup.md](docs/finetune_runpod_setup.md) for the full guide.

```bash
python finetune/finetune.py
```

---

## Method

**FAVOR+ (Fast Attention Via positive Orthogonal Random features)** approximates the softmax attention kernel using random feature maps:

$$\text{Attn}(Q, K, V) \approx \phi(Q) \left(\phi(K)^T V\right)$$

where $\phi$ is a positive random feature map. This reduces the attention computation from O(N²D) to O(NMD) where M is the number of random features (M=256 here).

**Knowledge distillation:** A frozen softmax teacher guides the performer student via:

$$\mathcal{L} = 0.5 \cdot \mathcal{L}_{CE} + 0.5 \cdot T^2 \cdot \mathcal{L}_{KL}$$

with temperature T=2.0.

**Curriculum:** 4 phases progressively replacing more heads (4→8→16→32) while unfreezing Q/K projections first, then Q/K/V/O.

---

## Implementation

### Core (`performer/`)

- `PerformerAttentionCore`: standalone FAVOR+ attention, no projections, plugs into any architecture. Takes pre-projected Q, K, V tensors [B, H, N, D].
- `PerformerAttention`: full module with projections, for isolated testing.
- `_sample_orf`: orthogonal random feature sampling (chi(d) norm scaling, FAVOR+ spec).
- `_phi`: positive feature map with per-query / global-key numerical stabilizer.
- `_python_scan`: causal sequential scan, O(M×D) memory, CPU/MPS/CUDA fallback.
- `_python_scan_checkpointed`: wraps `_python_scan` with `torch.utils.checkpoint`. Auto-selected inside `PerformerAttentionCore.forward` when `torch.is_grad_enabled()`. Reduces scan activation memory from O(N×M×D) to O(M×D). Inference paths unchanged.
- `triton_scan.py`: Triton CUDA kernels — used only when CUDA + Triton available and grad is disabled (inference only).
- `MixedPerformerAttention` (defined in `analysis.py`, `finetune.py`, `eval_post_training.py` — kept local to each to avoid shared state): wraps HF `LlamaAttention`, routes first K heads through FAVOR+ and remaining through softmax. Handles GQA, RoPE, KV-cache.

### Analysis (`analysis/`)

`analysis.ipynb` / `analysis.py` cover all three triangle vertices in a single run:

- **Quality** (Section 1): correctness checkpoint (0/32 heads = standard model), layerwise degradation sweep, per-token KL divergence and generation comparison at 4/32 heads.
- **Speed** (Section 2): prefill O(N²) vs O(N·M) scaling at N=256–4096, decode step benchmark, speed equivalence threshold (where N·M ≈ N²).
- **Approximation** (Section 3): FAVOR+ convergence as M/D ratio increases, attention weight pattern similarity vs softmax.

`analysis/approximation/Notebook - spectre approximations (2).ipynb`: eigenvalue spectrum of the softmax kernel and three approximations (trigonometric, positive PRF, hyperbolic PRF) on synthetic data. Variance analysis across omega draws. Will be extended to load real activation data from `spectral_data.json` (output of `eval_post_training.py`).

### Fine-tuning (`finetune/`)

`finetune.py` — production script for RunPod A100 PCIe 40GB (~$15 total, ~10–12h):
- Knowledge distillation: frozen softmax teacher → performer student.
- Loss: `0.5 * CE + 0.5 * T² * KL`, T=2.0.
- Train: WikiText-103, 50k chunks. Eval: WT103 val + C4 val (two domains).
- 4-phase curriculum (4→8→16→32 heads), single AdamW carried across phases (momentum preserved), GradScaler, 8-bit AdamW via bitsandbytes.
- One `best_{phase}.pt` checkpoint per phase. Resume: `--resume <path> --start_phase N`.

`eval_post_training.py` — run after all 4 phases:
- Perplexity before/after FT for each checkpoint on WT103 and C4.
- Speed benchmark: prefill latency vs softmax at N=256–4096 → `speed_results.json`.
- Spectral analysis: kernel eigenvalues from real TinyLlama activations → `spectral_data.json` (feeds directly into the spectrum notebook).
- lm-eval-harness: HellaSwag, ARC-Easy, WinoGrande for teacher and Phase 4 checkpoint.

See [docs/finetune_plan.md](docs/finetune_plan.md) for all decisions and [docs/finetune_runpod_setup.md](docs/finetune_runpod_setup.md) for setup.

---

## Key Design Decisions

- `num_features=256` random features per head (M). Trade-off: quality vs speed.
- First K heads (indices 0..K-1) replaced with performer; rest stay softmax. Not learned.
- Prefill (N>1): causal scan O(N·M·D). Decode (N=1): O(M·D) state lookup.
- `omega` buffer: `persistent=True`, survives save/load.
- Triton disabled during training (autograd incompatible); checkpointed Python scan used instead.
- Causal mask built manually in `MixedPerformerAttention` — SDPA `is_causal=True` bypassed in the mixed path.

---

## Next Steps

1. **Run fine-tuning**: `cd finetune && python finetune.py` on RunPod. Follow `docs/finetune_runpod_setup.md`. Download `eval_results.json`, `spectral_data.json`, `speed_results.json` when done.
2. **Run post-training eval**: `python eval_post_training.py --ckpt_dir /workspace/checkpoints`. Full comparison table: perplexity before/after FT per phase (WT103 + C4), downstream tasks, speed.
3. **Theory-code spectral link**: load `spectral_data.json` in the spectrum notebook and overlay real activation eigenvalues against the theoretical spectral decay from `docs/theory/integral_operator.md`.
4. **Industry cost analysis**: use `speed_results.json` to model cost-per-token at production scale (long contexts, large batches) for softmax vs. each K variant.
5. **Potential extensions** (theory-driven):
   - Hyperbolic feature map (`phi_hyp`) — lower variance than current `phi_pos`.
   - Fastfood/SORF structured omega sampling — O(d log d) instead of O(d²).
   - Extend beyond softmax to Gaussian kernel attention variants.

---

## References

- [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794) — Choromanski et al., 2020
- [TinyLlama](https://arxiv.org/abs/2401.02385) — Zhang et al., 2024
- Base model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

Theory summaries in `docs/theory/`:
- [performers.md](docs/theory/performers.md) — FAVOR+ algorithm, positive random features, unbiasedness guarantees
- [kernel_approximation.md](docs/theory/kernel_approximation.md) — Bochner's theorem, RFF, PRF, structured methods
- [integral_operator.md](docs/theory/integral_operator.md) — Spectral theory for kernel operators, Hoffman-Wielandt inequality
