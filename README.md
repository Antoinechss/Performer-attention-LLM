# Performer Attention LLM

M2 research project — CERMICS / Advansight.

Replacing a fraction of softmax attention heads in TinyLlama 1.1B with FAVOR+ (Performer) linear attention, then recovering output quality via knowledge distillation. The goal is to reduce attention complexity from O(N²) to O(N) while preserving generation quality.

---

## The Triangle

The project evaluates the softmax → FAVOR+ substitution across three independent axes:

```
                    QUALITY
                   (perplexity, generation)
                        ▲
                       / \
                      /   \
                     /     \
                    /       \
        SPEED ◄─────────────► APPROXIMATION
  (prefill/decode latency)    (kernel eigenvalues,
                               M/D convergence)
```

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
performer/
  performer_attention.py   # PerformerAttentionCore, FAVOR+ feature map, causal scan
  triton_scan.py           # Triton CUDA kernels (inference only)

analysis/
  analysis.ipynb           # Main notebook: quality, speed, approximation
  analysis.py              # Standalone script version
  approximation/           # Kernel eigenvalue spectrum notebook

finetune/
  finetune.py              # Training script (knowledge distillation, 4-phase curriculum)
  eval_post_training.py    # Post-training evaluation: perplexity, speed, spectral, lm-eval
  generate.py              # Generation comparison across checkpoints
  requirements_finetune.txt

docs/
  context.md               # Full project context
  finetune_plan.md         # Fine-tuning decisions and rationale
  finetune_runpod_setup.md # RunPod setup guide
  theory/                  # PDF summaries: FAVOR+, kernel approximation, spectral theory
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

## References

- [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794) — Choromanski et al., 2020
- [TinyLlama](https://arxiv.org/abs/2401.02385) — Zhang et al., 2024
- Base model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
