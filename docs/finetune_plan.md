# Fine-tuning Plan

## Goal

Produce 4 fine-tuned checkpoints (one per phase: 4/8/16/32 performer heads) via knowledge
distillation from a frozen softmax TinyLlama teacher. Each checkpoint enables a before/after
quality comparison at each head count, and all four together show the quality-vs-efficiency
curve. Final output: perplexity curves, downstream task scores (HellaSwag, ARC-Easy,
WinoGrande via lm-eval-harness), speed benchmarks, and spectral analysis (kernel eigenvalues
from real activations).

---

## Platform

**RunPod Secure Cloud, A100 PCIe 40GB**

- ~$1.19–1.39/hr, no preemption, persistent network volume ($0.07/GB/month)
- Full terminal, run `.py` scripts directly
- Estimated total cost: ~$12–15 for the full run (10–12 hours)
- Setup: attach a 20GB network volume at `/workspace/checkpoints` before launching the pod

The finetune entrypoint is a `.py` script (`finetune.py`), not a notebook.
The analysis/evaluation files remain notebooks (run on Colab or locally).

---

## What changes and what does not

### Files that are NOT touched

- `performer/triton_scan.py` — unchanged
- `models/analysis.py` — unchanged
- `analysis.ipynb` — unchanged
- `Notebook - spectre approximations (2).ipynb` — unchanged

### File that gets one additive change

- `performer/performer_attention.py` — one new function `_python_scan_checkpointed` is added.
  The existing `_python_scan`, `_phi`, `_sample_orf`, `PerformerAttentionCore`, and
  `PerformerAttention` are untouched. The public interface is identical.
  `PerformerAttentionCore.forward` gains a `training_mode` flag (default `False`) so the
  checkpointed scan is only activated when explicitly requested. Analysis code never sets this.

### New file

- `finetune.py` — standalone training script, self-contained, runs on RunPod.

---

## Architecture decisions

### Scan memory fix (critical)

`_python_scan` stores N intermediate `S` matrices for autograd: O(N × H × M × D) memory.
For Phase 4 (32 heads, N=512) this is ~22 GB across layers on a 40GB A100 — tight.

Fix: `torch.utils.checkpoint.checkpoint(_python_scan, phi_q, phi_k, v)` recomputes the
forward scan during backward instead of storing all N states. Memory becomes O(M × D).
Cost: 2× the scan compute, which is negligible vs. the rest of the forward pass.

This is activated only when `torch.is_grad_enabled()` and `use_checkpoint=True`.
Analysis and inference paths never hit it.

### GradScaler (critical)

All fp16 backward passes need loss scaling. Without it, gradients at lr=1e-5–2e-5 underflow
to zero in fp16. Use `torch.amp.GradScaler` throughout. On A100 with bf16, GradScaler is
not needed — but we keep fp16 for A100 PCIe compatibility.

### Optimizer continuity across phases (important)

A single `AdamW` instance lives for the entire training run. Between phases, we:
1. Change which parameters require grad
2. Call `optimizer.zero_grad()`
3. Do NOT recreate the optimizer — momentum states for previously trained params are preserved

This avoids the loss spike that occurs when momentum is reset at each phase transition.

### 8-bit AdamW (important)

Use `bitsandbytes.optim.AdamW8bit`. Cuts optimizer state from 2.95 GB to ~0.4 GB.
Falls back to standard AdamW if bitsandbytes is not installed (with a warning).

### Training data

- **Train**: WikiText-103, 50k chunks of 512 tokens (~25M tokens)
- **Eval (in-domain)**: WikiText-103 validation, 300 chunks
- **Eval (out-of-domain)**: C4 validation, 300 chunks
  Both eval sets are reported at every eval step. The C4 number is what goes in the
  comparison — same domain as training risks overfitting signal.

### Training phases

All 4 phases use the same single training loop. Phase transitions are just:
- `set_performer_heads(student, K)` — changes routing, no weight reset
- `unfreeze_fn(student)` — unfreezes the right projections
- LR is reduced for later phases (see table below)
- Optimizer is NOT recreated

| Phase | Heads | Unfrozen | LR | Epochs |
|-------|-------|----------|----|--------|
| 1 | 4/32 | Q, K | 2e-5 | 3 |
| 2 | 8/32 | Q, K, V, O | 2e-5 | 3 |
| 3 | 16/32 | Q, K, V, O | 1e-5 | 3 |
| 4 | 32/32 | Q, K, V, O | 1e-5 | 3 |

Effective batch size: 8 (micro_batch=1, grad_accum=8).
Cosine LR schedule with 5% warmup, reset per phase.

### Loss

`L = 0.5 * CE_loss + 0.5 * T² * KL_loss`  with T=2.0

CE = cross-entropy on next-token prediction (ground truth labels)
KL = KL divergence between student and teacher output distributions

### Checkpointing

At end of each phase: save `best_{phase_name}.pt` containing model state dict + optimizer
state dict + scheduler state dict + omega buffers + phase metadata.

Also save every 500 steps as `{phase_name}_step{N}.pt` for crash recovery.

Checkpoints go to `/workspace/checkpoints/` (RunPod network volume).

---

## Evaluation plan (post-training)

After all 4 phases complete, the script runs:

1. **Perplexity sweep**: for each checkpoint (4/8/16/32 heads), eval on WikiText-103 val and
   C4 val. Also eval the unmodified teacher for the baseline.

2. **lm-eval-harness** (via subprocess): HellaSwag, ARC-Easy, WinoGrande. Run for teacher
   and for the Phase 4 (32/32) checkpoint. These are the downstream task scores.

3. **Speed benchmark**: prefill latency at N=256/512/1024/2048/4096 for softmax vs. performer
   at each head count. This reuses the logic from `models/analysis.py` Section B.

4. **Spectral analysis**: extract Q and K activations from a real batch, compute the actual
   attention kernel matrix (softmax and FAVOR+ approximation), compute eigenvalues, compare
   decay curves. This is the theory-code link from context.md next step 2.
   Output: a JSON file of eigenvalue arrays per layer, per head count. The spectrum notebook
   can then load and plot these.

---

## File structure after implementation

```
performer/
  performer_attention.py       # +_python_scan_checkpointed (additive only)
finetune.py                    # new: full training script for RunPod
eval_post_training.py          # new: perplexity + lm-eval + speed + spectral
docs/
  finetune_plan.md             # this file
  finetune_runpod_setup.md     # RunPod setup instructions (pod config, volume mount)
```

---

## RunPod setup (before running)

See `docs/finetune_runpod_setup.md` for step-by-step pod configuration.
Short version:
1. Create a network volume (20 GB) in your region
2. Launch pod: A100 PCIe 40GB, RunPod PyTorch 2.1 template, attach the volume at `/workspace`
3. In the terminal: `git clone <repo>`, `pip install -r requirements_finetune.txt`
4. `python finetune.py`

The script auto-saves to `/workspace/checkpoints/` if that path exists, else falls back to
`./checkpoints/`.
