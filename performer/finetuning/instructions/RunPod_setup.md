# RunPod Setup for Fine-tuning

## Why RunPod Secure Cloud

- No preemption, dedicated hardware
- Persistent network volumes (data survives pod stop/start)
- Full terminal access, run `.py` scripts directly
- A100 PCIe 80GB at ~$1.39/hr: ~$14–17 for the full 10–12h run
- Total budget stays under $20 including storage

---

## Step 1 — Create a network volume

1. Go to runpod.io → Storage → Network Volumes → New Volume
2. Name: `performer-checkpoints`
3. Size: 20 GB (sufficient for 4 phase checkpoints + eval outputs)
4. Region: pick the same region you'll launch the pod in (US-TX-3 or EU-RO-1 are reliable)
5. Click Create

Cost: 20 GB × $0.07/GB/month = ~$1.40/month (negligible)

---

## Step 2 — Launch the pod

1. Go to Pods → New Pod
2. Select GPU: **A100 PCIe 80GB** (Secure Cloud tab — NOT Community Cloud)
3. Template: **RunPod PyTorch 2.1** (has CUDA 12.1, Python 3.10, torch pre-installed)
4. Container disk: 20 GB (for HuggingFace model cache)
5. Volume: attach the `performer-checkpoints` volume at `/workspace`
6. Expose HTTP ports: none needed
7. Click Deploy

---

## Step 3 — First-time setup (run once in the terminal)

Open the pod terminal via the web UI or SSH.

```bash
# Clone the repo
cd /workspace
git clone https://github.com/Antoinechss/Performer-attention-LLM.git
cd Performer-attention-LLM

# Install dependencies
pip install -r finetune/requirements_finetune.txt

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')"

# Verify bitsandbytes
python -c "import bitsandbytes as bnb; print('bnb ok')"

# Verify triton (optional, for inference speed only)
python -c "import triton; print('triton ok')"
```

---

## Step 4 — Run training

Always run inside `tmux` so the process survives SSH disconnects (hotspot drops,
closing your laptop, etc.). The training runs on RunPod's server — your connection
only needs to be alive to see the output, not to keep the job running.

```bash
cd /workspace/Performer-attention-LLM

# Start a tmux session
tmux new -s train

# Run training
python finetune/finetune.py

# Detach at any time with Ctrl+B then D — training keeps running
# Reattach later (from any connection) with:
#   tmux attach -t train
```

If you prefer a log file over live output:
```bash
nohup python finetune/finetune.py > /workspace/train.log 2>&1 &
tail -f /workspace/train.log
```

Checkpoints are auto-saved to `/workspace/checkpoints/` (the network volume).

Expected runtime per phase (A100 PCIe 40GB, 50k samples, 3 epochs):
- Phase 1 (4 heads, Q/K only): ~1.5–2h
- Phase 2 (8 heads, Q/K/V/O): ~2–2.5h
- Phase 3 (16 heads): ~2.5–3h
- Phase 4 (32 heads): ~3–3.5h
- **Total: ~9–11h → ~$11–15**

---

## Step 5 — Resuming after a disconnect

If the session drops (unlikely on Secure Cloud, but possible), the latest step checkpoint
is at `/workspace/checkpoints/{phase_name}_step{N}.pt`.

Resume from a specific phase and step:
```bash
python finetune/finetune.py --resume /workspace/checkpoints/phase2_K8_QKVO_step1000.pt --start_phase 1
```

`--start_phase` is 0-indexed (0=phase1, 1=phase2, 2=phase3, 3=phase4).

---

## Step 6 — Run evaluation

After all phases complete:
```bash
python finetune/eval_post_training.py --ckpt_dir /workspace/checkpoints
```

Outputs saved to `/workspace/checkpoints/`:
- `eval_results.json` — perplexity before/after FT for each phase, on WT103 and C4
- `speed_results.json` — prefill latency vs softmax
- `spectral_data.json` — kernel eigenvalues per layer (load in spectrum notebook)
- `lm_eval_results/` — HellaSwag, ARC-Easy, WinoGrande scores

---

## Step 7 — Download results

From your local machine:
```bash
# Copy results JSON files (small)
scp -P <PORT> root@<POD_IP>:/workspace/checkpoints/eval_results.json .
scp -P <PORT> root@<POD_IP>:/workspace/checkpoints/spectral_data.json .
scp -P <PORT> root@<POD_IP>:/workspace/checkpoints/speed_results.json .

# Copy best checkpoints (large, ~3GB each — only if you need them locally)
scp -P <PORT> root@<POD_IP>:/workspace/checkpoints/best_phase4_K32_QKVO.pt .
```

Or use the RunPod web UI file browser for small files.

**Stop the pod** (not just pause) after downloading to stop billing.
Network volume persists independently — checkpoints are safe.

---

## Cost breakdown

| Item | Hours | Rate | Cost |
|------|-------|------|------|
| A100 PCIe 80GB (Secure Cloud) | 11h | ~$1.39/hr | ~$15.30 |
| Network volume (20 GB) | 1 month | $0.07/GB/mo | ~$1.40 |
| Container disk | included | — | $0 |
| **Total** | | | **~$15.70** |

Well within the $20 budget.
