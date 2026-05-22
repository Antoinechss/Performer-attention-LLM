---
license: apache-2.0
datasets:
  - wikitext
  - allenai/c4
language:
  - en
metrics:
  - perplexity
base_model:
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0
pipeline_tag: text-generation
library_name: transformers
tags:
  - performer
  - linear-attention
  - favor+
  - knowledge-distillation
  - research
---

# Performer Attention — TinyLlama 1.1B Fine-tuned Checkpoints

Fine-tuned checkpoints from a research project replacing softmax attention heads in TinyLlama 1.1B with FAVOR+ (Performer) linear attention, trained via knowledge distillation.

Code and full analysis: [github.com/Antoinechss/Performer-attention-LLM](https://github.com/Antoinechss/Performer-attention-LLM)

---

## Checkpoints

| File | Performer heads | WT103 ppl | Notes |
|------|----------------|-----------|-------|
| `best_phase1_K4_QK.pt` | 4/32 | 12.28 | Q/K projections fine-tuned |
| `best_phase2_K8_QKVO.pt` | 8/32 | 12.14 | **Recommended** — best quality/speed trade-off |
| `best_phase3_K16_QKVO.pt` | 16/32 | 15.01 | Noticeable degradation |
| `best_phase4_K32_QKVO.pt` | 32/32 | 161.94 | Collapsed — reference only |

Teacher baseline (pure softmax TinyLlama): WT103 ppl = 11.52

**Phase 2 (8/32 heads)** is the recommended checkpoint: linear attention on 25% of heads, perplexity within 0.6 points of the softmax teacher.

---

## Method

FAVOR+ approximates softmax attention using positive orthogonal random features:

$$\text{Attn}(Q,K,V) \approx \phi(Q)\left(\phi(K)^T V\right)$$

reducing prefill complexity from O(N²) to O(NM) where M=256 random features.

**Training:** Knowledge distillation from a frozen softmax teacher.  
Loss: `0.5 * CE + 0.5 * T² * KL`, T=2.0  
Data: WikiText-103 (20k chunks, SEQ_LEN=256)  
4-phase curriculum: 4→8→16→32 performer heads, progressively unfreezing Q/K then Q/K/V/O projections.

---

## Usage

These checkpoints require the custom `MixedPerformerAttention` module from the project repo:

```bash
git clone https://github.com/Antoinechss/Performer-attention-LLM.git
pip install transformers==4.46.3 torch
```

```python
import sys, torch
sys.path.insert(0, "Performer-attention-LLM/performer")

from performer_attention import PerformerAttentionCore
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

# Download checkpoint
ckpt_path = hf_hub_download("antoinechss/performer-checkpoints", "best_phase2_K8_QKVO.pt")
ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)

# Load and patch model
from finetune.generate import patch_model, load_checkpoint
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model, n_heads, phase = load_checkpoint(ckpt_path)

# Generate
inputs = tokenizer("The history of artificial intelligence", return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

---

## Limitations

- Phase 4 (32/32 performer heads) collapsed during training — not suitable for inference
- FAVOR+ causal scan during training uses a sequential Python loop (slow); inference uses Triton kernels
- Trained on short sequences (SEQ_LEN=256) — quality may degrade on longer contexts
- Based on TinyLlama 1.1B — not a production model

---

## Citation

```bibtex
@misc{performer-attention-llm,
  author = {Chosson, Antoine},
  title  = {Performer Attention in TinyLlama via Knowledge Distillation},
  year   = {2026},
  url    = {https://github.com/Antoinechss/Performer-attention-LLM}
}
```

Based on:
```bibtex
@article{choromanski2020rethinking,
  title   = {Rethinking Attention with Performers},
  author  = {Choromanski, Krzysztof and others},
  journal = {ICLR},
  year    = {2021}
}
```
