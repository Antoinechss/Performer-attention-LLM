"""
Check whether the model bypasses Performer heads by zeroing out their o_proj columns.

For each phase checkpoint, computes the L2 norm of the o_proj weight columns
corresponding to Performer heads vs softmax heads. If the model learned to ignore
Performer output, the Performer column norms should be near zero.

Usage:
    python finetune/oproj_norm_check.py --ckpt_dir /workspace/checkpoints
"""

import sys, os, argparse
import torch

_REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PERFORMER_DIR = os.path.join(_REPO_ROOT, "performer")
if _PERFORMER_DIR not in sys.path:
    sys.path.insert(0, _PERFORMER_DIR)

from performer_attention import PerformerAttentionCore
from transformers import AutoModelForCausalLM

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE   = "cpu"   # norm check needs no GPU
DTYPE    = torch.float32


class MixedPerformerAttention(torch.nn.Module):
    def __init__(self, original_attn, num_performer_heads):
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
        self.performer_core = PerformerAttentionCore(head_dim=self.head_dim, num_features=256)
        self.config    = original_attn.config
        self.layer_idx = original_attn.layer_idx
        self.is_causal = True


def patch_model(model, num_performer_heads):
    for layer in model.model.layers:
        layer.self_attn = MixedPerformerAttention(layer.self_attn, num_performer_heads)
    return model


def load_checkpoint(ckpt_path):
    ckpt    = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    phase   = ckpt["phase"]
    n_heads = int(phase.split("_K")[1].split("_")[0])
    base    = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    model   = patch_model(base, n_heads)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, n_heads, phase


def oproj_norms(model, num_performer_heads):
    """
    Returns (perf_norms, softmax_norms) where each is a list of per-layer
    average column norms for Performer heads and softmax heads respectively.
    """
    perf_norms_per_layer    = []
    softmax_norms_per_layer = []

    for layer in model.model.layers:
        W        = layer.self_attn.o_proj.weight.float()  # [hidden, H * head_dim]
        head_dim = layer.self_attn.head_dim
        n_heads  = layer.self_attn.num_heads

        col_norms = []
        for h in range(n_heads):
            block = W[:, h * head_dim : (h + 1) * head_dim]
            col_norms.append(block.norm().item())

        perf_norms_per_layer.append(col_norms[:num_performer_heads])
        softmax_norms_per_layer.append(col_norms[num_performer_heads:])

    return perf_norms_per_layer, softmax_norms_per_layer


def summarise(perf_norms_per_layer, softmax_norms_per_layer, num_performer_heads):
    n_layers = len(perf_norms_per_layer)

    print(f"\n{'Layer':>5}  {'Perf avg':>10}  {'Perf min':>10}  {'Softmax avg':>12}  {'Softmax min':>12}  {'Ratio':>7}")
    print("-" * 65)

    global_perf    = []
    global_softmax = []

    for i in range(n_layers):
        p  = perf_norms_per_layer[i]
        s  = softmax_norms_per_layer[i]
        pa = sum(p) / len(p)
        sa = sum(s) / len(s)
        ratio = pa / sa if sa > 0 else float("inf")
        global_perf.extend(p)
        global_softmax.extend(s)
        print(f"{i:>5}  {pa:>10.3f}  {min(p):>10.3f}  {sa:>12.3f}  {min(s):>12.3f}  {ratio:>7.3f}")

    gpa = sum(global_perf) / len(global_perf)
    gsa = sum(global_softmax) / len(global_softmax)
    print("-" * 65)
    print(f"{'ALL':>5}  {gpa:>10.3f}  {min(global_perf):>10.3f}  {gsa:>12.3f}  {min(global_softmax):>12.3f}  {gpa/gsa:>7.3f}")
    print()

    if gpa / gsa < 0.1:
        verdict = "BYPASS LIKELY — Performer column norms are <10% of softmax norms."
    elif gpa / gsa < 0.5:
        verdict = "PARTIAL SUPPRESSION — Performer columns are significantly weaker."
    else:
        verdict = "NO BYPASS — Performer and softmax column norms are comparable."
    print(f"Verdict: {verdict}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="./checkpoints")
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir
    if not os.path.isdir(ckpt_dir):
        print(f"Checkpoint directory not found: {ckpt_dir}")
        return

    ckpt_files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
    if not ckpt_files:
        print(f"No .pt checkpoints found in {ckpt_dir}")
        return

    for fname in ckpt_files:
        path = os.path.join(ckpt_dir, fname)
        print(f"\n{'='*65}")
        print(f"Checkpoint: {fname}")
        model, n_heads, phase = load_checkpoint(path)
        print(f"Phase: {phase}  |  Performer heads: {n_heads}/32")

        perf_norms, softmax_norms = oproj_norms(model, n_heads)
        summarise(perf_norms, softmax_norms, n_heads)


if __name__ == "__main__":
    main()
