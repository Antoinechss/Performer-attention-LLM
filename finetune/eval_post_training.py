"""
Post-training evaluation: perplexity, downstream tasks, speed, spectral analysis.

Run after finetune.py has produced all 4 phase checkpoints:
    python eval_post_training.py --ckpt_dir /workspace/checkpoints

Outputs:
  - eval_results.json   : perplexity on WT103 + C4 for each checkpoint + teacher
  - speed_results.json  : prefill/decode latency vs. softmax
  - spectral_data.json  : kernel eigenvalues per layer for softmax and performer
  - lm_eval_results/    : lm-eval-harness outputs (HellaSwag, ARC-Easy, WinoGrande)
"""

import sys, os, json, time, math, argparse, subprocess
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PERFORMER_DIR = os.path.join(_REPO_ROOT, "performer")
if _PERFORMER_DIR not in sys.path:
    sys.path.insert(0, _PERFORMER_DIR)

from performer_attention import PerformerAttentionCore, _HAS_TRITON, _phi

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID    = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DTYPE       = torch.float16
DEVICE      = "cuda"
SEQ_LEN     = 512
MAX_VAL_SAMPLES = 300
SEED        = 42
torch.manual_seed(SEED)


# ── MixedPerformerAttention — same as finetune.py ─────────────────────────────
# (copied to keep this file self-contained; no shared state with finetune.py)

class MixedPerformerAttention(torch.nn.Module):
    def __init__(self, original_attn, num_performer_heads):
        super().__init__()
        self.head_dim              = original_attn.head_dim
        self.num_heads             = original_attn.config.num_attention_heads
        self.num_key_value_heads   = original_attn.config.num_key_value_heads
        self.num_key_value_groups  = self.num_heads // self.num_key_value_heads
        self.scaling               = self.head_dim ** -0.5
        self.num_performer_heads   = num_performer_heads
        self.num_standard_heads    = self.num_heads - num_performer_heads
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj
        self.performer_core = PerformerAttentionCore(head_dim=self.head_dim, num_features=256)
        self.config    = original_attn.config
        self.layer_idx = original_attn.layer_idx
        self.is_causal = True

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary(self, q, k, cos, sin):
        cos = cos.unsqueeze(1); sin = sin.unsqueeze(1)
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
            if attention_mask is not None: scores = scores + attention_mask
            w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_out = torch.matmul(w, v)
        else:
            Kp = self.num_performer_heads
            out_p = self.performer_core(q[:, :Kp], k[:, :Kp], v[:, :Kp])
            q_s, k_s, v_s = q[:, Kp:], k[:, Kp:], v[:, Kp:]
            scores = torch.matmul(q_s, k_s.transpose(-2, -1)) * self.scaling
            if attention_mask is not None: scores = scores + attention_mask
            w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q_s.dtype)
            out_s = torch.matmul(w, v_s)
            attn_out = torch.cat([out_p, out_s], dim=1)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, N, -1)
        return self.o_proj(attn_out), None, None


def patch_model(model, num_performer_heads):
    for layer in model.model.layers:
        layer.self_attn = MixedPerformerAttention(layer.self_attn, num_performer_heads)
    return model


def set_performer_heads(model, num_performer_heads):
    for layer in model.model.layers:
        layer.self_attn.num_performer_heads = num_performer_heads
        layer.self_attn.num_standard_heads  = model.config.num_attention_heads - num_performer_heads


def load_performer_checkpoint(ckpt_path, base_model, tokenizer):
    """Load a phase checkpoint into a freshly patched model."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    phase = ckpt["phase"]
    # Infer head count from phase name: phase1_K4_QK -> 4
    n_heads = int(phase.split("_K")[1].split("_")[0])
    model = patch_model(base_model, n_heads)
    # Restore omegas
    for i, layer in enumerate(model.model.layers):
        key = f"layer_{i}"
        if key in ckpt["omegas"] and hasattr(layer.self_attn, "performer_core"):
            layer.self_attn.performer_core.omega.copy_(ckpt["omegas"][key].to(DEVICE))
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, n_heads, phase


# ── Dataset ───────────────────────────────────────────────────────────────────

class LMDataset(Dataset):
    def __init__(self, token_ids, seq_len, max_samples):
        n = (len(token_ids) // seq_len) * seq_len
        data = token_ids[:n].view(-1, seq_len)[:max_samples]
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx]
        return {"input_ids": x, "labels": x.clone()}


def get_val_loaders(tokenizer):
    wt = load_dataset("wikitext", "wikitext-103-raw-v1")
    tokens_wt = []
    for text in wt["validation"]["text"]:
        if text.strip():
            tokens_wt.extend(tokenizer.encode(text, add_special_tokens=False))

    c4_raw = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    tokens_c4 = []
    needed = MAX_VAL_SAMPLES * SEQ_LEN + SEQ_LEN
    for ex in c4_raw:
        tokens_c4.extend(tokenizer.encode(ex["text"], add_special_tokens=False))
        if len(tokens_c4) >= needed:
            break

    ds_wt = LMDataset(torch.tensor(tokens_wt, dtype=torch.long), SEQ_LEN, MAX_VAL_SAMPLES)
    ds_c4 = LMDataset(torch.tensor(tokens_c4, dtype=torch.long), SEQ_LEN, MAX_VAL_SAMPLES)

    ldr_wt = DataLoader(ds_wt, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    ldr_c4 = DataLoader(ds_c4, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    return ldr_wt, ldr_c4


# ── Perplexity ────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_ppl(model, loader):
    model.eval()
    total_ce, n_tokens = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        labels    = batch["labels"].to(DEVICE)
        with torch.amp.autocast("cuda", dtype=DTYPE):
            out = model(input_ids=input_ids, use_cache=False)
        logits = out.logits[:, :-1].float()
        tgt    = labels[:, 1:].contiguous()
        ce     = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="sum")
        total_ce += ce.item()
        n_tokens += tgt.numel()
    return math.exp(min(total_ce / n_tokens, 20))


# ── Speed benchmark ───────────────────────────────────────────────────────────

def time_fn(fn, repeats=5, cuda=True):
    if cuda: torch.cuda.synchronize()
    fn()  # warmup
    if cuda: torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if cuda: torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000


def run_speed_benchmark():
    """Prefill latency: softmax vs performer at various seq lengths and head counts."""
    H, D    = 32, 64
    M       = 256
    SEQ_LEN_LIST = [256, 512, 1024, 2048, 4096]
    HEAD_COUNTS  = [0, 4, 8, 16, 32]   # 0 = pure softmax

    results = {}
    _dtype  = torch.float16

    for N in SEQ_LEN_LIST:
        results[N] = {}
        q = torch.randn(1, H, N, D, device=DEVICE, dtype=_dtype)
        k = torch.randn(1, H, N, D, device=DEVICE, dtype=_dtype)
        v = torch.randn(1, H, N, D, device=DEVICE, dtype=_dtype)

        def std_attn():
            w = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * (D ** -0.5), dim=-1)
            return torch.matmul(w, v)

        std_ms = time_fn(std_attn)
        results[N]["softmax"] = std_ms

        for K in HEAD_COUNTS:
            if K == 0:
                continue
            core = PerformerAttentionCore(head_dim=D, num_features=M).to(DEVICE)
            def mixed(c=core, _K=K):
                # K performer heads, rest softmax
                out_p = c(q[:, :_K], k[:, :_K], v[:, :_K])
                if _K < H:
                    w = torch.softmax(torch.matmul(q[:, _K:], k[:, _K:].transpose(-2,-1)) * (D**-0.5), dim=-1)
                    out_s = torch.matmul(w, v[:, _K:])
                    return torch.cat([out_p, out_s], dim=1)
                return out_p

            results[N][f"K{K}"] = time_fn(mixed)

    return results


# ── Spectral analysis ─────────────────────────────────────────────────────────

def extract_kernel_eigenvalues(model_std, model_perf, tokenizer, n_heads_perf, n_samples=10):
    """
    For each layer, compute the attention kernel matrix for softmax and FAVOR+
    on real input activations, then compute eigenvalue spectra.

    Returns dict: {layer_idx: {"softmax": [eigs], "performer": [eigs]}}
    averaged across n_samples batches.
    """
    import numpy as np

    PROMPT = "The quick brown fox jumps over the lazy dog. " * 8  # ~100 tokens, repeat for length
    input_ids = tokenizer(PROMPT, return_tensors="pt", max_length=SEQ_LEN,
                          truncation=True)["input_ids"].to(DEVICE)

    # We'll hook into layer 0, layer 5, layer 10, layer 15, layer 21 for efficiency
    target_layers = [0, 5, 10, 15, 21]
    spectral_data = {l: {"softmax": [], "performer": []} for l in target_layers}

    model_std.eval()
    model_perf.eval()

    for sample_idx in range(n_samples):
        # Slightly vary input by trimming to different lengths
        length = SEQ_LEN - sample_idx * 4
        ids = input_ids[:, :length]

        for layer_idx in target_layers:
            # Hook to capture Q and K at this layer
            q_captured, k_captured = {}, {}

            def make_hook(name, storage):
                def hook(module, args, kwargs_hook, output):
                    # output[0] is attn_out, we need Q/K before attention
                    pass
                return hook

            # Simpler: run forward and extract Q/K from the attention module directly
            # We instrument the forward pass by temporarily wrapping
            layer_std  = model_std.model.layers[layer_idx]
            layer_perf = model_perf.model.layers[layer_idx]

            # Capture Q, K by running the projection manually
            with torch.no_grad():
                # Get hidden states at this layer by running up to layer_idx
                # Use a minimal forward hook approach
                hidden_std  = [None]
                hidden_perf = [None]

                def hook_std(module, inp, kwargs_h, out):
                    hs = kwargs_h.get("hidden_states") or (inp[0] if inp else None)
                    if hs is not None:
                        hidden_std[0] = hs.detach()

                def hook_perf(module, inp, kwargs_h, out):
                    hs = kwargs_h.get("hidden_states") or (inp[0] if inp else None)
                    if hs is not None:
                        hidden_perf[0] = hs.detach()

                h_std  = layer_std.self_attn.register_forward_hook(hook_std,  with_kwargs=True)
                h_perf = layer_perf.self_attn.register_forward_hook(hook_perf, with_kwargs=True)

                with torch.amp.autocast("cuda", dtype=DTYPE):
                    model_std(input_ids=ids, use_cache=False)
                    model_perf(input_ids=ids, use_cache=False)

                h_std.remove()
                h_perf.remove()

                hs_std  = hidden_std[0]   # [1, N, H_dim]
                hs_perf = hidden_perf[0]  # [1, N, H_dim]

                # Compute Q, K from std model projections (same for perf model pre-attention)
                B, N, _ = hs_std.shape
                attn_std = layer_std.self_attn
                num_heads  = attn_std.config.num_attention_heads
                head_dim   = attn_std.head_dim
                num_kv     = attn_std.config.num_key_value_heads
                num_groups = num_heads // num_kv

                q_std = attn_std.q_proj(hs_std).view(B, N, num_heads, head_dim).transpose(1, 2)
                k_std = attn_std.k_proj(hs_std).view(B, N, num_kv,    head_dim).transpose(1, 2)
                if num_groups > 1:
                    k_std = k_std.repeat_interleave(num_groups, dim=1)

                # Softmax kernel matrix: K_softmax[i,j] = exp(q_i · k_j / sqrt(d))
                # Average over heads
                scale     = head_dim ** -0.5
                scores    = torch.matmul(q_std.float(), k_std.float().transpose(-2, -1)) * scale
                # scores: [1, H, N, N]
                # Kernel matrix (pre-softmax exp): exp(scores) averaged over heads
                K_softmax = torch.exp(scores).mean(dim=1).squeeze(0).cpu().numpy()  # [N, N]

                # FAVOR+ kernel matrix: phi(q)^T phi(k) averaged over performer heads
                attn_perf = layer_perf.self_attn
                q_perf = attn_perf.q_proj(hs_perf).view(B, N, num_heads, head_dim).transpose(1, 2)
                k_perf = attn_perf.k_proj(hs_perf).view(B, N, num_kv,    head_dim).transpose(1, 2)
                if num_groups > 1:
                    k_perf = k_perf.repeat_interleave(num_groups, dim=1)

                core      = attn_perf.performer_core
                q_s       = q_perf.float() * (head_dim ** -0.25)
                k_s       = k_perf.float() * (head_dim ** -0.25)
                phi_q     = core.phi(q_s[:, :n_heads_perf], is_query=True)   # [1, K, N, M]
                phi_k     = core.phi(k_s[:, :n_heads_perf], is_query=False)  # [1, K, N, M]
                # Approx kernel: phi_q[i] · phi_k[j] averaged over performer heads
                K_perf = torch.bmm(
                    phi_q.squeeze(0).mean(0),               # [N, M]
                    phi_k.squeeze(0).mean(0).transpose(0, 1)  # [M, N]
                ).cpu().numpy()  # [N, N]

                # Eigenvalues of symmetric part (kernel matrices are not symmetric
                # in general due to causal masking, so use SVD singular values)
                import numpy.linalg as nla
                eigs_softmax  = sorted(nla.svd(K_softmax, compute_uv=False).tolist(), reverse=True)
                eigs_performer = sorted(nla.svd(K_perf,    compute_uv=False).tolist(), reverse=True)

                spectral_data[layer_idx]["softmax"].append(eigs_softmax[:64])
                spectral_data[layer_idx]["performer"].append(eigs_performer[:64])

    # Average across samples
    averaged = {}
    for l in target_layers:
        import numpy as np
        avg_s = np.array(spectral_data[l]["softmax"]).mean(axis=0).tolist()
        avg_p = np.array(spectral_data[l]["performer"]).mean(axis=0).tolist()
        averaged[str(l)] = {"softmax": avg_s, "performer": avg_p}

    return averaged


# ── lm-eval-harness ───────────────────────────────────────────────────────────

def run_lm_eval(ckpt_path, output_dir, tasks="hellaswag,arc_easy,winogrande"):
    """Run lm-eval-harness on a checkpoint via subprocess."""
    os.makedirs(output_dir, exist_ok=True)

    # lm-eval expects a HF model path or a saved model directory.
    # Since we have a custom architecture, we use the --model hf flag with a
    # wrapper approach: save the student as a HF model temporarily, then eval.
    # Alternatively, use the lm_eval Python API.
    try:
        import lm_eval
        _HAS_LMEVAL = True
    except ImportError:
        _HAS_LMEVAL = False

    if not _HAS_LMEVAL:
        print("lm-eval not installed. Skipping downstream task eval.")
        print("Install with: pip install lm-eval")
        return None

    # Use lm_eval Python API
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    # Load the checkpoint into a model and wrap for lm-eval
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    model_eval, n_heads, phase_name = load_performer_checkpoint(ckpt_path, base, tokenizer)

    lm = HFLM(pretrained=model_eval, tokenizer=tokenizer, batch_size=4)

    results = simple_evaluate(
        model=lm,
        tasks=tasks.split(","),
        num_fewshot=0,
        log_samples=False,
    )

    out_path = os.path.join(output_dir, f"{phase_name}_lmeval.json")
    with open(out_path, "w") as f:
        json.dump(results["results"], f, indent=2)
    print(f"lm-eval results saved to {out_path}")
    return results["results"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str,
                        default="/workspace/checkpoints" if os.path.isdir("/workspace") else "./checkpoints")
    parser.add_argument("--skip_speed",   action="store_true")
    parser.add_argument("--skip_spectral", action="store_true")
    parser.add_argument("--skip_lmeval",  action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    val_wt_loader, val_c4_loader = get_val_loaders(tokenizer)

    eval_results = {}

    # ── 1. Teacher baseline ───────────────────────────────────────────────────
    print("\n[1/4] Teacher perplexity baseline...")
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    teacher.eval()
    ppl_wt = compute_ppl(teacher, val_wt_loader)
    ppl_c4 = compute_ppl(teacher, val_c4_loader)
    print(f"  Teacher: WT103={ppl_wt:.2f}  C4={ppl_c4:.2f}")
    eval_results["teacher"] = {"wt103": ppl_wt, "c4": ppl_c4}

    # ── 2. Per-checkpoint perplexity ──────────────────────────────────────────
    print("\n[2/4] Checkpoint perplexity evaluation...")
    phase_names = [
        "phase1_K4_QK",
        "phase2_K8_QKVO",
        "phase3_K16_QKVO",
        "phase4_K32_QKVO",
    ]
    for phase_name in phase_names:
        ckpt_path = os.path.join(args.ckpt_dir, f"best_{phase_name}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {phase_name} (checkpoint not found)")
            continue

        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
        model, n_heads, _ = load_performer_checkpoint(ckpt_path, base, tokenizer)

        # Before fine-tuning: eval a fresh patched model (untrained)
        base_raw = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
        model_raw = patch_model(base_raw, n_heads)
        model_raw.eval()
        ppl_wt_raw = compute_ppl(model_raw, val_wt_loader)
        ppl_c4_raw = compute_ppl(model_raw, val_c4_loader)

        ppl_wt_ft = compute_ppl(model, val_wt_loader)
        ppl_c4_ft = compute_ppl(model, val_c4_loader)

        print(f"  {phase_name}:")
        print(f"    Before FT : WT103={ppl_wt_raw:.2f}  C4={ppl_c4_raw:.2f}")
        print(f"    After  FT : WT103={ppl_wt_ft:.2f}  C4={ppl_c4_ft:.2f}")

        eval_results[phase_name] = {
            "n_heads":       n_heads,
            "before_ft":     {"wt103": ppl_wt_raw, "c4": ppl_c4_raw},
            "after_ft":      {"wt103": ppl_wt_ft,  "c4": ppl_c4_ft},
        }

        del model, model_raw, base, base_raw
        torch.cuda.empty_cache()

    out_path = os.path.join(args.ckpt_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nPerplexity results saved to {out_path}")

    # ── 3. Speed benchmark ────────────────────────────────────────────────────
    if not args.skip_speed:
        print("\n[3/4] Speed benchmark (prefill latency)...")
        speed = run_speed_benchmark()
        speed_path = os.path.join(args.ckpt_dir, "speed_results.json")
        with open(speed_path, "w") as f:
            json.dump(speed, f, indent=2)
        print(f"Speed results saved to {speed_path}")
        print("\nPrefill latency (ms):")
        header = f"{'N':>6}  {'softmax':>10}" + "".join(f"  {'K='+str(k):>8}" for k in [4,8,16,32])
        print(header)
        for N, row in speed.items():
            line = f"{N:>6}  {row['softmax']:>10.2f}"
            for k in [4, 8, 16, 32]:
                key = f"K{k}"
                line += f"  {row.get(key, float('nan')):>8.2f}"
            print(line)

    # ── 4. Spectral analysis ──────────────────────────────────────────────────
    if not args.skip_spectral:
        print("\n[4/4] Spectral analysis (kernel eigenvalues on real activations)...")
        # Use the Phase 4 checkpoint (32/32 heads) for spectral comparison
        ckpt_path = os.path.join(args.ckpt_dir, "best_phase4_K32_QKVO.pt")
        if os.path.exists(ckpt_path):
            base_std  = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
            base_perf = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
            model_perf, n_heads_perf, _ = load_performer_checkpoint(ckpt_path, base_perf, tokenizer)
            base_std.eval()
            model_perf.eval()

            spectral = extract_kernel_eigenvalues(base_std, model_perf, tokenizer, n_heads_perf)

            spectral_path = os.path.join(args.ckpt_dir, "spectral_data.json")
            with open(spectral_path, "w") as f:
                json.dump(spectral, f, indent=2)
            print(f"Spectral data saved to {spectral_path}")
            print("  Layers analysed:", list(spectral.keys()))

            del base_std, base_perf, model_perf
            torch.cuda.empty_cache()
        else:
            print("  Phase 4 checkpoint not found. Skipping spectral analysis.")

    # ── 5. lm-eval-harness ───────────────────────────────────────────────────
    if not args.skip_lmeval:
        print("\n[+] Downstream task evaluation (lm-eval-harness)...")
        lmeval_dir = os.path.join(args.ckpt_dir, "lm_eval_results")
        # Run on teacher and on Phase 4 checkpoint
        print("  Teacher...")
        # lm-eval on vanilla HF model
        try:
            import lm_eval
            from lm_eval import simple_evaluate
            from lm_eval.models.huggingface import HFLM
            lm_teacher = HFLM(pretrained=teacher, tokenizer=tokenizer, batch_size=4)
            res_teacher = simple_evaluate(lm_teacher, tasks=["hellaswag","arc_easy","winogrande"], num_fewshot=0, log_samples=False)
            os.makedirs(lmeval_dir, exist_ok=True)
            with open(os.path.join(lmeval_dir, "teacher_lmeval.json"), "w") as f:
                json.dump(res_teacher["results"], f, indent=2)
            print("  Teacher lm-eval done.")

            ckpt_p4 = os.path.join(args.ckpt_dir, "best_phase4_K32_QKVO.pt")
            if os.path.exists(ckpt_p4):
                print("  Phase 4 (32/32 performer)...")
                run_lm_eval(ckpt_p4, lmeval_dir)
        except ImportError:
            print("  lm-eval not installed. Run: pip install lm-eval")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
