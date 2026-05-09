"""
Performer attention fine-tuning via knowledge distillation.

Run on RunPod A100 PCIe 40GB:
    python finetune.py

Produces one checkpoint per phase (4/8/16/32 performer heads).
Checkpoints saved to /workspace/checkpoints/ if it exists, else ./checkpoints/.

See docs/finetune_plan.md for the full plan.
"""

import sys, os, time, math, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Path setup ────────────────────────────────────────────────────────────────
# Add performer/ to sys.path. Never add the repo root (shadows HF transformers/).
_REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PERFORMER_DIR = os.path.join(_REPO_ROOT, "performer")
if _PERFORMER_DIR not in sys.path:
    sys.path.insert(0, _PERFORMER_DIR)

from performer_attention import PerformerAttentionCore, _HAS_TRITON

# HF imports — must come after sys.path fix
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from datasets import load_dataset

# ── Optional: 8-bit AdamW ─────────────────────────────────────────────────────
try:
    import bitsandbytes as bnb
    _HAS_BNB = True
except ImportError:
    _HAS_BNB = False
    print("WARNING: bitsandbytes not installed. Using fp32 AdamW (higher VRAM).")
    print("         Install with: pip install bitsandbytes")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DTYPE      = torch.float16
DEVICE     = "cuda"
SEQ_LEN    = 512
SEED       = 42

# Distillation
ALPHA       = 0.5   # CE weight; (1-ALPHA) for KL
TEMPERATURE = 2.0

# Training
MICRO_BATCH        = 1
GRAD_ACCUM         = 8      # effective batch = 8
SAVE_STEPS         = 500
EVAL_STEPS         = 250
MAX_TRAIN_SAMPLES  = 50_000  # ~25M tokens; enough for phases 3-4
MAX_VAL_SAMPLES    = 300

# Phases: (num_performer_heads, unfreeze_mode, lr, epochs)
PHASES = [
    (4,  "qk",   2e-5, 3),
    (8,  "qkvo", 2e-5, 3),
    (16, "qkvo", 1e-5, 3),
    (32, "qkvo", 1e-5, 3),
]

# Checkpoint directory: prefer RunPod network volume
CKPT_DIR = "/workspace/checkpoints" if os.path.isdir("/workspace") else "./checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

torch.manual_seed(SEED)
print(f"Checkpoint dir : {CKPT_DIR}")
print(f"Triton available: {_HAS_TRITON}")
print(f"8-bit AdamW    : {_HAS_BNB}")


# ── MixedPerformerAttention ───────────────────────────────────────────────────

class MixedPerformerAttention(nn.Module):
    """Wraps HF LlamaAttention, routing the first K heads through FAVOR+.

    Identical contract to the version in analysis.py / analysis.ipynb.
    Key differences for training:
      - omega is persistent (survives checkpoint save/load)
      - Python scan uses gradient checkpointing (via PerformerAttentionCore,
        which now auto-selects _python_scan_checkpointed when grad is enabled)
      - num_performer_heads can be changed between phases without re-creating
        the module
    """

    def __init__(self, original_attn, num_performer_heads):
        super().__init__()
        self.head_dim              = original_attn.head_dim
        self.num_heads             = original_attn.config.num_attention_heads
        self.num_key_value_heads   = original_attn.config.num_key_value_heads
        self.num_key_value_groups  = self.num_heads // self.num_key_value_heads
        self.scaling               = self.head_dim ** -0.5
        self.num_performer_heads   = num_performer_heads
        self.num_standard_heads    = self.num_heads - num_performer_heads

        # Reuse pretrained projection weights (shared nn.Linear objects)
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj

        self.performer_core = PerformerAttentionCore(
            head_dim=self.head_dim, num_features=256
        )

        self.config    = original_attn.config
        self.layer_idx = original_attn.layer_idx
        self.is_causal = True

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary(self, q, k, cos, sin):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

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
            causal  = torch.full((N, key_len), torch.finfo(q.dtype).min,
                                 device=q.device, dtype=q.dtype)
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
            Kp    = self.num_performer_heads
            out_p = self.performer_core(q[:, :Kp], k[:, :Kp], v[:, :Kp])

            q_s, k_s, v_s = q[:, Kp:], k[:, Kp:], v[:, Kp:]
            scores = torch.matmul(q_s, k_s.transpose(-2, -1)) * self.scaling
            if attention_mask is not None:
                scores = scores + attention_mask
            w     = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q_s.dtype)
            out_s = torch.matmul(w, v_s)

            attn_out = torch.cat([out_p, out_s], dim=1)

        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, N, -1)
        return self.o_proj(attn_out), None


# ── Model helpers ─────────────────────────────────────────────────────────────

def patch_model(model, num_performer_heads):
    for layer in model.model.layers:
        layer.self_attn = MixedPerformerAttention(
            layer.self_attn, num_performer_heads=num_performer_heads
        )
    return model


def set_performer_heads(model, num_performer_heads):
    num_heads = model.config.num_attention_heads
    for layer in model.model.layers:
        layer.self_attn.num_performer_heads = num_performer_heads
        layer.self_attn.num_standard_heads  = num_heads - num_performer_heads


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_qk(model):
    for layer in model.model.layers:
        layer.self_attn.q_proj.weight.requires_grad = True
        layer.self_attn.k_proj.weight.requires_grad = True


def unfreeze_qkvo(model):
    for layer in model.model.layers:
        for proj in [layer.self_attn.q_proj, layer.self_attn.k_proj,
                     layer.self_attn.v_proj, layer.self_attn.o_proj]:
            proj.weight.requires_grad = True


def count_params(model):
    total    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def save_checkpoint(model, optimizer, scaler, scheduler, step, phase_name, path):
    omegas = {
        f"layer_{i}": layer.self_attn.performer_core.omega.cpu()
        for i, layer in enumerate(model.model.layers)
        if hasattr(layer.self_attn, "performer_core")
    }
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict":    scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step":                 step,
        "phase":                phase_name,
        "omegas":               omegas,
    }, path)
    print(f"  [ckpt] saved {os.path.basename(path)}")


def load_checkpoint(model, optimizer, scaler, scheduler, path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    for i, layer in enumerate(model.model.layers):
        key = f"layer_{i}"
        if key in ckpt["omegas"] and hasattr(layer.self_attn, "performer_core"):
            layer.self_attn.performer_core.omega.copy_(ckpt["omegas"][key].to(DEVICE))
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"  [ckpt] loaded phase={ckpt['phase']}, step={ckpt['step']}")
    return ckpt["step"], ckpt["phase"]


# ── Dataset ───────────────────────────────────────────────────────────────────

class LMDataset(Dataset):
    def __init__(self, token_ids, seq_len, max_samples=None):
        n = (len(token_ids) // seq_len) * seq_len
        data = token_ids[:n].view(-1, seq_len)
        if max_samples is not None:
            data = data[:max_samples]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return {"input_ids": x, "labels": x.clone()}


def _tokenize_split(raw_split, tokenizer):
    tokens = []
    for text in raw_split["text"]:
        if text.strip():
            tokens.extend(tokenizer.encode(text, add_special_tokens=False))
    return torch.tensor(tokens, dtype=torch.long)


def build_dataloaders(tokenizer):
    print("Loading WikiText-103...")
    wt  = load_dataset("wikitext", "wikitext-103-raw-v1")
    print("Loading C4 validation split (streaming)...")
    c4_val_raw = load_dataset("allenai/c4", "en", split="validation", streaming=True)

    train_tokens = _tokenize_split(wt["train"],      tokenizer)
    val_wt_tokens = _tokenize_split(wt["validation"], tokenizer)

    # C4 val: take first MAX_VAL_SAMPLES * SEQ_LEN tokens
    c4_tokens = []
    needed = MAX_VAL_SAMPLES * SEQ_LEN + SEQ_LEN
    for example in c4_val_raw:
        c4_tokens.extend(tokenizer.encode(example["text"], add_special_tokens=False))
        if len(c4_tokens) >= needed:
            break
    c4_tokens = torch.tensor(c4_tokens, dtype=torch.long)

    print(f"WikiText-103 train tokens : {len(train_tokens):,}")
    print(f"WikiText-103 val tokens   : {len(val_wt_tokens):,}")
    print(f"C4 val tokens collected   : {len(c4_tokens):,}")

    train_ds  = LMDataset(train_tokens,  SEQ_LEN, MAX_TRAIN_SAMPLES)
    val_wt_ds = LMDataset(val_wt_tokens, SEQ_LEN, MAX_VAL_SAMPLES)
    val_c4_ds = LMDataset(c4_tokens,     SEQ_LEN, MAX_VAL_SAMPLES)

    print(f"Train samples  : {len(train_ds)}")
    print(f"Val WT samples : {len(val_wt_ds)}")
    print(f"Val C4 samples : {len(val_c4_ds)}")

    train_loader  = DataLoader(train_ds,  batch_size=MICRO_BATCH, shuffle=True,  drop_last=True,  num_workers=2, pin_memory=True)
    val_wt_loader = DataLoader(val_wt_ds, batch_size=MICRO_BATCH, shuffle=False, drop_last=False, num_workers=2, pin_memory=True)
    val_c4_loader = DataLoader(val_c4_ds, batch_size=MICRO_BATCH, shuffle=False, drop_last=False, num_workers=2, pin_memory=True)

    return train_loader, val_wt_loader, val_c4_loader


# ── Training ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def _teacher_ppl(teacher, loader):
    """Perplexity of the frozen teacher model. Does not touch train/eval mode."""
    total_ce, n_tokens = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        labels    = batch["labels"].to(DEVICE)
        with torch.amp.autocast("cuda", dtype=DTYPE):
            out = teacher(input_ids=input_ids, use_cache=False)
        logits = out.logits[:, :-1].float()
        tgt    = labels[:, 1:].contiguous()
        ce     = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="sum")
        total_ce += ce.item()
        n_tokens += tgt.numel()
    return math.exp(min(total_ce / n_tokens, 20))


@torch.no_grad()
def evaluate(student, teacher, loader, label):
    student.eval()
    total_ce, total_kl, n_tokens = 0.0, 0.0, 0

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        labels    = batch["labels"].to(DEVICE)

        with torch.amp.autocast("cuda", dtype=DTYPE):
            s_out = student(input_ids=input_ids, use_cache=False)
            t_out = teacher(input_ids=input_ids, use_cache=False)

        s_logits = s_out.logits[:, :-1].float()
        t_logits = t_out.logits[:, :-1].float()
        tgt      = labels[:, 1:].contiguous()

        ce = F.cross_entropy(s_logits.reshape(-1, s_logits.size(-1)),
                             tgt.reshape(-1), reduction="sum")
        s_lp = F.log_softmax(s_logits / TEMPERATURE, dim=-1)
        t_p  = F.softmax(t_logits / TEMPERATURE, dim=-1)
        kl   = F.kl_div(s_lp, t_p, reduction="sum") * (TEMPERATURE ** 2)

        n          = tgt.numel()
        total_ce  += ce.item()
        total_kl  += kl.item()
        n_tokens  += n

    ppl    = math.exp(min(total_ce / n_tokens, 20))
    avg_kl = total_kl / n_tokens
    student.train()
    return ppl, avg_kl


def run_phase(phase_idx, student, teacher,
              train_loader, val_wt_loader, val_c4_loader,
              optimizer, scaler,
              num_performer_heads, unfreeze_mode, lr, num_epochs,
              phase_name, resume_step=0):

    set_performer_heads(student, num_performer_heads)
    freeze_all(student)
    if unfreeze_mode == "qk":
        unfreeze_qk(student)
    else:
        unfreeze_qkvo(student)

    total, trainable = count_params(student)
    print(f"\n{'='*70}")
    print(f"Phase {phase_idx+1}: {phase_name}")
    print(f"Performer heads : {num_performer_heads}/32")
    print(f"Trainable       : {trainable/1e6:.0f}M / {total/1e6:.0f}M ({100*trainable/total:.1f}%)")
    print(f"LR              : {lr}  |  Epochs: {num_epochs}")
    print(f"{'='*70}\n")

    # Update optimizer param groups to match the newly unfrozen params.
    # We do NOT recreate the optimizer — existing momentum states are preserved
    # for params that were already being trained in a previous phase.
    trainable_params = get_trainable_params(student)
    # Replace param groups (keeps optimizer buffers for params that persist)
    optimizer.param_groups.clear()
    optimizer.add_param_group({"params": trainable_params, "lr": lr})

    total_steps  = (len(train_loader) * num_epochs) // GRAD_ACCUM
    warmup_steps = max(total_steps // 20, 10)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Fast-forward scheduler if resuming mid-phase
    for _ in range(resume_step // GRAD_ACCUM):
        scheduler.step()

    student.train()
    global_step = resume_step
    best_ppl_wt = float("inf")
    log_interval = 50
    t0 = time.time()

    # Steps per epoch (full passes over the loader)
    steps_per_epoch = len(train_loader) // GRAD_ACCUM
    # Skip epochs already completed when resuming mid-phase
    start_epoch = resume_step // steps_per_epoch if steps_per_epoch > 0 else 0

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = epoch_ce = epoch_kl = 0.0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            labels    = batch["labels"].to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=DTYPE):
                s_out = student(input_ids=input_ids, use_cache=False)
                with torch.no_grad():
                    t_out = teacher(input_ids=input_ids, use_cache=False)

            s_logits = s_out.logits[:, :-1].float()
            t_logits = t_out.logits[:, :-1].float()
            tgt      = labels[:, 1:].contiguous()

            ce_loss = F.cross_entropy(
                s_logits.reshape(-1, s_logits.size(-1)), tgt.reshape(-1)
            )
            s_lp    = F.log_softmax(s_logits / TEMPERATURE, dim=-1)
            t_p     = F.softmax(t_logits / TEMPERATURE, dim=-1)
            kl_loss = F.kl_div(s_lp, t_p, reduction="batchmean") * (TEMPERATURE ** 2)

            loss = (ALPHA * ce_loss + (1 - ALPHA) * kl_loss) / GRAD_ACCUM

            scaler.scale(loss).backward()

            epoch_ce   += ce_loss.item()
            epoch_kl   += kl_loss.item()
            epoch_loss += loss.item() * GRAD_ACCUM

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % log_interval == 0:
                    n = batch_idx + 1
                    print(f"  step {global_step:>5d} | "
                          f"loss {epoch_loss/n:.4f} | "
                          f"CE {epoch_ce/n:.4f} | "
                          f"KL {epoch_kl/n:.4f} | "
                          f"lr {scheduler.get_last_lr()[0]:.2e} | "
                          f"scale {scaler.get_scale():.0f} | "
                          f"{time.time()-t0:.0f}s")

                if global_step % EVAL_STEPS == 0:
                    ppl_wt, kl_wt = evaluate(student, teacher, val_wt_loader, "WT103")
                    ppl_c4, kl_c4 = evaluate(student, teacher, val_c4_loader, "C4")
                    print(f"  >> eval step {global_step} | "
                          f"WT103 ppl={ppl_wt:.2f} kl={kl_wt:.4f} | "
                          f"C4 ppl={ppl_c4:.2f} kl={kl_c4:.4f}")
                    if ppl_wt < best_ppl_wt:
                        best_ppl_wt = ppl_wt
                        save_checkpoint(student, optimizer, scaler, scheduler,
                                        global_step, phase_name,
                                        os.path.join(CKPT_DIR, f"best_{phase_name}.pt"))
                    student.train()

                if global_step % SAVE_STEPS == 0:
                    save_checkpoint(student, optimizer, scaler, scheduler,
                                    global_step, phase_name,
                                    os.path.join(CKPT_DIR, f"{phase_name}_step{global_step}.pt"))

        # End of epoch eval
        ppl_wt, kl_wt = evaluate(student, teacher, val_wt_loader, "WT103")
        ppl_c4, kl_c4 = evaluate(student, teacher, val_c4_loader, "C4")
        print(f"\n  Epoch {epoch+1}/{num_epochs} | "
              f"WT103 ppl={ppl_wt:.2f} | C4 ppl={ppl_c4:.2f} | "
              f"best WT103={best_ppl_wt:.2f}")
        if ppl_wt < best_ppl_wt:
            best_ppl_wt = ppl_wt
            save_checkpoint(student, optimizer, scaler, scheduler,
                            global_step, phase_name,
                            os.path.join(CKPT_DIR, f"best_{phase_name}.pt"))

    print(f"\n  Phase {phase_name} complete. Best WT103 ppl: {best_ppl_wt:.2f}")
    return best_ppl_wt, global_step


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--start_phase", type=int, default=0,
                        help="Phase index to start from (0-3)")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataloaders
    train_loader, val_wt_loader, val_c4_loader = build_dataloaders(tokenizer)

    # Load teacher (frozen)
    print("\nLoading teacher model...")
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Load student and patch
    print("Loading student model...")
    student = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)
    student = patch_model(student, num_performer_heads=PHASES[args.start_phase][0])

    # Freeze everything initially
    freeze_all(student)

    # VRAM report
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved  = torch.cuda.memory_reserved() / 1e9
    print(f"\nVRAM after model load: {allocated:.1f} GB allocated / {reserved:.1f} GB reserved")

    # Build optimizer — single instance for the entire training run
    if _HAS_BNB:
        optimizer = bnb.optim.AdamW8bit(
            [p for p in student.parameters()],  # param groups updated per phase
            lr=PHASES[args.start_phase][2],
            weight_decay=0.01,
        )
        print("Optimizer: AdamW8bit (bitsandbytes)")
    else:
        optimizer = torch.optim.AdamW(
            [p for p in student.parameters()],
            lr=PHASES[args.start_phase][2],
            weight_decay=0.01,
        )
        print("Optimizer: AdamW fp32")

    scaler = torch.amp.GradScaler("cuda")

    # Baseline eval before any training
    print("\nBaseline evaluation (before fine-tuning)...")
    set_performer_heads(student, PHASES[0][0])
    ppl_wt_base, _ = evaluate(student, teacher, val_wt_loader, "WT103")
    ppl_c4_base, _ = evaluate(student, teacher, val_c4_loader, "C4")
    print(f"Baseline (4 heads, no training): WT103 ppl={ppl_wt_base:.2f} | C4 ppl={ppl_c4_base:.2f}")

    # Teacher baseline — compute directly, don't use evaluate() which calls .train() on its first arg
    teacher_ppl_wt = _teacher_ppl(teacher, val_wt_loader)
    teacher_ppl_c4 = _teacher_ppl(teacher, val_c4_loader)
    print(f"Teacher  (softmax baseline):     WT103 ppl={teacher_ppl_wt:.2f} | C4 ppl={teacher_ppl_c4:.2f}")

    # Resume handling — scheduler is built inside run_phase, so we only restore
    # model + optimizer + scaler state here. Scheduler state is intentionally
    # not restored: run_phase fast-forwards it cleanly from global_step.
    resume_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        for i, layer in enumerate(student.model.layers):
            key = f"layer_{i}"
            if key in ckpt["omegas"] and hasattr(layer.self_attn, "performer_core"):
                layer.self_attn.performer_core.omega.copy_(ckpt["omegas"][key].to(DEVICE))
        student.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        resume_step = ckpt["step"]
        print(f"  [resume] loaded phase={ckpt['phase']}, step={resume_step}")

    # Training loop across phases
    results = {
        "teacher": {"wt103": teacher_ppl_wt, "c4": teacher_ppl_c4},
        "baseline_4heads": {"wt103": ppl_wt_base, "c4": ppl_c4_base},
    }

    global_step = resume_step
    for phase_idx, (n_heads, unfreeze_mode, lr, n_epochs) in enumerate(PHASES):
        if phase_idx < args.start_phase:
            continue

        phase_name = f"phase{phase_idx+1}_K{n_heads}_{unfreeze_mode.upper()}"

        best_ppl, global_step = run_phase(
            phase_idx, student, teacher,
            train_loader, val_wt_loader, val_c4_loader,
            optimizer, scaler,
            n_heads, unfreeze_mode, lr, n_epochs,
            phase_name,
            resume_step=global_step if phase_idx == args.start_phase else 0,
        )

        results[phase_name] = {"best_wt103_ppl": best_ppl}

        # Save end-of-phase summary
        with open(os.path.join(CKPT_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    # Final summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Teacher  (softmax):            WT103={teacher_ppl_wt:.2f}  C4={teacher_ppl_c4:.2f}")
    print(f"  Baseline (4 heads, no FT):     WT103={ppl_wt_base:.2f}  C4={ppl_c4_base:.2f}")
    for k, v in results.items():
        if k.startswith("phase"):
            print(f"  {k:<35} WT103={v['best_wt103_ppl']:.2f}")
    print(f"\nAll checkpoints in: {CKPT_DIR}")
    print(f"Results JSON      : {os.path.join(CKPT_DIR, 'results.json')}")


if __name__ == "__main__":
    main()
