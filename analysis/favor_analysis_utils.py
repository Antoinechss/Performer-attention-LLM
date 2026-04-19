import math
import os
import random
import sys
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PERFORMER_DIR = os.path.join(_PROJECT_ROOT, "performer")
if _PERFORMER_DIR not in sys.path:
    sys.path.insert(0, _PERFORMER_DIR)

from performer_attention import PerformerAttentionCore, _HAS_TRITON


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_feature_map_name(feature_map: str) -> str:
    key = feature_map.lower().strip().replace(" ", "").replace("-", "_")
    aliases = {
        "favor+": "favor_plus",
        "favorplus": "favor_plus",
        "favor_plus": "favor_plus",
        "favor#": "favor_sharp",
        "favorsharp": "favor_sharp",
        "favor_sharp": "favor_sharp",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported feature map '{feature_map}'.")
    return aliases[key]


class MixedPerformerAttention(nn.Module):
    """Wrap HF LlamaAttention and route a subset of heads through Performer."""

    def __init__(
        self,
        original_attn,
        num_performer_heads: int,
        num_features: int = 256,
        feature_map: str = "favor_plus",
        device=None,
        capture_qkv: bool = False,
    ):
        super().__init__()
        self.original = getattr(original_attn, "original", original_attn)
        self.head_dim = self.original.head_dim
        self.num_heads = self.original.config.num_attention_heads
        self.num_key_value_heads = self.original.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.num_performer_heads = num_performer_heads
        self.num_standard_heads = self.num_heads - num_performer_heads
        self.num_features = num_features
        self.feature_map = _normalize_feature_map_name(feature_map)
        self.capture_qkv = capture_qkv
        self.last_capture = None

        self.q_proj = self.original.q_proj
        self.k_proj = self.original.k_proj
        self.v_proj = self.original.v_proj
        self.o_proj = self.original.o_proj

        core_device = self.q_proj.weight.device if device is None else device
        self.performer_core = PerformerAttentionCore(
            head_dim=self.head_dim,
            num_features=self.num_features,
            feature_map=self.feature_map,
        ).to(core_device)

        self.config = self.original.config
        self.layer_idx = self.original.layer_idx
        self.is_causal = True

    def set_num_performer_heads(self, num_performer_heads: int) -> None:
        self.num_performer_heads = num_performer_heads
        self.num_standard_heads = self.num_heads - num_performer_heads

    def clear_capture(self) -> None:
        self.last_capture = None

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

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = self._apply_rotary(q, k, cos, sin)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        if self.capture_qkv:
            self.last_capture = {
                "q": q.detach().float().cpu(),
                "k": k.detach().float().cpu(),
                "v": v.detach().float().cpu(),
            }

        if attention_mask is None and seq_len > 1 and self.num_standard_heads > 0:
            key_len = k.shape[2]
            causal = torch.full(
                (seq_len, key_len),
                torch.finfo(q.dtype).min,
                device=q.device,
                dtype=q.dtype,
            )
            attention_mask = torch.triu(causal, diagonal=key_len - seq_len + 1)[None, None]

        if self.num_standard_heads == 0:
            attn_out = self.performer_core(q, k, v)
        elif self.num_performer_heads == 0:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
            if attention_mask is not None:
                scores = scores + attention_mask
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_out = torch.matmul(weights, v)
        else:
            num_perf = self.num_performer_heads
            out_perf = self.performer_core(q[:, :num_perf], k[:, :num_perf], v[:, :num_perf])
            q_std = q[:, num_perf:]
            k_std = k[:, num_perf:]
            v_std = v[:, num_perf:]
            scores = torch.matmul(q_std, k_std.transpose(-2, -1)) * self.scaling
            if attention_mask is not None:
                scores = scores + attention_mask
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q_std.dtype)
            out_std = torch.matmul(weights, v_std)
            attn_out = torch.cat([out_perf, out_std], dim=1)

        attn_out = attn_out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_out), None


def patch_model_attention_layers(
    model,
    num_performer_heads: int,
    num_features: int = 256,
    feature_map: str = "favor_plus",
    capture_qkv: bool = False,
):
    for layer in model.model.layers:
        layer.self_attn = MixedPerformerAttention(
            layer.self_attn,
            num_performer_heads=num_performer_heads,
            num_features=num_features,
            feature_map=feature_map,
            capture_qkv=capture_qkv,
        )
    return model


def set_performer_heads(model, num_performer_heads: int) -> None:
    for layer in model.model.layers:
        if not isinstance(layer.self_attn, MixedPerformerAttention):
            raise TypeError("Expected model to be patched with MixedPerformerAttention.")
        layer.self_attn.set_num_performer_heads(num_performer_heads)


def clear_attention_captures(model) -> None:
    for layer in model.model.layers:
        if isinstance(layer.self_attn, MixedPerformerAttention):
            layer.self_attn.clear_capture()


def load_reference_model_and_tokenizer(model_id, torch_dtype=torch.float16, device_map="cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    load_kwargs = {"torch_dtype": torch_dtype}
    if device_map in {"cpu", "auto", "balanced", "balanced_low_0", "sequential"}:
        load_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if device_map in {"cuda", "mps"}:
        model = model.to(torch.device(device_map))
    model.eval()
    return model, tokenizer


def load_patched_model(
    model_id,
    num_performer_heads: int,
    num_features: int,
    feature_map: str,
    torch_dtype=torch.float16,
    device_map="cpu",
    capture_qkv: bool = False,
):
    model, tokenizer = load_reference_model_and_tokenizer(
        model_id=model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    patch_model_attention_layers(
        model,
        num_performer_heads=num_performer_heads,
        num_features=num_features,
        feature_map=feature_map,
        capture_qkv=capture_qkv,
    )
    return model, tokenizer


def greedy_generate(model, input_ids, max_new_tokens: int, eos_token_id: Optional[int] = None):
    current_ids = input_ids.clone()
    generated_ids = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=current_ids, use_cache=False)
            logits = outputs.logits[0, -1].float()
            token_id = int(logits.argmax().item())
            generated_ids.append(token_id)
            next_token = torch.tensor([[token_id]], device=current_ids.device)
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            if eos_token_id is not None and token_id == eos_token_id:
                break
    return generated_ids


def collect_reference_trace(model, tokenizer, prompt: str, max_new_tokens: int, device=None):
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    if device is not None:
        prompt_ids = prompt_ids.to(device)

    current_ids = prompt_ids.clone()
    steps = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=current_ids, use_cache=False)
            logits = outputs.logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1)
            token_id = int(logits.argmax().item())
            steps.append(
                {
                    "token_id": token_id,
                    "token": tokenizer.decode([token_id]),
                    "logits": logits.cpu(),
                    "probs": probs.cpu(),
                }
            )
            next_token = torch.tensor([[token_id]], device=current_ids.device)
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break

    generated_ids = [step["token_id"] for step in steps]
    return {
        "prompt": prompt,
        "prompt_ids": prompt_ids.cpu(),
        "steps": steps,
        "generated_ids": generated_ids,
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
    }


def kl_reference_to_approx(reference_probs, approx_probs, eps: float = 1e-8):
    reference_probs = torch.clamp(reference_probs, min=eps)
    approx_probs = torch.clamp(approx_probs, min=eps)
    return float(torch.sum(reference_probs * (torch.log(reference_probs) - torch.log(approx_probs))).item())


def js_divergence(reference_probs, approx_probs, eps: float = 1e-8):
    reference_probs = torch.clamp(reference_probs, min=eps)
    approx_probs = torch.clamp(approx_probs, min=eps)
    midpoint = 0.5 * (reference_probs + approx_probs)
    js = 0.5 * torch.sum(reference_probs * (torch.log(reference_probs) - torch.log(midpoint)))
    js = js + 0.5 * torch.sum(approx_probs * (torch.log(approx_probs) - torch.log(midpoint)))
    return float(js.item())


def longest_common_prefix(reference_ids: Sequence[int], approx_ids: Sequence[int]) -> int:
    limit = min(len(reference_ids), len(approx_ids))
    matched = 0
    while matched < limit and reference_ids[matched] == approx_ids[matched]:
        matched += 1
    return matched


def compare_model_to_reference(
    model,
    tokenizer,
    reference_trace,
    max_new_tokens: int,
    device=None,
    prompt_test_draws: int = 2048,
    prompt_test_trials: int = 512,
    prompt_test_seed: int = 0,
):
    prompt_ids = reference_trace["prompt_ids"]
    if device is not None:
        prompt_ids = prompt_ids.to(device)

    current_ids = prompt_ids.clone()
    metrics = {
        "kl_values": [],
        "js_values": [],
        "top1_match": [],
        "p_reference_top1": [],
    }
    approx_top1_ids = []
    reference_prob_steps = []
    approx_prob_steps = []

    with torch.no_grad():
        for step in reference_trace["steps"]:
            outputs = model(input_ids=current_ids, use_cache=False)
            logits = outputs.logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1).cpu()

            reference_probs = step["probs"]
            reference_top1 = step["token_id"]
            approx_top1 = int(logits.argmax().item())

            metrics["kl_values"].append(kl_reference_to_approx(reference_probs, probs))
            metrics["js_values"].append(js_divergence(reference_probs, probs))
            metrics["top1_match"].append(float(approx_top1 == reference_top1))
            metrics["p_reference_top1"].append(float(probs[reference_top1].item()))
            approx_top1_ids.append(approx_top1)
            reference_prob_steps.append(reference_probs)
            approx_prob_steps.append(probs)

            next_token = torch.tensor([[reference_top1]], device=current_ids.device)
            current_ids = torch.cat([current_ids, next_token], dim=-1)

    greedy_ids = greedy_generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    )
    reference_ids = reference_trace["generated_ids"]
    match_limit = min(len(reference_ids), len(greedy_ids))
    token_match_rate = 0.0
    if reference_ids:
        token_match_rate = sum(
            int(reference_ids[idx] == greedy_ids[idx]) for idx in range(match_limit)
        ) / len(reference_ids)
    prefix_match_rate = 0.0
    if reference_ids:
        prefix_match_rate = longest_common_prefix(reference_ids, greedy_ids) / len(reference_ids)

    prompt_p_value = monte_carlo_mean_js_p_value(
        reference_prob_steps,
        approx_prob_steps,
        num_draws=prompt_test_draws,
        num_trials=prompt_test_trials,
        seed=prompt_test_seed,
    )

    return {
        "prompt": reference_trace["prompt"],
        "avg_kl": float(np.mean(metrics["kl_values"])) if metrics["kl_values"] else 0.0,
        "avg_js": float(np.mean(metrics["js_values"])) if metrics["js_values"] else 0.0,
        "top1_match_rate": float(np.mean(metrics["top1_match"])) if metrics["top1_match"] else 0.0,
        "p_reference_top1": float(np.mean(metrics["p_reference_top1"])) if metrics["p_reference_top1"] else 0.0,
        "token_match_rate": token_match_rate,
        "prefix_match_rate": prefix_match_rate,
        "teacher_forced_top1_ids": approx_top1_ids,
        "greedy_generated_ids": greedy_ids,
        "greedy_generated_text": tokenizer.decode(greedy_ids, skip_special_tokens=True),
        "kl_values": metrics["kl_values"],
        "js_values": metrics["js_values"],
        "prompt_p_value": float(prompt_p_value),
    }


def aggregate_prompt_metrics(results: Iterable[Dict]):
    results = list(results)
    if not results:
        return {}
    keys = [
        "avg_kl",
        "avg_js",
        "top1_match_rate",
        "p_reference_top1",
        "token_match_rate",
        "prefix_match_rate",
    ]
    return {key: float(np.mean([row[key] for row in results])) for key in keys}


def causal_softmax_attention(q, k, v):
    head_dim = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
    seq_len = q.shape[2]
    mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v), weights


def approximate_causal_attention_weights(core: PerformerAttentionCore, q, k):
    scale = q.shape[-1] ** -0.25
    phi_q, phi_k = core.feature_maps(q * scale, k * scale)
    weights = torch.matmul(phi_q.float(), phi_k.float().transpose(-2, -1))
    seq_len = q.shape[2]
    mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
    weights = weights.masked_fill(mask, 0.0)
    return weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)


def approximate_causal_attention(q, k, v, num_features: int, feature_map: str, seed: int = 0):
    seed_everything(seed)
    core = PerformerAttentionCore(
        head_dim=q.shape[-1],
        num_features=num_features,
        feature_map=feature_map,
    ).to(q.device)
    with torch.no_grad():
        out = core(q, k, v)
        weights = approximate_causal_attention_weights(core, q, k)
    return out, weights


def rowwise_js_divergence(exact_weights, approx_weights, eps: float = 1e-8):
    exact_weights = torch.clamp(exact_weights.float(), min=eps)
    approx_weights = torch.clamp(approx_weights.float(), min=eps)
    midpoint = 0.5 * (exact_weights + approx_weights)
    js = 0.5 * torch.sum(
        exact_weights * (torch.log(exact_weights) - torch.log(midpoint)),
        dim=-1,
    )
    js = js + 0.5 * torch.sum(
        approx_weights * (torch.log(approx_weights) - torch.log(midpoint)),
        dim=-1,
    )
    return js


def rowwise_relative_output_error(exact_out, approx_out, eps: float = 1e-8):
    return (approx_out - exact_out).norm(dim=-1) / (exact_out.norm(dim=-1) + eps)


def monte_carlo_js_p_value(
    reference_probs,
    approx_probs,
    num_draws: int = 2048,
    num_trials: int = 512,
    seed: int = 0,
):
    seed_everything(seed)
    reference_probs = reference_probs.float()
    approx_probs = approx_probs.float()
    observed = js_divergence(reference_probs, approx_probs)
    midpoint = 0.5 * (reference_probs + approx_probs)
    midpoint = midpoint / midpoint.sum()

    exceed = 0
    for _ in range(num_trials):
        counts_a = torch.multinomial(midpoint, num_draws, replacement=True)
        counts_b = torch.multinomial(midpoint, num_draws, replacement=True)
        hist_a = torch.bincount(counts_a, minlength=midpoint.numel()).float()
        hist_b = torch.bincount(counts_b, minlength=midpoint.numel()).float()
        trial_ref = hist_a / hist_a.sum().clamp_min(1.0)
        trial_approx = hist_b / hist_b.sum().clamp_min(1.0)
        trial_stat = js_divergence(trial_ref, trial_approx)
        if trial_stat >= observed:
            exceed += 1

    return (1 + exceed) / (num_trials + 1)


def monte_carlo_mean_js_p_value(
    reference_prob_steps: Sequence[torch.Tensor],
    approx_prob_steps: Sequence[torch.Tensor],
    num_draws: int = 2048,
    num_trials: int = 512,
    seed: int = 0,
):
    """Prompt-level Monte Carlo test using the mean JS over decoding steps."""
    if len(reference_prob_steps) != len(approx_prob_steps):
        raise ValueError("reference_prob_steps and approx_prob_steps must have the same length.")
    if not reference_prob_steps:
        return 1.0

    seed_everything(seed)
    observed = float(
        np.mean(
            [
                js_divergence(reference_probs, approx_probs)
                for reference_probs, approx_probs in zip(reference_prob_steps, approx_prob_steps)
            ]
        )
    )

    midpoints = []
    for reference_probs, approx_probs in zip(reference_prob_steps, approx_prob_steps):
        midpoint = 0.5 * (reference_probs.float() + approx_probs.float())
        midpoint = midpoint / midpoint.sum().clamp_min(1e-8)
        midpoints.append(midpoint)

    exceed = 0
    for _ in range(num_trials):
        trial_stats = []
        for midpoint in midpoints:
            counts_a = torch.multinomial(midpoint, num_draws, replacement=True)
            counts_b = torch.multinomial(midpoint, num_draws, replacement=True)
            hist_a = torch.bincount(counts_a, minlength=midpoint.numel()).float()
            hist_b = torch.bincount(counts_b, minlength=midpoint.numel()).float()
            trial_ref = hist_a / hist_a.sum().clamp_min(1.0)
            trial_approx = hist_b / hist_b.sum().clamp_min(1.0)
            trial_stats.append(js_divergence(trial_ref, trial_approx))
        if float(np.mean(trial_stats)) >= observed:
            exceed += 1

    return (1 + exceed) / (num_trials + 1)


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05):
    if not p_values:
        return [], []

    order = np.argsort(p_values)
    ranked = np.array(p_values, dtype=np.float64)[order]
    count = len(ranked)

    adjusted = np.empty(count, dtype=np.float64)
    running = 1.0
    for idx in range(count - 1, -1, -1):
        rank = idx + 1
        candidate = ranked[idx] * count / rank
        running = min(running, candidate)
        adjusted[idx] = min(running, 1.0)

    adjusted_original = np.empty(count, dtype=np.float64)
    adjusted_original[order] = adjusted
    rejected = adjusted_original <= alpha
    return adjusted_original.tolist(), rejected.tolist()


def _worst_row_index(row_scores):
    flat_idx = int(row_scores.reshape(-1).argmax().item())
    batch, head, row = np.unravel_index(flat_idx, row_scores.shape)
    return batch, head, row


def evaluate_divergence_case(
    case: Dict,
    feature_map: str,
    num_features: int,
    seed: int = 0,
    num_draws: int = 2048,
    num_trials: int = 512,
):
    q = case["q"].float()
    k = case["k"].float()
    v = case["v"].float()

    with torch.no_grad():
        exact_out, exact_weights = causal_softmax_attention(q, k, v)
        approx_out, approx_weights = approximate_causal_attention(
            q,
            k,
            v,
            num_features=num_features,
            feature_map=feature_map,
            seed=seed,
        )

    row_js = rowwise_js_divergence(exact_weights, approx_weights)
    row_rel = rowwise_relative_output_error(exact_out, approx_out)
    batch_idx, head_idx, row_idx = _worst_row_index(row_js)
    support = row_idx + 1
    exact_row = exact_weights[batch_idx, head_idx, row_idx, :support].cpu()
    approx_row = approx_weights[batch_idx, head_idx, row_idx, :support].cpu()

    p_value = monte_carlo_js_p_value(
        exact_row,
        approx_row,
        num_draws=num_draws,
        num_trials=num_trials,
        seed=seed,
    )

    result = dict(case["metadata"])
    result.update(
        {
            "feature_map": _normalize_feature_map_name(feature_map),
            "max_row_js": float(row_js.max().item()),
            "avg_row_js": float(row_js.mean().item()),
            "max_row_relative_output_error": float(row_rel.max().item()),
            "avg_row_relative_output_error": float(row_rel.mean().item()),
            "worst_row": int(row_idx),
            "p_value": float(p_value),
        }
    )
    return result


def evaluate_divergence_cases(
    cases: Sequence[Dict],
    feature_map: str,
    num_features: int,
    seed: int = 0,
    num_draws: int = 2048,
    num_trials: int = 512,
    alpha: float = 0.05,
):
    results = [
        evaluate_divergence_case(
            case,
            feature_map=feature_map,
            num_features=num_features,
            seed=seed,
            num_draws=num_draws,
            num_trials=num_trials,
        )
        for case in cases
    ]
    q_values, rejected = benjamini_hochberg([row["p_value"] for row in results], alpha=alpha)
    for idx, row in enumerate(results):
        row["q_value"] = q_values[idx]
        row["reject_null"] = rejected[idx]
    return sorted(
        results,
        key=lambda row: (
            -row["max_row_js"],
            -row["max_row_relative_output_error"],
            row["p_value"],
        ),
    )


def build_synthetic_qk_cases(seq_len: int = 64, head_dim: int = 64, seed: int = 0):
    seed_everything(seed)

    def wrap_case(name, q, k, v):
        return {
            "q": q.unsqueeze(0).unsqueeze(0).float(),
            "k": k.unsqueeze(0).unsqueeze(0).float(),
            "v": v.unsqueeze(0).unsqueeze(0).float(),
            "metadata": {"pool": "synthetic", "case": name},
        }

    base_v = torch.randn(seq_len, head_dim)
    cases = []

    q = torch.randn(seq_len, head_dim)
    k = torch.randn(seq_len, head_dim)
    cases.append(wrap_case("isotropic_gaussian", q, k, base_v))

    q = 0.35 * torch.randn(seq_len, head_dim)
    k = 1.75 * torch.randn(seq_len, head_dim)
    cases.append(wrap_case("norm_mismatched_gaussian", q, k, base_v))

    scales = torch.logspace(-1, 1, head_dim)
    q = torch.randn(seq_len, head_dim) * scales
    k = torch.randn(seq_len, head_dim) * torch.flip(scales, dims=[0])
    cases.append(wrap_case("anisotropic_diagonal_covariance", q, k, base_v))

    basis = F.normalize(torch.randn(head_dim), dim=0)
    q = basis.unsqueeze(0).repeat(seq_len, 1) + 0.03 * torch.randn(seq_len, head_dim)
    k = basis.unsqueeze(0).repeat(seq_len, 1) + 0.03 * torch.randn(seq_len, head_dim)
    cases.append(wrap_case("near_colinear", q, k, base_v))

    q = torch.randn(seq_len, head_dim)
    k = 0.15 * torch.randn(seq_len, head_dim)
    k[-1] = 6.0 * F.normalize(q[-1], dim=0)
    cases.append(wrap_case("one_dominant_key", q, k, base_v))

    q = torch.cat(
        [
            0.25 * torch.randn(seq_len // 2, head_dim),
            2.5 * torch.randn(seq_len - seq_len // 2, head_dim),
        ],
        dim=0,
    )
    k = torch.cat(
        [
            2.5 * torch.randn(seq_len // 2, head_dim),
            0.25 * torch.randn(seq_len - seq_len // 2, head_dim),
        ],
        dim=0,
    )
    cases.append(wrap_case("heterogeneous_mixed_scale", q, k, base_v))

    return cases


def collect_model_qk_cases(
    model,
    tokenizer,
    prompts: Sequence[str],
    device=None,
    layer_indices: Optional[Sequence[int]] = None,
    head_indices: Optional[Sequence[int]] = None,
):
    if not all(isinstance(layer.self_attn, MixedPerformerAttention) for layer in model.model.layers):
        raise TypeError("Model must be patched with MixedPerformerAttention and capture enabled.")

    num_layers = len(model.model.layers)
    selected_layers = list(range(num_layers)) if layer_indices is None else list(layer_indices)
    cases = []

    for prompt_idx, prompt in enumerate(prompts):
        clear_attention_captures(model)
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        if device is not None:
            prompt_ids = prompt_ids.to(device)

        with torch.no_grad():
            model(input_ids=prompt_ids, use_cache=False)

        for layer_idx in selected_layers:
            capture = model.model.layers[layer_idx].self_attn.last_capture
            if capture is None:
                continue

            q = capture["q"][0]
            k = capture["k"][0]
            v = capture["v"][0]
            selected_heads = list(range(q.shape[0])) if head_indices is None else list(head_indices)

            for head_idx in selected_heads:
                cases.append(
                    {
                        "q": q[head_idx : head_idx + 1].unsqueeze(0).float(),
                        "k": k[head_idx : head_idx + 1].unsqueeze(0).float(),
                        "v": v[head_idx : head_idx + 1].unsqueeze(0).float(),
                        "metadata": {
                            "pool": "model",
                            "prompt_index": prompt_idx,
                            "prompt": prompt,
                            "layer": int(layer_idx),
                            "head": int(head_idx),
                        },
                    }
                )

    return cases
