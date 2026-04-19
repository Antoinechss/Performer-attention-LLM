"""
Compare FAVOR+ and FAVOR# against exact softmax attention on a curated prompt suite.

This script focuses on quality metrics, plus a divergence study on synthetic and
model-extracted Q/K/V cases.
"""
import gc
import os
import sys

import torch

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from favor_analysis_utils import (
    aggregate_prompt_metrics,
    benjamini_hochberg,
    build_synthetic_qk_cases,
    collect_model_qk_cases,
    collect_reference_trace,
    compare_model_to_reference,
    evaluate_divergence_cases,
    load_patched_model,
    load_reference_model_and_tokenizer,
    seed_everything,
)


# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE_MAP = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
SEED = 42

NUM_PERFORMER_HEADS = 4
NUM_FEATURES = 256
MAX_NEW_TOKENS = 30
PROMPT_TEST_NUM_DRAWS = 2048
PROMPT_TEST_NUM_TRIALS = 256
PROMPT_TEST_ALPHA = 0.05

PROMPTS = [
    "<|user|>\nWhat is the capital of Japan?</s>\n<|assistant|>\n",
    "<|user|>\nSummarize the benefits of regular exercise in two sentences.</s>\n<|assistant|>\n",
    "<|user|>\nWrite a polite email asking for a deadline extension on a project.</s>\n<|assistant|>\n",
    "<|user|>\nTranslate this to French: The weather is beautiful today.</s>\n<|assistant|>\n",
    "<|user|>\nExplain recursion to a beginner using a simple analogy.</s>\n<|assistant|>\n",
    "<|user|>\nWrite Python code for a function that checks if a string is a palindrome.</s>\n<|assistant|>\n",
    "<|user|>\nGive me three creative names for a coffee shop.</s>\n<|assistant|>\n",
    "<|user|>\nIf a train leaves at 3 PM and arrives 2 hours 45 minutes later, what time does it arrive?</s>\n<|assistant|>\n",
    "<|user|>\nList four safety tips for hiking in the mountains.</s>\n<|assistant|>\n",
    "<|user|>\nCompare solar and wind energy in three concise bullet points.</s>\n<|assistant|>\n",
]

RUN_PROMPT_BENCHMARK = True
RUN_DIVERGENCE_STUDY = True

DIVERGENCE_SYNTH_SEQ_LEN = 64
DIVERGENCE_SYNTH_HEAD_DIM = 64
DIVERGENCE_MODEL_PROMPTS = 3
DIVERGENCE_LAYER_INDICES = [0, 1, 2, 4, 8, 16, 24, 31]
DIVERGENCE_HEAD_INDICES = [0, 1, 2, 3]
MC_NUM_DRAWS = 2048
MC_NUM_TRIALS = 256
DIVERGENCE_ALPHA = 0.05
TOP_CASES_TO_PRINT = 5
# ────────────────────────────────────────────────────────────────────────────


def _resolve_device():
    if DEVICE_MAP == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def _method_label(feature_map: str) -> str:
    return "FAVOR+" if feature_map == "favor_plus" else "FAVOR#"


def _release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_prompt_table(prompt_results):
    print(f"\n{'=' * 154}")
    print("Prompt-level quality metrics against exact softmax")
    print(f"{'=' * 154}")
    header = (
        f"{'Prompt':>6}  {'Method':>7}  {'KL':>8}  {'JS':>8}  {'Top1':>8}"
        f"  {'p(ref)':>8}  {'TokMatch':>10}  {'Prefix':>8}  {'p_value':>9}"
        f"  {'q_value':>9}  {'Reject':>6}"
    )
    print(header)
    print("-" * len(header))

    prompt_count = len(next(iter(prompt_results.values())))
    for prompt_idx in range(prompt_count):
        for feature_map in ["favor_plus", "favor_sharp"]:
            row = prompt_results[feature_map][prompt_idx]
            print(
                f"{prompt_idx:>6}  {_method_label(feature_map):>7}"
                f"  {row['avg_kl']:>8.4f}  {row['avg_js']:>8.4f}"
                f"  {row['top1_match_rate'] * 100:>7.2f}%"
                f"  {row['p_reference_top1'] * 100:>7.2f}%"
                f"  {row['token_match_rate'] * 100:>9.2f}%"
                f"  {row['prefix_match_rate'] * 100:>7.2f}%"
                f"  {row['prompt_p_value']:>9.4f}"
                f"  {row['prompt_q_value']:>9.4f}"
                f"  {str(row['reject_prompt_null']):>6}"
            )
        print("-" * len(header))


def print_aggregate_table(prompt_results):
    print(f"\n{'=' * 104}")
    print("Aggregate summary across prompts")
    print(f"{'=' * 104}")
    header = (
        f"{'Method':>7}  {'KL':>8}  {'JS':>8}  {'Top1':>8}"
        f"  {'p(ref)':>8}  {'TokMatch':>10}  {'Prefix':>8}  {'Rejects':>8}"
    )
    print(header)
    print("-" * len(header))
    for feature_map in ["favor_plus", "favor_sharp"]:
        summary = aggregate_prompt_metrics(prompt_results[feature_map])
        reject_count = sum(int(row["reject_prompt_null"]) for row in prompt_results[feature_map])
        print(
            f"{_method_label(feature_map):>7}"
            f"  {summary['avg_kl']:>8.4f}  {summary['avg_js']:>8.4f}"
            f"  {summary['top1_match_rate'] * 100:>7.2f}%"
            f"  {summary['p_reference_top1'] * 100:>7.2f}%"
            f"  {summary['token_match_rate'] * 100:>9.2f}%"
            f"  {summary['prefix_match_rate'] * 100:>7.2f}%"
            f"  {reject_count:>8}"
        )


def print_prompt_generations(reference_traces, prompt_results):
    print(f"\n{'=' * 120}")
    print("Generated text by prompt and method")
    print(f"{'=' * 120}")
    for prompt_idx, reference_trace in enumerate(reference_traces):
        print(f"\nPrompt {prompt_idx}")
        print(f"User prompt: {reference_trace['prompt'].replace(chr(10), ' ')}")
        print(f"Exact softmax: {reference_trace['generated_text']}")
        print(f"FAVOR+:      {prompt_results['favor_plus'][prompt_idx]['greedy_generated_text']}")
        print(f"FAVOR#:      {prompt_results['favor_sharp'][prompt_idx]['greedy_generated_text']}")


def print_prompt_hypothesis():
    print(f"\n{'=' * 120}")
    print("Prompt-level hypothesis test")
    print(f"{'=' * 120}")
    print(
        "H0: for a given prompt, the approximate method and exact softmax produce the same next-token "
        "distribution at each decoding step, up to Monte Carlo sampling noise at the evaluation resolution."
    )
    print("Statistic: mean Jensen-Shannon divergence across teacher-forced decoding steps.")
    print(
        f"Decision rule: reject H0 when BH-corrected q_value <= {PROMPT_TEST_ALPHA:.2f}. "
        "Small p_value means the prompt-level distributional mismatch is unlikely under H0."
    )


def print_divergence_results(title, results):
    print(f"\n{'=' * 132}")
    print(title)
    print(f"{'=' * 132}")
    header = (
        f"{'Rank':>4}  {'Pool':>9}  {'Case':>28}  {'Layer':>5}  {'Head':>4}  {'Row':>4}"
        f"  {'JS':>10}  {'RelErr':>10}  {'p':>9}  {'q':>9}  {'Reject':>6}"
    )
    print(header)
    print("-" * len(header))
    for rank, row in enumerate(results[:TOP_CASES_TO_PRINT], start=1):
        case_name = row.get("case", f"prompt{row.get('prompt_index', '-')}")
        layer = row.get("layer", "-")
        head = row.get("head", "-")
        print(
            f"{rank:>4}  {row['pool']:>9}  {case_name[:28]:>28}"
            f"  {str(layer):>5}  {str(head):>4}  {row['worst_row']:>4}"
            f"  {row['max_row_js']:>10.6f}  {row['max_row_relative_output_error']:>10.6f}"
            f"  {row['p_value']:>9.4f}  {row['q_value']:>9.4f}  {str(row['reject_null']):>6}"
        )


def run_prompt_benchmark():
    device = _resolve_device()

    print(f"\n{'=' * 84}")
    print(f"Loading exact reference model on {DEVICE_MAP} with dtype={DTYPE}")
    print(f"{'=' * 84}")
    reference_model, tokenizer = load_reference_model_and_tokenizer(
        model_id=MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE_MAP,
    )

    print("\nCollecting exact softmax reference traces...")
    reference_traces = [
        collect_reference_trace(
            reference_model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            device=device,
        )
        for prompt in PROMPTS
    ]

    prompt_results = {}
    for feature_map in ["favor_plus", "favor_sharp"]:
        print(f"\nLoading {_method_label(feature_map)} model...")
        approx_model, _ = load_patched_model(
            model_id=MODEL_ID,
            num_performer_heads=NUM_PERFORMER_HEADS,
            num_features=NUM_FEATURES,
            feature_map=feature_map,
            torch_dtype=DTYPE,
            device_map=DEVICE_MAP,
        )
        prompt_results[feature_map] = [
            compare_model_to_reference(
                approx_model,
                tokenizer,
                reference_trace=trace,
                max_new_tokens=MAX_NEW_TOKENS,
                device=device,
                prompt_test_draws=PROMPT_TEST_NUM_DRAWS,
                prompt_test_trials=PROMPT_TEST_NUM_TRIALS,
                prompt_test_seed=SEED,
            )
            for trace in reference_traces
        ]
        q_values, rejected = benjamini_hochberg(
            [row["prompt_p_value"] for row in prompt_results[feature_map]],
            alpha=PROMPT_TEST_ALPHA,
        )
        for idx, row in enumerate(prompt_results[feature_map]):
            row["prompt_q_value"] = q_values[idx]
            row["reject_prompt_null"] = rejected[idx]
        _release_model(approx_model)

    print_prompt_hypothesis()
    print_prompt_table(prompt_results)
    print_aggregate_table(prompt_results)
    print_prompt_generations(reference_traces, prompt_results)
    return reference_model, tokenizer, prompt_results


def run_divergence_study(reference_model, tokenizer):
    device = _resolve_device()

    synthetic_cases = build_synthetic_qk_cases(
        seq_len=DIVERGENCE_SYNTH_SEQ_LEN,
        head_dim=DIVERGENCE_SYNTH_HEAD_DIM,
        seed=SEED,
    )
    for feature_map in ["favor_plus", "favor_sharp"]:
        results = evaluate_divergence_cases(
            synthetic_cases,
            feature_map=feature_map,
            num_features=NUM_FEATURES,
            seed=SEED,
            num_draws=MC_NUM_DRAWS,
            num_trials=MC_NUM_TRIALS,
            alpha=DIVERGENCE_ALPHA,
        )
        print_divergence_results(
            f"Synthetic divergence sweep — {_method_label(feature_map)}",
            results,
        )

    _release_model(reference_model)

    print(f"\n{'=' * 84}")
    print("Loading capture model for real Q/K/V extraction")
    print(f"{'=' * 84}")
    capture_model, _ = load_patched_model(
        model_id=MODEL_ID,
        num_performer_heads=0,
        num_features=NUM_FEATURES,
        feature_map="favor_plus",
        torch_dtype=DTYPE,
        device_map=DEVICE_MAP,
        capture_qkv=True,
    )

    valid_layers = [
        layer_idx
        for layer_idx in DIVERGENCE_LAYER_INDICES
        if layer_idx < len(capture_model.model.layers)
    ]
    model_cases = collect_model_qk_cases(
        capture_model,
        tokenizer,
        prompts=PROMPTS[:DIVERGENCE_MODEL_PROMPTS],
        device=device,
        layer_indices=valid_layers,
        head_indices=DIVERGENCE_HEAD_INDICES,
    )
    _release_model(capture_model)

    for feature_map in ["favor_plus", "favor_sharp"]:
        results = evaluate_divergence_cases(
            model_cases,
            feature_map=feature_map,
            num_features=NUM_FEATURES,
            seed=SEED,
            num_draws=MC_NUM_DRAWS,
            num_trials=MC_NUM_TRIALS,
            alpha=DIVERGENCE_ALPHA,
        )
        print_divergence_results(
            f"Model-extracted divergence sweep — {_method_label(feature_map)}",
            results,
        )


def main():
    seed_everything(SEED)
    reference_model = None
    tokenizer = None

    if RUN_PROMPT_BENCHMARK:
        reference_model, tokenizer, _ = run_prompt_benchmark()
    if RUN_DIVERGENCE_STUDY:
        if reference_model is None or tokenizer is None:
            reference_model, tokenizer = load_reference_model_and_tokenizer(
                model_id=MODEL_ID,
                torch_dtype=DTYPE,
                device_map=DEVICE_MAP,
            )
        run_divergence_study(reference_model, tokenizer)


if __name__ == "__main__":
    main()
