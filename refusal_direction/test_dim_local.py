"""
Local validation of DIM pipeline stages 1-3 (Mac/MPS-safe).

Runs everything EXCEPT the expensive direction selection and evaluation steps.
Validates: dataset loading, refusal score filtering, activation extraction,
and mean-diff direction computation.

Usage (from repo root):
    export SAVE_DIR="results"
    export DIM_DIR="dim_directions"
    python test_dim_local.py --model_path Qwen/Qwen2.5-3B-Instruct

Expected time: ~10-20 min on Apple Silicon (mostly forward passes for filtering).
"""

import torch
import random
import json
import os
import sys
import argparse
import time
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.load_dataset import load_dataset_split
from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import get_refusal_scores


def parse_arguments():
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"), override=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--n_filter_test', type=int, default=32,
                        help='Number of prompts for quick filtering test')
    parser.add_argument('--full_filter', action='store_true',
                        help='Run filtering on full dataset (~10 min on Mac)')
    parser.add_argument('--skip_directions', action='store_true',
                        help='Skip direction generation (just test filtering)')
    return parser.parse_args()


def main():
    args = parse_arguments()
    t0 = time.time()

    model_alias = os.path.basename(args.model_path)
    cfg = Config(model_alias=model_alias, model_path=args.model_path)

    # ==================================================================
    # Stage 0: Load model
    # ==================================================================
    print("\n" + "=" * 60)
    print("STAGE 0: Loading model")
    print("=" * 60)
    model_base = construct_model_base(cfg.model_path)
    print(f"  Model loaded: {cfg.model_path}")
    print(f"  Device: {next(model_base.model.parameters()).device}")
    print(f"  Dtype: {next(model_base.model.parameters()).dtype}")
    print(f"  Block modules: {len(model_base.model_block_modules)} layers")

    # ==================================================================
    # Stage 1: Load datasets
    # ==================================================================
    print("\n" + "=" * 60)
    print("STAGE 1: Loading datasets")
    print("=" * 60)

    random.seed(42)
    harmful_train = load_dataset_split(harmtype='harmful', split='train', instructions_only=True)
    harmless_train = load_dataset_split(harmtype='harmless', split='train', instructions_only=True)[:len(harmful_train)]
    harmful_val = load_dataset_split(harmtype='harmful', split='val', instructions_only=True)
    harmless_val = load_dataset_split(harmtype='harmless', split='val', instructions_only=True)[:len(harmful_val)]

    print(f"  harmful_train:  {len(harmful_train)}")
    print(f"  harmless_train: {len(harmless_train)}")
    print(f"  harmful_val:    {len(harmful_val)}")
    print(f"  harmless_val:   {len(harmless_val)}")
    print(f"\n  Sample harmful:  '{harmful_train[0][:80]}...'")
    print(f"  Sample harmless: '{harmless_train[0][:80]}...'")

    # ==================================================================
    # Stage 2: Test refusal score filtering
    # ==================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: Testing refusal score filtering")
    print("=" * 60)

    if args.full_filter:
        n_test = len(harmful_train)
        print(f"  Running on full training set ({n_test} examples)...")
    else:
        n_test = min(args.n_filter_test, len(harmful_train))
        print(f"  Running on subset ({n_test} examples, use --full_filter for all)...")

    harmful_subset = harmful_train[:n_test]
    harmless_subset = harmless_train[:n_test]

    print(f"\n  Computing refusal scores on {n_test} harmful prompts...")
    t1 = time.time()
    harmful_scores = get_refusal_scores(
        model_base.model,
        harmful_subset,
        model_base.tokenize_instructions_fn,
        model_base.refusal_toks
    )
    t2 = time.time()
    print(f"  Done in {t2-t1:.1f}s")

    print(f"  Computing refusal scores on {n_test} harmless prompts...")
    harmless_scores = get_refusal_scores(
        model_base.model,
        harmless_subset,
        model_base.tokenize_instructions_fn,
        model_base.refusal_toks
    )
    t3 = time.time()
    print(f"  Done in {t3-t2:.1f}s")

    n_harmful_kept = sum(1 for s in harmful_scores.tolist() if s > 0)
    n_harmless_kept = sum(1 for s in harmless_scores.tolist() if s < 0)

    print(f"\n  Harmful  kept: {n_harmful_kept}/{n_test} ({100*n_harmful_kept/n_test:.1f}%) -- model refuses these")
    print(f"  Harmless kept: {n_harmless_kept}/{n_test} ({100*n_harmless_kept/n_test:.1f}%) -- model doesn't refuse these")

    if n_harmful_kept / n_test < 0.5:
        print("\n  WARNING: Model refuses less than half of harmful prompts.")
        print("  Check: refusal token IDs, chat template, or model safety tuning.")
    elif n_harmful_kept / n_test > 0.9:
        print("\n  Good: model refuses most harmful prompts.")

    if n_harmless_kept / n_test < 0.5:
        print("  WARNING: Model refuses more than half of harmless prompts (over-refusal).")
    elif n_harmless_kept / n_test > 0.9:
        print("  Good: model answers most harmless prompts.")

    print(f"\n  Score statistics:")
    print(f"    Harmful  -- mean: {harmful_scores.mean():.3f}, min: {harmful_scores.min():.3f}, max: {harmful_scores.max():.3f}")
    print(f"    Harmless -- mean: {harmless_scores.mean():.3f}, min: {harmless_scores.min():.3f}, max: {harmless_scores.max():.3f}")

    if args.skip_directions:
        elapsed = time.time() - t0
        print(f"\n  Skipping direction generation (--skip_directions). Done in {elapsed:.0f}s.")
        return

    # ==================================================================
    # Stage 3: Generate candidate directions
    # ==================================================================
    print("\n" + "=" * 60)
    print("STAGE 3: Generating candidate directions (mean-diff)")
    print("=" * 60)

    if args.full_filter:
        harmful_filtered = [inst for inst, score in zip(harmful_train, harmful_scores.tolist()) if score > 0]
        harmless_filtered = [inst for inst, score in zip(harmless_train, harmless_scores.tolist()) if score < 0][:len(harmful_filtered)]
    else:
        harmful_filtered = harmful_train
        harmless_filtered = harmless_train
        print("  Note: using unfiltered data (subset filtering not meaningful for directions)")

    print(f"  Using {len(harmful_filtered)} harmful and {len(harmless_filtered)} harmless prompts")

    artifact_dir = os.path.join(cfg.artifact_path(), "generate_directions")
    os.makedirs(artifact_dir, exist_ok=True)

    print(f"  Forward passes on {len(harmful_filtered) + len(harmless_filtered)} prompts...")
    print(f"  On Mac/MPS: ~10-30 min. On A100: ~5-10 min.")
    print(f"  Saving to: {artifact_dir}")

    t4 = time.time()
    mean_diffs = generate_directions(
        model_base,
        harmful_filtered,
        harmless_filtered,
        artifact_dir=artifact_dir
    )
    t5 = time.time()
    print(f"\n  Direction generation complete in {t5-t4:.1f}s")

    # Inspect output
    if isinstance(mean_diffs, dict):
        print(f"  Output: dict with {len(mean_diffs)} candidate directions")
        for i, (key, val) in enumerate(mean_diffs.items()):
            if i < 5:
                print(f"    Key: {key}, Shape: {val.shape}, Norm: {val.float().norm():.3f}")
            elif i == 5:
                print(f"    ... ({len(mean_diffs) - 5} more)")
                break
        n_candidates = len(mean_diffs)
    elif isinstance(mean_diffs, torch.Tensor):
        print(f"  Output: tensor, Shape: {mean_diffs.shape}")
        n_candidates = mean_diffs.shape[0] if mean_diffs.dim() > 1 else 1
    else:
        print(f"  Output type: {type(mean_diffs)}")
        n_candidates = "?"

    save_path = os.path.join(artifact_dir, 'mean_diffs.pt')
    torch.save(mean_diffs, save_path)
    print(f"  Saved to: {save_path}")

    # ==================================================================
    # Summary
    # ==================================================================
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"LOCAL VALIDATION COMPLETE -- {elapsed:.0f}s total")
    print("=" * 60)
    print(f"""
  Dataset loading:           OK
  Refusal score computation: OK
  Direction generation:      OK ({n_candidates} candidates)
  
  Skipped (expensive): direction selection, completion generation, evaluation.
  These test ~{n_candidates} candidates x 128 val prompts each -> run on Della.

  Next: sbatch run_dim_della.slurm
""")


if __name__ == "__main__":
    main()
