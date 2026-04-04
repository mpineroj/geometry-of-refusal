"""
Reconstruct datasets for the geometry-of-refusal pipeline.

Creates the following files (relative to repo root):
  data/saladbench_splits/harmful_train.json   (1184 harmful instructions)
  data/saladbench_splits/harmful_val.json      (128 harmful instructions)
  data/saladbench_splits/harmless_train.json   (1184 harmless instructions)
  data/saladbench_splits/harmless_val.json     (128 harmless instructions)
  data/saladbench_splits/harmless_test.json    (128 harmless instructions)
  dataset/processed/jailbreakbench.json        (JailbreakBench eval set)

Based on Section A.1 of Wollschläger et al. (2025):
  - Harmful: SALAD-BENCH, excluding Multilingual and ToxicChat sources,
    up to 256 per remaining source, then split 1184/128.
  - Harmless: Stanford ALPACA, sampled to match harmful counts.

Requirements:
  pip install datasets

Usage:
  python reconstruct_datasets.py --repo_root /path/to/geometry-of-refusal

  If --repo_root is omitted, assumes current directory is repo root.
"""

import json
import os
import random
import argparse
from collections import Counter


def load_saladbench():
    """Load SALAD-BENCH and return harmful instructions grouped by source."""
    from datasets import load_dataset

    print("Loading SALAD-BENCH from HuggingFace...")
    # SALAD-BENCH has multiple configs; the main attack prompts are in the
    # base split. The dataset has columns like 'question' and '1-category',
    # '2-category', '3-category', plus a 'source' or 'baseq_source' column.
    #
    # We try the known HF paths. The exact column names may vary by version.

    try:
        ds = load_dataset("walledai/SaladBench", "prompts", split="base")
    except Exception as e:
        print(f"Failed to load SALAD-BENCH: {e}")
        print("Try: pip install datasets && huggingface-cli login")
       
    print(f"  Loaded {len(ds)} rows")

    # Identify columns
    cols = ds.column_names
    print(f"  Columns: {cols}")

    # prompt column
    instruction_col = "prompt"

    #  source column
    source_col = "source"

    # Group by source
    by_source = {}
    for row in ds:
        instruction = row[instruction_col]
        source = row[source_col]

        if source not in by_source:
            by_source[source] = []
        by_source[source].append(instruction)

    print(f"\n  Sources found ({len(by_source)}):")
    for src, items in sorted(by_source.items(), key=lambda x: -len(x[1])):
        print(f"    {src}: {len(items)} prompts")

    return by_source


def filter_and_sample_harmful(by_source, max_per_source=256, seed=42):
    """
    Exclude Multilingual and ToxicChat sources, sample up to 256 per source.
    Returns a flat list of instruction strings.
    """
    random.seed(seed)

    # Exclude sources matching these keywords (case-insensitive)
    exclude_keywords = ["Multilingual", "ToxicChat"]

    kept_sources = []
    excluded_sources = []

    for src in by_source:
        src_lower = src.lower().replace(" ", "").replace("_", "").replace("-", "")
        if any(kw.lower().replace(" ", "").replace("_", "").replace("-", "") in src_lower
               for kw in exclude_keywords):
            excluded_sources.append(src)
        else:
            kept_sources.append(src)

    print(f"\n  Excluded sources: {excluded_sources}")
    print(f"  Kept sources: {kept_sources}")

    all_instructions = []
    for src in kept_sources:
        items = by_source[src]
        if len(items) > max_per_source:
            items = random.sample(items, max_per_source)
        all_instructions.extend(items)

    random.shuffle(all_instructions)
    print(f"  Total harmful instructions after filtering: {len(all_instructions)}")

    return all_instructions


def load_alpaca():
    """Load Stanford ALPACA instructions."""
    from datasets import load_dataset

    print("\nLoading ALPACA from HuggingFace...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"  Loaded {len(ds)} rows")

    instructions = [row["instruction"] for row in ds]
    return instructions


def load_jailbreakbench():
    """Load JailbreakBench evaluation prompts."""
    from datasets import load_dataset

    print("\nLoading JailbreakBench...")
    try:
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
        cols = ds.column_names
        print(f"  Columns: {cols}")

        # Find the prompt column
        instruction_col = None
        for candidate in ["Goal", "goal", "prompt", "instruction", "question"]:
            if candidate in cols:
                instruction_col = candidate
                break

        if instruction_col is None:
            print(f"  WARNING: Could not find instruction column. First row: {ds[0]}")
            raise ValueError("Check JailbreakBench schema")

        category_col = None
        for candidate in ["Category", "category"]:
            if candidate in cols:
                category_col = candidate
                break

        instructions = []
        for row in ds:
            entry = {"instruction": row[instruction_col]}
            if category_col:
                entry["category"] = row[category_col]
            instructions.append(entry)

        print(f"  Loaded {len(instructions)} JailbreakBench prompts")
        return instructions

    except Exception as e:
        print(f"  WARNING: Could not load JailbreakBench: {e}")
        print("  You can manually place jailbreakbench.json in dataset/processed/")
        return None


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(data)} items → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=".",
                        help="Path to geometry-of-refusal repo root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=1184,
                        help="Number of training examples (paper uses 1184)")
    parser.add_argument("--n_val", type=int, default=128,
                        help="Number of validation examples")
    parser.add_argument("--n_test", type=int, default=128,
                        help="Number of test examples (harmless only)")
    args = parser.parse_args()

    random.seed(args.seed)
    splits_dir = os.path.join(args.repo_root, "data", "saladbench_splits")
    processed_dir = os.path.join(args.repo_root, "dataset", "processed")

    # =========================================================================
    # Harmful instructions (SALAD-BENCH)
    # =========================================================================
    by_source = load_saladbench()
    harmful_all = filter_and_sample_harmful(by_source, seed=args.seed)

    if len(harmful_all) < args.n_train + args.n_val:
        print(f"\n  WARNING: Only {len(harmful_all)} harmful instructions available, "
              f"need {args.n_train + args.n_val}. Adjusting n_train.")
        args.n_train = len(harmful_all) - args.n_val

    harmful_train = [{"instruction": inst} for inst in harmful_all[:args.n_train]]
    harmful_val = [{"instruction": inst} for inst in harmful_all[args.n_train:args.n_train + args.n_val]]

    print(f"\n  Harmful split: {len(harmful_train)} train, {len(harmful_val)} val")

    # =========================================================================
    # Harmless instructions (ALPACA)
    # =========================================================================
    alpaca_instructions = load_alpaca()
    random.shuffle(alpaca_instructions)

    # Match harmful counts, plus extra for test
    n_harmless_total = args.n_train + args.n_val + args.n_test
    if len(alpaca_instructions) < n_harmless_total:
        print(f"  WARNING: ALPACA has {len(alpaca_instructions)} instructions, need {n_harmless_total}")

    harmless_train = [{"instruction": inst} for inst in alpaca_instructions[:args.n_train]]
    harmless_val = [{"instruction": inst} for inst in alpaca_instructions[args.n_train:args.n_train + args.n_val]]
    harmless_test = [{"instruction": inst} for inst in alpaca_instructions[args.n_train + args.n_val:args.n_train + args.n_val + args.n_test]]

    print(f"  Harmless split: {len(harmless_train)} train, {len(harmless_val)} val, {len(harmless_test)} test")

    # =========================================================================
    # Save train/val/test splits
    # =========================================================================
    print(f"\nSaving splits to {splits_dir}/")

    save_json(harmful_train, os.path.join(splits_dir, "harmful_train.json"))
    save_json(harmful_val, os.path.join(splits_dir, "harmful_val.json"))
    save_json(harmless_train, os.path.join(splits_dir, "harmless_train.json"))
    save_json(harmless_val, os.path.join(splits_dir, "harmless_val.json"))
    save_json(harmless_test, os.path.join(splits_dir, "harmless_test.json"))

    # =========================================================================
    # JailbreakBench (evaluation dataset)
    # =========================================================================
    jbb_data = load_jailbreakbench()
    if jbb_data:
        save_json(jbb_data, os.path.join(processed_dir, "jailbreakbench.json"))

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Dataset reconstruction complete.")
    print(f"  {splits_dir}/")
    for f in sorted(os.listdir(splits_dir)):
        filepath = os.path.join(splits_dir, f)
        with open(filepath) as fh:
            n = len(json.load(fh))
        print(f"    {f}: {n} items")

    if os.path.exists(processed_dir):
        print(f"  {processed_dir}/")
        for f in sorted(os.listdir(processed_dir)):
            if f.endswith(".json"):
                filepath = os.path.join(processed_dir, f)
                with open(filepath) as fh:
                    n = len(json.load(fh))
                print(f"    {f}: {n} items")

    print("\nSanity check — first instruction from each split:")
    print(f"  harmful_train[0]: {harmful_train[0]['instruction'][:80]}...")
    print(f"  harmless_train[0]: {harmless_train[0]['instruction'][:80]}...")
    print("=" * 60)


if __name__ == "__main__":
    main()
