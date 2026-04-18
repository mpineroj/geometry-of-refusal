"""
Evaluate a trained RDO direction on JailbreakBench.

Loads the direction from disk, runs directional ablation and activation
addition attacks, and reports ASR via substring matching.

Usage (from refusal_direction/):
    python eval_rdo_direction.py \
        --model_path Qwen/Qwen2.5-3B-Instruct \
        --vector_path ../results/rdo/Qwen2.5-3B-Instruct/vectors/lowest_loss_vector_7ifl0hjw.pt \
        --dim_metadata_path ../results/dim_directions/Qwen2.5-3B-Instruct/direction_metadata.json
"""

import torch
import random
import json
import os
import sys
import argparse
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.load_dataset import load_dataset_split, load_dataset
from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak


def parse_args():
    load_dotenv("..", override=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--vector_path', type=str, required=True,
                        help='Path to the trained RDO direction (.pt file)')
    parser.add_argument('--dim_metadata_path', type=str, required=True,
                        help='Path to DIM direction_metadata.json (for layer info)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (default: next to vector)')
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(42)

    # Load the trained direction
    print(f"Loading RDO direction from: {args.vector_path}")
    direction = torch.load(args.vector_path, map_location='cpu')
    if direction.dim() > 1:
        # If it's a cone basis, take the first vector for single-direction eval
        print(f"  Vector shape: {direction.shape} — using first vector")
        direction = direction[0]
    print(f"  Direction shape: {direction.shape}, norm: {direction.float().norm():.4f}")

    # Load DIM metadata for layer info
    print(f"Loading DIM metadata from: {args.dim_metadata_path}")
    with open(args.dim_metadata_path) as f:
        metadata = json.load(f)
    add_layer = metadata["layer"]
    print(f"  Using layer: {add_layer}")

    # Setup output directory — unique per vector file
    if args.output_dir is None:
        vector_name = os.path.splitext(os.path.basename(args.vector_path))[0]
        args.output_dir = os.path.join(os.path.dirname(args.vector_path), f"eval_{vector_name}")

    # Load model
    model_alias = os.path.basename(args.model_path)
    cfg = Config(model_alias=model_alias, model_path=args.model_path)
    model_base = construct_model_base(cfg.model_path)

    # Move direction to model dtype/device
    direction = direction.to(dtype=next(model_base.model.parameters()).dtype,
                             device=next(model_base.model.parameters()).device)

    # Load evaluation dataset
    print("\nLoading JailbreakBench...")
    eval_dataset = load_dataset("jailbreakbench")
    print(f"  {len(eval_dataset)} prompts")

    # Load alpha (norm of DIM direction) for proper scaling
    dim_direction = torch.load(os.path.join(os.path.dirname(args.dim_metadata_path), "direction.pt"), map_location='cpu')
    alpha = dim_direction.norm().item()
    print(f"  Alpha (DIM direction norm): {alpha:.4f}")

   
   # Setup hooks
    baseline_hooks = ([], [])
    ablation_hooks = get_all_direction_ablation_hooks(model_base, direction)
    actadd_hooks = (
        [(model_base.model_block_modules[add_layer],
        get_activation_addition_input_pre_hook(vector=direction * alpha, coeff=-1.0))],
        []
    )


    # Generate completions for each intervention
    interventions = {
        "baseline": baseline_hooks,
        "ablation": ablation_hooks,
        "actadd": actadd_hooks,
    }

    for name, (pre_hooks, hooks) in interventions.items():
        print(f"\nGenerating completions: {name}")
        completions = model_base.generate_completions(
            eval_dataset,
            fwd_pre_hooks=pre_hooks,
            fwd_hooks=hooks,
            max_new_tokens=cfg.max_new_tokens,
            batch_size=cfg.completions_batch_size,
        )

        comp_path = os.path.join(args.output_dir, "completions",
                                 f"jailbreakbench_{name}_completions.json")
        with open(comp_path, "w") as f:
            json.dump(completions, f, indent=4)
        print(f"  Saved to: {comp_path}")

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (RDO direction)")
    print("=" * 60)

    results = {}
    for name in interventions:
        comp_path = os.path.join(args.output_dir, "completions",
                                 f"jailbreakbench_{name}_completions.json")
        eval_path = os.path.join(args.output_dir, "completions",
                                 f"jailbreakbench_{name}_evaluations.json")

        with open(comp_path) as f:
            completions = json.load(f)

        evaluation = evaluate_jailbreak(
            completions=completions,
            methodologies=["substring_matching"],
            evaluation_path=eval_path,
        )

        asr = evaluation.get("substring_matching_success_rate", "N/A")
        results[name] = asr
        print(f"  {name}: ASR = {asr}")

    # Save summary
    summary = {
        "model": args.model_path,
        "vector_path": args.vector_path,
        "layer": add_layer,
        "direction_norm": direction.float().norm().item(),
        "results": results,
    }
    summary_path = os.path.join(args.output_dir, "rdo_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
