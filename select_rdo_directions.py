#!/usr/bin/env python3
"""
select_rdo_directions.py

Validation-based re-selection for RDO and cone candidate pools.

Fixes the missing selection step from run_geometry_pipeline.slurm, which trains
candidates (vectors_<run_id>.pt) but never runs validation-based selection.
run_rdo_pipeline.py was the intended script but requires wandb API access and
only ever loads lowest_loss_vector.pt (pool size = 1, no real selection).

What this script does:
  - Loads full vectors_<run_id>.pt candidate pools from GEOMETRY_RESULTS_FINAL
  - Scores each candidate on 128 harmful + 128 harmless val prompts
  - Filter: KL(ablated harmless || baseline harmless) <= kl_threshold
            AND mean harmless refusal score when direction is added >= 0.0
  - K=1 (RDO): rank survivors by lowest mean harmful refusal score when ablated
  - K>1 (cone): rank survivors by lowest mean harmful refusal score across
                MC samples from the cone; KL is computed over MC samples too
  - Selects the whole K-dimensional basis as an atomic unit (no cross-run mixing)
  - Writes selected_vector_<run_id>.pt + selected_metadata_<run_id>.json
    to results_dir/rdo/<model_name>/selected/

Usage:
  python select_rdo_directions.py \\
      --model_path /scratch/network/$USER/models/Qwen2.5-3B-Instruct-ER-fullweight \\
      --model_name Qwen2.5-full-lora \\
      --results_dir /home/$USER/geometry-of-refusal/geometry_results_final/results

  # To process all ER models in one pass (omit --model_name to use basename):
  for MODEL in Qwen2.5-full-lora Qwen2.5-refusal-only-lora \\
               Qwen2.5-explanation-only-lora Qwen2.5-justification-only-lora; do
    python select_rdo_directions.py \\
        --model_path /scratch/network/$USER/models/<full-model-name> \\
        --model_name $MODEL \\
        --results_dir /path/to/geometry_results_final/results
  done
"""

import argparse
import json
import math
import os
import sys

import torch
from tqdm import tqdm

# Make refusal_direction importable regardless of cwd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "refusal_direction"))

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import (
    get_activation_addition_input_pre_hook,
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
)
from pipeline.submodules.select_direction import (
    get_refusal_scores,
    get_last_position_logits,
    kl_div_fn,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Validate-and-select RDO/cone directions from candidate pools."
    )
    p.add_argument(
        "--model_path",
        required=True,
        help="Local filesystem path to the HuggingFace model (used for loading).",
    )
    p.add_argument(
        "--model_name",
        default=None,
        help=(
            "Model name as it appears under results_dir/rdo/ (e.g. 'Qwen2.5-full-lora'). "
            "Defaults to basename of --model_path."
        ),
    )
    p.add_argument(
        "--results_dir",
        default="geometry_results_final/results",
        help="Path to results dir containing rdo/ and dim_directions/ subdirectories.",
    )
    p.add_argument(
        "--data_dir",
        default=None,
        help=(
            "Path to the data directory containing saladbench_splits/. "
            "Defaults to <script_dir>/data."
        ),
    )
    p.add_argument(
        "--kl_threshold",
        type=float,
        default=1.0,
        help="KL divergence threshold for filtering candidates (default: 1.0).",
    )
    p.add_argument(
        "--induce_threshold",
        type=float,
        default=0.0,
        help="Minimum refusal induction score (harmless steering) to pass filter (default: 0.0).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Per-GPU batch size for scoring forward passes (default: 32).",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=8,
        help="Number of Monte Carlo samples for cone scoring (default: 8).",
    )
    p.add_argument(
        "--run_id",
        default=None,
        help="If set, process only this specific run_id (useful for debugging or retry).",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip run_ids that already have a selected_metadata_<run_id>.json file.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_val_instructions(data_dir):
    """Return (harmful_instrs, harmless_instrs) as plain string lists."""
    harmful = json.load(
        open(os.path.join(data_dir, "saladbench_splits", "harmful_val.json"))
    )
    harmless = json.load(
        open(os.path.join(data_dir, "saladbench_splits", "harmless_val.json"))
    )
    return [x["instruction"] for x in harmful], [x["instruction"] for x in harmless]


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def _ablation_hooks(model_base, direction):
    """Full-network directional ablation (pre-hooks on blocks + output hooks on attn/mlp)."""
    n = model_base.model.config.num_hidden_layers
    pre = [
        (model_base.model_block_modules[l], get_direction_ablation_input_pre_hook(direction=direction))
        for l in range(n)
    ]
    out = [
        (model_base.model_attn_modules[l], get_direction_ablation_output_hook(direction=direction))
        for l in range(n)
    ]
    out += [
        (model_base.model_mlp_modules[l], get_direction_ablation_output_hook(direction=direction))
        for l in range(n)
    ]
    return pre, out


def _addition_hook(model_base, direction, add_layer):
    """Add direction at a single layer (pre-hook only)."""
    coeff = torch.tensor(1.0)
    pre = [
        (
            model_base.model_block_modules[add_layer],
            get_activation_addition_input_pre_hook(vector=direction, coeff=coeff),
        )
    ]
    return pre, []


# ---------------------------------------------------------------------------
# Candidate pool utilities
# ---------------------------------------------------------------------------

def infer_k_and_stack(vecs):
    """
    vecs: Python list of tensors loaded from vectors_<run_id>.pt

    Each element is either:
      - shape (d,)      -> K=1 RDO direction
      - shape (1, d)    -> K=1 cone (treat as K=1)
      - shape (K, d)    -> K-dimensional cone

    Returns:
      candidates  (N, d) or (N, K, d) float tensor (CPU)
      k_dim       int (1 for RDO, K for cone)
      original_ndim  int (1 or 2) — original per-element ndim, preserved for output format
    """
    sample = vecs[0]
    if sample.ndim == 1:
        # (d,) -> stack to (N, d)
        candidates = torch.stack(vecs).float()
        return candidates, 1, 1
    elif sample.ndim == 2:
        k = sample.shape[0]
        if k == 1:
            # (1, d) -> squeeze and treat as K=1
            candidates = torch.stack([v[0] for v in vecs]).float()
            return candidates, 1, 2
        else:
            # (K, d) -> stack to (N, K, d)
            candidates = torch.stack(vecs).float()
            return candidates, k, 2
    else:
        raise ValueError(f"Unexpected tensor ndim: {sample.ndim}, shape: {sample.shape}")


# ---------------------------------------------------------------------------
# Cone MC sampling (no wandb)
# ---------------------------------------------------------------------------

def _sample_hypersphere(n_samples, dim, device):
    """Sample unit vectors from the positive orthant of the hypersphere (seeded, reproducible)."""
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    s = torch.randn(n_samples, dim).abs()
    s = s / s.norm(dim=1, keepdim=True)
    torch.set_rng_state(rng)
    return s.to(device)


def _make_cone_samples(basis, alpha, n_samples):
    """
    basis: (K, d) normalized per-row tensor on device
    alpha: scalar — scale factor (norm of DIM direction)
    Returns: list of n_samples (d,) tensors representing random cone directions
    """
    K = basis.shape[0]
    device = basis.device
    norm_basis = basis / (basis.norm(dim=-1, keepdim=True) + 1e-8)  # (K, d)
    weights = _sample_hypersphere(n_samples, K, device)  # (n_samples, K)

    samples = []
    for si in range(n_samples):
        v = torch.matmul(weights[si], norm_basis)  # (K,) @ (K, d) -> (d,)
        v = v / (v.norm() + 1e-8)
        v = v * alpha
        samples.append(v)
    return samples


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_rdo_candidates(model_base, candidates, harmful_instrs, harmless_instrs, add_layer, batch_size):
    """
    Score K=1 candidates. candidates: (N, d) float CPU tensor.
    Returns (kl_scores, abl_scores, steer_scores) each shape (N,) float64 CPU.
    """
    n = candidates.shape[0]
    kl_s = torch.zeros(n, dtype=torch.float64)
    abl_s = torch.zeros(n, dtype=torch.float64)
    steer_s = torch.zeros(n, dtype=torch.float64)

    baseline_logits = get_last_position_logits(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=harmless_instrs,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        batch_size=batch_size,
    )

    for i in tqdm(range(n), desc="  RDO candidates", leave=False):
        direction = candidates[i].to(model_base.model.device)
        pre, out = _ablation_hooks(model_base, direction)

        # KL: ablate on harmless
        logits = get_last_position_logits(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            instructions=harmless_instrs,
            tokenize_instructions_fn=model_base.tokenize_instructions_fn,
            fwd_pre_hooks=pre,
            fwd_hooks=out,
            batch_size=batch_size,
        )
        kl_s[i] = kl_div_fn(baseline_logits, logits).mean().item()

        # Ablation refusal: ablate on harmful
        abl = get_refusal_scores(
            model_base.model,
            harmful_instrs,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
            fwd_pre_hooks=pre,
            fwd_hooks=out,
            batch_size=batch_size,
        )
        abl_s[i] = abl.mean().item()

        # Steering refusal: add at add_layer on harmless
        s_pre, s_out = _addition_hook(model_base, direction, add_layer)
        steer = get_refusal_scores(
            model_base.model,
            harmless_instrs,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
            fwd_pre_hooks=s_pre,
            fwd_hooks=s_out,
            batch_size=batch_size,
        )
        steer_s[i] = steer.mean().item()

    return kl_s, abl_s, steer_s


def score_cone_candidates(model_base, candidates, harmful_instrs, harmless_instrs, add_layer, alpha, n_samples, batch_size):
    """
    Score K>1 candidates. candidates: (N, K, d) float CPU tensor.
    Returns list of per-candidate score dicts.
    """
    n, K, _ = candidates.shape
    device = model_base.model.device

    baseline_logits = get_last_position_logits(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=harmless_instrs,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        batch_size=batch_size,
    )

    results = []
    for ci in tqdm(range(n), desc=f"  Cone K={K} candidates", leave=False):
        basis = candidates[ci].to(device)  # (K, d)

        # Per-basis-vector: ablation refusal + steering (for basis filter)
        basis_abl, basis_steer = [], []
        for ki in range(K):
            bv = basis[ki].to(model_base.model.dtype)
            b_pre, b_out = _ablation_hooks(model_base, bv)
            abl = get_refusal_scores(
                model_base.model, harmful_instrs, model_base.tokenize_instructions_fn,
                model_base.refusal_toks, fwd_pre_hooks=b_pre, fwd_hooks=b_out, batch_size=batch_size,
            )
            basis_abl.append(abl.mean().item())

            s_pre, _ = _addition_hook(model_base, bv, add_layer)
            steer = get_refusal_scores(
                model_base.model, harmless_instrs, model_base.tokenize_instructions_fn,
                model_base.refusal_toks, fwd_pre_hooks=s_pre, fwd_hooks=[], batch_size=batch_size,
            )
            basis_steer.append(steer.mean().item())

        # Monte Carlo cone samples: KL + ablation refusal + steering
        mc_samples = _make_cone_samples(basis.to(torch.float32), alpha, n_samples)

        sample_kl, sample_abl, sample_steer = [], [], []
        for sv in mc_samples:
            sv = sv.to(model_base.model.dtype)
            s_pre, s_out = _ablation_hooks(model_base, sv)

            logits = get_last_position_logits(
                model=model_base.model, tokenizer=model_base.tokenizer,
                instructions=harmless_instrs, tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                fwd_pre_hooks=s_pre, fwd_hooks=s_out, batch_size=batch_size,
            )
            sample_kl.append(kl_div_fn(baseline_logits, logits).mean().item())

            abl = get_refusal_scores(
                model_base.model, harmful_instrs, model_base.tokenize_instructions_fn,
                model_base.refusal_toks, fwd_pre_hooks=s_pre, fwd_hooks=s_out, batch_size=batch_size,
            )
            sample_abl.append(abl.mean().item())

            t_pre, _ = _addition_hook(model_base, sv, add_layer)
            steer = get_refusal_scores(
                model_base.model, harmless_instrs, model_base.tokenize_instructions_fn,
                model_base.refusal_toks, fwd_pre_hooks=t_pre, fwd_hooks=[], batch_size=batch_size,
            )
            sample_steer.append(steer.mean().item())

        results.append({
            "candidate_idx": ci,
            "basis_abl": basis_abl,
            "basis_steer": basis_steer,
            "sample_kl": sample_kl,
            "sample_abl": sample_abl,
            "sample_steer": sample_steer,
        })

    return results


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_rdo(kl_s, abl_s, steer_s, kl_threshold, induce_threshold):
    """
    Filter + rank RDO candidates.
    Filter: KL <= kl_threshold AND steer >= induce_threshold (both must pass)
    Rank: ascending by ablation refusal score (lower = more effective attack)
    Fallback: lowest KL candidate if nothing passes.
    Returns (best_idx, fallback, n_passed, scores_at_best)
    """
    n = len(kl_s)
    passed = []
    for i in range(n):
        kl = float(kl_s[i])
        abl = float(abl_s[i])
        steer = float(steer_s[i])
        if math.isnan(kl) or math.isnan(abl) or math.isnan(steer):
            continue
        if kl <= kl_threshold and steer >= induce_threshold:
            passed.append((abl, i))

    if not passed:
        best_i = int(min(range(n), key=lambda x: float(kl_s[x])))
        return best_i, True, 0, {
            "ablation_refusal_score": float(abl_s[best_i]),
            "steering_score": float(steer_s[best_i]),
            "kl_div_score": float(kl_s[best_i]),
        }

    _, best_i = min(passed, key=lambda x: x[0])
    return best_i, False, len(passed), {
        "ablation_refusal_score": float(abl_s[best_i]),
        "steering_score": float(steer_s[best_i]),
        "kl_div_score": float(kl_s[best_i]),
    }


def select_cone(results, kl_threshold, induce_threshold):
    """
    Filter + rank cone candidates, matching the logic in select_direction.select_cone_basis:
      Basis filter:  min(basis_steer) >= induce_threshold (KL disabled per original code)
      Sample filter: max(sample_kl) <= kl_threshold AND mean(sample_steer) >= induce_threshold
      If all bases fail basis filter  -> ignore basis filter (all_basis_discarded)
      If all samples fail sample filter -> ignore induce_threshold in sample filter
    Rank: ascending by mean(sample_abl) (lower = more effective attack on average over cone)
    Fallback: candidate with lowest max(sample_kl) if nothing passes both filters.
    Returns (best_idx, fallback, n_passed, scores_at_best)
    """
    n = len(results)

    # Pre-compute aggregates
    max_basis_kl = [9999.0] * n  # KL disabled for basis filter (matches original code)
    min_basis_steer = [min(r["basis_steer"]) for r in results]
    max_sample_kl = [max(r["sample_kl"]) for r in results]
    mean_sample_abl = [sum(r["sample_abl"]) / len(r["sample_abl"]) for r in results]
    mean_sample_steer = [sum(r["sample_steer"]) / len(r["sample_steer"]) for r in results]

    # Check if all fail each filter globally (to enable fallback modes)
    all_basis_discarded = all(min_basis_steer[i] < induce_threshold for i in range(n))
    all_sample_discarded = all(
        max_sample_kl[i] > kl_threshold or mean_sample_steer[i] < induce_threshold
        for i in range(n)
    )

    if all_basis_discarded:
        print("    WARNING: all candidates fail basis steering filter — basis filter disabled.")
    if all_sample_discarded:
        print("    WARNING: all candidates fail sample filter — relaxing sample induce threshold.")

    passed = []
    for i in range(n):
        discard_basis = (not all_basis_discarded) and (min_basis_steer[i] < induce_threshold)

        eff_induce = -999.0 if all_sample_discarded else induce_threshold
        discard_sample = (
            max_sample_kl[i] > kl_threshold or mean_sample_steer[i] < eff_induce
        )

        if discard_basis or discard_sample:
            continue
        passed.append((mean_sample_abl[i], i))

    if not passed:
        best_i = int(min(range(n), key=lambda x: max_sample_kl[x]))
        r = results[best_i]
        return best_i, True, 0, {
            "mean_sample_abl": mean_sample_abl[best_i],
            "mean_sample_steer": mean_sample_steer[best_i],
            "max_sample_kl": max_sample_kl[best_i],
            "min_basis_steer": min_basis_steer[best_i],
            "basis_abl": r["basis_abl"],
            "basis_steer": r["basis_steer"],
        }

    _, best_i = min(passed, key=lambda x: x[0])
    r = results[best_i]
    return best_i, False, len(passed), {
        "mean_sample_abl": mean_sample_abl[best_i],
        "mean_sample_steer": mean_sample_steer[best_i],
        "max_sample_kl": max_sample_kl[best_i],
        "min_basis_steer": min_basis_steer[best_i],
        "basis_abl": r["basis_abl"],
        "basis_steer": r["basis_steer"],
    }


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------

def process_run(
    run_id,
    vectors_path,
    model_base,
    harmful_instrs,
    harmless_instrs,
    add_layer,
    alpha,
    model_name,
    output_dir,
    kl_threshold,
    induce_threshold,
    n_samples,
    batch_size,
    skip_existing,
):
    meta_path = os.path.join(output_dir, f"selected_metadata_{run_id}.json")
    vec_path = os.path.join(output_dir, f"selected_vector_{run_id}.pt")

    if skip_existing and os.path.exists(meta_path):
        print(f"  [{run_id}] already exists, skipping.")
        return

    print(f"\n  [{run_id}] loading candidates from {vectors_path}")
    raw_vecs = torch.load(vectors_path, map_location="cpu")
    candidates, k_dim, original_ndim = infer_k_and_stack(raw_vecs)
    n_candidates = candidates.shape[0]
    print(f"  [{run_id}] K={k_dim}, n_candidates={n_candidates}")

    os.makedirs(output_dir, exist_ok=True)

    if k_dim == 1:
        # RDO / single-direction selection
        kl_s, abl_s, steer_s = score_rdo_candidates(
            model_base, candidates, harmful_instrs, harmless_instrs, add_layer, batch_size
        )
        best_i, fallback, n_passed, best_scores = select_rdo(
            kl_s, abl_s, steer_s, kl_threshold, induce_threshold
        )

        # Preserve original storage format
        if original_ndim == 1:
            selected = candidates[best_i]               # (d,)
        else:
            selected = candidates[best_i].unsqueeze(0)  # (1, d)

        metadata = {
            "run_id": run_id,
            "model_name": model_name,
            "k_dim": 1,
            "candidate_idx": int(best_i),
            "n_candidates_total": n_candidates,
            "n_candidates_passed_filter": n_passed,
            "kl_threshold": kl_threshold,
            "induce_threshold": induce_threshold,
            "add_layer": add_layer,
            "fallback": fallback,
            **best_scores,
        }

    else:
        # Cone selection
        cone_results = score_cone_candidates(
            model_base, candidates, harmful_instrs, harmless_instrs, add_layer, alpha, n_samples, batch_size
        )
        best_i, fallback, n_passed, best_scores = select_cone(
            cone_results, kl_threshold, induce_threshold
        )

        selected = candidates[best_i]  # (K, d)

        metadata = {
            "run_id": run_id,
            "model_name": model_name,
            "k_dim": k_dim,
            "candidate_idx": int(best_i),
            "n_candidates_total": n_candidates,
            "n_candidates_passed_filter": n_passed,
            "kl_threshold": kl_threshold,
            "induce_threshold": induce_threshold,
            "add_layer": add_layer,
            "n_samples": n_samples,
            "fallback": fallback,
            **best_scores,
        }

    torch.save(selected, vec_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    status = "FALLBACK" if fallback else "OK"
    print(
        f"  [{run_id}] {status} — selected candidate {best_i} "
        f"({n_passed}/{n_candidates} passed filter) → {vec_path}"
    )
    if fallback:
        print(f"  [{run_id}] WARNING: no candidate passed KL+induction filter; fallback used.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    model_name = args.model_name or os.path.basename(args.model_path)
    data_dir = args.data_dir or os.path.join(_SCRIPT_DIR, "data")
    results_dir = os.path.abspath(args.results_dir)

    vectors_dir = os.path.join(results_dir, "rdo", model_name, "vectors")
    dim_dir = os.path.join(results_dir, "dim_directions", model_name)
    output_dir = os.path.join(results_dir, "rdo", model_name, "selected")

    # ------------------------------------------------------------------
    # Validate paths
    # ------------------------------------------------------------------
    if not os.path.isdir(vectors_dir):
        sys.exit(f"ERROR: vectors directory not found: {vectors_dir}")
    if not os.path.isdir(dim_dir):
        sys.exit(f"ERROR: DIM direction directory not found: {dim_dir}")

    dim_metadata_path = os.path.join(dim_dir, "direction_metadata.json")
    dim_direction_path = os.path.join(dim_dir, "direction.pt")
    if not os.path.isfile(dim_metadata_path) or not os.path.isfile(dim_direction_path):
        sys.exit(
            f"ERROR: DIM metadata or direction not found in {dim_dir}. "
            "Run DIM extraction first."
        )

    # ------------------------------------------------------------------
    # Load DIM metadata (add_layer + alpha)
    # ------------------------------------------------------------------
    dim_meta = json.load(open(dim_metadata_path))
    add_layer = int(dim_meta["layer"])
    dim_direction = torch.load(dim_direction_path, map_location="cpu").float()
    alpha = float(dim_direction.norm().item())
    print(f"DIM: add_layer={add_layer}, direction_norm (alpha)={alpha:.4f}")

    # ------------------------------------------------------------------
    # Load val data
    # ------------------------------------------------------------------
    harmful_instrs, harmless_instrs = load_val_instructions(data_dir)
    print(f"Val data: {len(harmful_instrs)} harmful, {len(harmless_instrs)} harmless")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model from {args.model_path} ...")
    model_base = construct_model_base(args.model_path)
    print(f"Model loaded: {model_base.model.__class__.__name__}")

    # ------------------------------------------------------------------
    # Enumerate runs
    # ------------------------------------------------------------------
    if args.run_id:
        run_ids = [args.run_id]
    else:
        run_ids = sorted(
            f.replace("vectors_", "").replace(".pt", "")
            for f in os.listdir(vectors_dir)
            if f.startswith("vectors_") and f.endswith(".pt")
        )

    print(f"\nProcessing {len(run_ids)} run(s) for model '{model_name}':")
    for rid in run_ids:
        print(f"  {rid}")

    # ------------------------------------------------------------------
    # Process each run
    # ------------------------------------------------------------------
    for run_id in run_ids:
        vectors_path = os.path.join(vectors_dir, f"vectors_{run_id}.pt")
        if not os.path.isfile(vectors_path):
            print(f"\n  [{run_id}] ERROR: {vectors_path} not found — skipping.")
            continue
        try:
            process_run(
                run_id=run_id,
                vectors_path=vectors_path,
                model_base=model_base,
                harmful_instrs=harmful_instrs,
                harmless_instrs=harmless_instrs,
                add_layer=add_layer,
                alpha=alpha,
                model_name=model_name,
                output_dir=output_dir,
                kl_threshold=args.kl_threshold,
                induce_threshold=args.induce_threshold,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                skip_existing=args.skip_existing,
            )
        except Exception as exc:
            print(f"\n  [{run_id}] FAILED: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Selected vectors written to {output_dir}")


if __name__ == "__main__":
    main()
