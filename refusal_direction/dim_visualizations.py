"""
Local validation of DIM pipeline stages 1-3 (Mac/MPS-safe).
Adapted with --viz flag to produce DIM visualization figures.

New flags:
  --viz                Produce visualization after Stage 3
  --viz_out PATH       Where to save the figure (default: dim_visualization.png)
  --viz_position INT   Position index into mean_diffs (default: 0 = -n_eoi_toks)

Usage:
    python test_dim_local_viz.py --model_path Qwen/Qwen2.5-3B-Instruct --viz
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


# ── Visualization ──────────────────────────────────────────────────────────────

def visualize_dim_results(mean_diffs, selected_layer=None, viz_position=0, save_path="dim_visualization.png"):
    """
    Produce the 3-panel DIM visualization using real pipeline outputs.

    Panels:
      Top row   – PCA scatter showing μ_harmful and μ_harmless at 4 showcase layers,
                  with the DIM direction arrow between them (means-only; generate_directions
                  does not save per-prompt activations).
      Bottom-L  – ‖μ_harmful − μ_harmless‖ across all layers (layer selection story).
      Bottom-R  – Final DIM direction arrow at selected layer.

    Args:
        mean_diffs    : output of generate_directions — Tensor (n_positions, n_layers, d_model).
        selected_layer: if None, uses argmax of norms across layers.
        viz_position  : which token position index to visualise (default 0 = position -n_eoi_toks,
                        matching the Wollschläger pipeline's first eoi position).
        save_path     : output PNG path.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from sklearn.decomposition import PCA

    # ── 1. Slice position → (n_layers, d_model) ───────────────────────────────
    # generate_directions always returns Tensor (n_positions, n_layers, d_model).
    diffs = mean_diffs[viz_position].float().cpu()     # (n_layers, d_model)
    print(f"  [viz] Using position index {viz_position}, shape after slice: {diffs.shape}")

    n_layers, d_model = diffs.shape
    dim_norms = diffs.norm(dim=-1).numpy()             # (n_layers,)

    if selected_layer is None:
        selected_layer = int(dim_norms.argmax())

    showcase_layers = [
        0,
        n_layers // 3,
        2 * n_layers // 3,
        n_layers - 1,
    ]

    # ── 2. PCA basis from mean diff vectors ───────────────────────────────────
    # generate_directions only saves mean_diffs.pt, not per-prompt activations,
    # so scatter panels show the two mean endpoints only.
    # Fit PCA on all layer diff vectors to get a consistent 2D basis.
    pca = PCA(n_components=2).fit(diffs.numpy())       # (n_layers, d_model)

    def _proj(arr):
        """arr: (D,) → (2,)"""
        return pca.transform(arr[None])[0]

    def _mean_proj(layer):
        """Place μ_harmful and μ_harmless symmetrically around the PCA origin."""
        half = diffs[layer].numpy() / 2
        return _proj(half), _proj(-half)

    # ── 4. Draw ───────────────────────────────────────────────────────────────
    C_HARM   = '#ff6b6b'
    C_HARM_L = '#cc0000'
    C_SAFE   = '#74b9ff'
    C_SAFE_L = '#0055cc'
    C_DIM    = '#ffd700'
    C_ACCENT = '#a29bfe'
    C_FG     = '#cccccc'
    C_BG     = '#1a1a2e'

    def style_ax(ax, title):
        ax.set_facecolor(C_BG)
        ax.tick_params(colors='#666666', labelsize=8)
        for s in ax.spines.values():
            s.set_color('#333333')
        ax.set_title(title, color=C_FG, fontsize=9, pad=7)

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#0f0f0f')
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.35)

    # Top row — scatter at showcase layers (means-only)
    # Each panel is centered on the midpoint of its two means and padded
    # proportionally to that layer's norm, so arrow size is comparable across panels.
    max_norm = dim_norms.max()

    for i, layer in enumerate(showcase_layers):
        ax = fig.add_subplot(gs[0, i])
        norm_label = f'‖Δμ‖={dim_norms[layer]:.1f}'
        style_ax(ax, f'Layer {layer}  ({norm_label})')

        mh, ms = _mean_proj(layer)
        mid = (np.array(mh) + np.array(ms)) / 2

        # Pad = 60% of current layer norm mapped to PCA units; use max-norm layer
        # as reference so early layers show small arrows in proportional space.
        pad = 0.6 * (dim_norms[layer] / max_norm) * np.linalg.norm(np.array(mh) - np.array(ms)) + 0.5
        ax.set_xlim(mid[0] - pad, mid[0] + pad)
        ax.set_ylim(mid[1] - pad, mid[1] + pad)

        layer_norm = dim_norms[layer]
        if layer_norm < 0.5:
            ax.text(0.5, 0.5, 'near-zero\nseparation', transform=ax.transAxes,
                    color='#888888', fontsize=8, ha='center', va='center',
                    style='italic')
        else:
            ax.scatter(*mh, c=C_HARM_L, s=90,  marker='X', zorder=6,
                       edgecolors='white', linewidths=0.6, label='μ harmful')
            ax.scatter(*ms, c=C_SAFE_L, s=90,  marker='X', zorder=6,
                       edgecolors='white', linewidths=0.6, label='μ harmless')
            ax.annotate('', xy=mh, xytext=ms,
                        arrowprops=dict(arrowstyle='->', color=C_DIM,
                                        lw=1.8, mutation_scale=14))

        if i == 0:
            ax.legend(fontsize=6.5, framealpha=0.3, loc='upper left',
                      labelcolor='white', facecolor='#222222', edgecolor='#444444')

    fig.text(0.005, 0.77, 'Activation Space\n(PCA 2D)', color='#888888',
             fontsize=8, va='center', ha='left')

    # Bottom-left — norm across layers
    ax_n = fig.add_subplot(gs[1, :2])
    style_ax(ax_n, '‖μ_harmful − μ_harmless‖ across layers  →  layer selection')

    ax_n.plot(range(n_layers), dim_norms, color=C_ACCENT, lw=2, zorder=2)
    ax_n.fill_between(range(n_layers), dim_norms, alpha=0.15, color=C_ACCENT)

    for layer in showcase_layers:
        ax_n.axvline(layer, color='#444455', lw=0.9, linestyle=':')
        ax_n.text(layer + 0.2, dim_norms.min() * 0.97, str(layer),
                  color='#666688', fontsize=7)

    ax_n.axvline(selected_layer, color=C_DIM, lw=1.8, linestyle='--', zorder=3)
    ax_n.scatter([selected_layer], [dim_norms[selected_layer]],
                 color=C_DIM, s=80, zorder=5)
    ax_n.text(selected_layer + 0.5, dim_norms[selected_layer] * 0.93,
              f'Selected: layer {selected_layer}', color=C_DIM, fontsize=8)
    # Annotate that selection is by KL+ASR criteria, not just argmax norm
    ax_n.text(0.98, 0.97,
              'Selection: ablation ASR + actadd ASR + KL ≤ threshold\n(not argmax norm)',
              transform=ax_n.transAxes, color='#888888', fontsize=7,
              ha='right', va='top', style='italic')
    ax_n.set_xlabel('Layer', color='#888888', fontsize=9)
    ax_n.set_ylabel('Vector norm', color='#888888', fontsize=9)

    # Bottom-right — final direction
    ax_d = fig.add_subplot(gs[1, 2:])
    style_ax(ax_d, f'DIM Direction  r = μ_harmful − μ_harmless   (Layer {selected_layer})')

    mh, ms = _mean_proj(selected_layer)

    ax_d.scatter(*mh, c=C_HARM_L, s=130, marker='X', zorder=6,
                 edgecolors='white', linewidths=0.8, label='μ harmful')
    ax_d.scatter(*ms, c=C_SAFE_L, s=130, marker='X', zorder=6,
                 edgecolors='white', linewidths=0.8, label='μ harmless')
    ax_d.annotate('', xy=mh, xytext=ms,
                  arrowprops=dict(arrowstyle='->', color=C_DIM,
                                  lw=2.8, mutation_scale=22))

    mid = (np.array(mh) + np.array(ms)) / 2
    ax_d.text(mid[0] + 0.07, mid[1] + 0.07,
              r'$\mathbf{r}_{DIM}$ = μ_harmful − μ_harmless',
              color=C_DIM, fontsize=9.5, fontweight='bold')
    ax_d.legend(fontsize=8, framealpha=0.3, labelcolor='white',
                facecolor='#222222', edgecolor='#444444')

    fig.suptitle('Difference in Means: Extracting the Refusal Direction Across Layers',
                 color='white', fontsize=13, fontweight='bold', y=0.99)

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  [viz] Saved → {save_path}")
    plt.close(fig)


# ── CLI & pipeline ─────────────────────────────────────────────────────────────

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
    parser.add_argument('--viz', action='store_true',
                        help='Produce DIM visualization after Stage 3')
    parser.add_argument('--viz_out', type=str, default='dim_visualization.png',
                        help='Output path for visualization PNG (default: dim_visualization.png)')
    parser.add_argument('--selected_layer', type=int, default=None,
                        help='Override selected layer for viz (default: argmax of norms)')
    parser.add_argument('--viz_position', type=int, default=0,
                        help='Position index into mean_diffs[n_positions] to visualise '
                             '(default: 0 = first eoi token position, i.e. -n_eoi_toks)')
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
    harmful_train  = load_dataset_split(harmtype='harmful',  split='train', instructions_only=True)
    harmless_train = load_dataset_split(harmtype='harmless', split='train', instructions_only=True)[:len(harmful_train)]
    harmful_val    = load_dataset_split(harmtype='harmful',  split='val',   instructions_only=True)
    harmless_val   = load_dataset_split(harmtype='harmless', split='val',   instructions_only=True)[:len(harmful_val)]

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

    harmful_subset  = harmful_train[:n_test]
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

    n_harmful_kept  = sum(1 for s in harmful_scores.tolist()  if s >  0)
    n_harmless_kept = sum(1 for s in harmless_scores.tolist() if s <  0)

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
        harmful_filtered  = [inst for inst, score in zip(harmful_train, harmful_scores.tolist()) if score > 0]
        harmless_filtered = [inst for inst, score in zip(harmless_train, harmless_scores.tolist()) if score < 0][:len(harmful_filtered)]
    else:
        harmful_filtered  = harmful_train
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
    # Stage 4 (optional): Visualize
    # ==================================================================
    if args.viz:
        print("\n" + "=" * 60)
        print("STAGE 4: Visualizing DIM results")
        print("=" * 60)
        visualize_dim_results(
            mean_diffs=mean_diffs,
            selected_layer=args.selected_layer,
            viz_position=args.viz_position,
            save_path=args.viz_out,
        )

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
  Visualization:             {'OK → ' + args.viz_out if args.viz else 'skipped (pass --viz)'}

  Skipped (expensive): direction selection, completion generation, evaluation.
  These test ~{n_candidates} candidates x 128 val prompts each -> run on Adroit.

  Next: sbatch run_dim_della.slurm
""")


if __name__ == "__main__":
    main()