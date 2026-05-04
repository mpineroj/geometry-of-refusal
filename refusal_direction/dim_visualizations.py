"""
Local validation of DIM pipeline stages 1-3 (Mac/MPS-safe).
Adapted with --viz flag to produce DIM visualization figures.

New flags:
  --viz                Produce visualization after Stage 3
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
    Produce the DIM visualization using real pipeline outputs.

    Panels:
      Top       – Trajectories of μ_harmful and μ_harmless across ALL layers (PCA 2D,
                  colored by layer depth), with the selected layer's DIM arrow highlighted.
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

    # if save path doesn't exist, create directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ── 1. Slice position → (n_layers, d_model) ───────────────────────────────
    # generate_directions always returns Tensor (n_positions, n_layers, d_model).
    diffs = mean_diffs[viz_position].float().cpu()     # (n_layers, d_model)
    print(f"  [viz] Using position index {viz_position}, shape after slice: {diffs.shape}")

    n_layers, d_model = diffs.shape
    dim_norms = diffs.norm(dim=-1).numpy()             # (n_layers,)

    if selected_layer is None:
        selected_layer = int(dim_norms.argmax())

    showcase_layers = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]

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

    # ── 3. Draw ───────────────────────────────────────────────────────────────
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

    # ── Top row: single wide panel — trajectories across ALL layers ───────────
    ax_traj = fig.add_subplot(gs[0, :])
    style_ax(ax_traj, 'Mean Activation Trajectories Across All Layers  (PCA 2D, colored by layer)')

    mh_all = np.array([_mean_proj(l)[0] for l in range(n_layers)])   # (L, 2)
    ms_all = np.array([_mean_proj(l)[1] for l in range(n_layers)])   # (L, 2)

    # Connecting lines (thin, low alpha) to show trajectory shape
    ax_traj.plot(mh_all[:, 0], mh_all[:, 1], color=C_HARM, lw=0.6, alpha=0.3, zorder=1)
    ax_traj.plot(ms_all[:, 0], ms_all[:, 1], color=C_SAFE, lw=0.6, alpha=0.3, zorder=1)

    # Dots colored by layer depth via plasma colormap
    sc_h = ax_traj.scatter(mh_all[:, 0], mh_all[:, 1],
                           c=range(n_layers), cmap='plasma', s=28,
                           edgecolors=C_HARM, linewidths=0.8, zorder=3, label='μ harmful')
    ax_traj.scatter(ms_all[:, 0], ms_all[:, 1],
                    c=range(n_layers), cmap='plasma', s=28,
                    edgecolors=C_SAFE, linewidths=0.8, zorder=3,
                    marker='s', label='μ harmless')

    # Highlight selected layer with bold DIM arrow
    mh_sel, ms_sel = _mean_proj(selected_layer)
    ax_traj.annotate('', xy=mh_sel, xytext=ms_sel,
                     arrowprops=dict(arrowstyle='->', color=C_DIM, lw=2.5,
                                     mutation_scale=18), zorder=5)
    ax_traj.scatter(*mh_sel, c=C_HARM_L, s=120, marker='X', zorder=6,
                    edgecolors='white', linewidths=0.8)
    ax_traj.scatter(*ms_sel, c=C_SAFE_L, s=120, marker='X', zorder=6,
                    edgecolors='white', linewidths=0.8)
    ax_traj.text(mh_sel[0] + 0.3, mh_sel[1],
                 f'Layer {selected_layer} (selected)', color=C_DIM, fontsize=8)

    # Milestone layer labels on both trajectories
    for l in showcase_layers:
        ax_traj.text(mh_all[l, 0] + 0.15, mh_all[l, 1] + 0.15,
                     str(l), color='#aaaaaa', fontsize=6.5, zorder=7)
        ax_traj.text(ms_all[l, 0] + 0.15, ms_all[l, 1] + 0.15,
                     str(l), color='#aaaaaa', fontsize=6.5, zorder=7)

    cb = plt.colorbar(sc_h, ax=ax_traj, pad=0.01, fraction=0.015)
    cb.set_label('Layer', color=C_FG, fontsize=8)
    cb.ax.yaxis.set_tick_params(color='#666666', labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#666666')

    ax_traj.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax_traj.legend(fontsize=8, framealpha=0.3, labelcolor='white',
                   facecolor='#222222', edgecolor='#444444', loc='upper left')

    fig.text(0.005, 0.77, 'Activation Space\n(PCA 2D)', color='#888888',
             fontsize=8, va='center', ha='left')

    # ── Bottom-left: norm across layers ───────────────────────────────────────
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
    ax_n.text(0.15, 0.06,
              'Selected by: ablation ASR + actadd ASR + KL ≤ threshold  (not argmax norm)',
              transform=ax_n.transAxes, color='#888888', fontsize=7,
              ha='left', va='bottom', style='italic')
    ax_n.set_xlabel('Layer', color='#888888', fontsize=9)
    ax_n.set_ylabel('Vector norm', color='#888888', fontsize=9)

    # ── Bottom-right: final direction at selected layer ────────────────────────
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
            save_path="results/dim_visualization_" + model_alias + ".png"
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
  Visualization:             {'OK → results/dim_visualization_' + model_alias + '.png' if args.viz else 'SKIPPED (use --viz)'}

  Skipped (expensive): direction selection, completion generation, evaluation.
  These test ~{n_candidates} candidates x 128 val prompts each -> run on Adroit.

  Next: sbatch run_dim_della.slurm
""")


if __name__ == "__main__":
    main()