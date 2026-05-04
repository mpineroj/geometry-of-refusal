"""
Run the geometry-of-refusal pipeline (RDO + Cones + RepInd) on Modal.

Setup (one-time):
  1. pip install modal
  2. modal token new  (authenticate)
  3. Upload your repo and data to the Modal volume:
       modal run modal_geometry.py::upload --model-path /scratch/network/mp3687/models/Qwen2.5-3B-Instruct-ER-fullweight

  4. Run the pipeline:
       modal run modal_geometry.py::run_pipeline --model-name Qwen2.5-3B-Instruct-ER-fullweight

  5. Download results:
       modal run modal_geometry.py::download_results --model-name Qwen2.5-3B-Instruct-ER-fullweight

Steps 3-5 can also be run all at once:
       modal run modal_geometry.py::run_all --model-path /scratch/network/mp3687/models/Qwen2.5-3B-Instruct-ER-fullweight
"""

import modal
import os

# ── Modal setup ─────────────────────────────────────────────
app = modal.App("geometry-of-refusal")

# Persistent volume for repo, models, data, and results
vol = modal.Volume.from_name("geometry-vol", create_if_missing=True)
VOL_PATH = "/vol"

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.44.0",
        "accelerate",
        "datasets",
        "nnsight==0.3.6",
        "wandb",
        "python-dotenv",
        "jaxtyping",
        "einops",
        "scipy",
        "numpy",
    )
)


# ── Upload function ─────────────────────────────────────────
@app.function(volumes={VOL_PATH: vol}, timeout=1800)
def upload(model_path: str = ""):
    """
    Upload repo, data, and model to the Modal volume.
    Run from Adroit or local machine.
    
    This function just creates the directory structure.
    You'll upload files using modal volume commands.
    """
    import subprocess
    
    dirs = [
        f"{VOL_PATH}/repo",
        f"{VOL_PATH}/models",
        f"{VOL_PATH}/results/dim_directions",
        f"{VOL_PATH}/results/rdo",
        f"{VOL_PATH}/data/saladbench_splits",
        f"{VOL_PATH}/cache",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    print("Directory structure created on volume.")
    print("Now upload files with:")
    print("")
    print("  # From your local/Adroit machine:")
    print("  modal volume put geometry-vol ~/geometry-of-refusal /repo")
    print("  modal volume put geometry-vol /scratch/network/$USER/models/Qwen2.5-3B-Instruct-ER-fullweight /models/Qwen2.5-3B-Instruct-ER-fullweight")
    print("")
    
    vol.commit()


# ── Main pipeline function ──────────────────────────────────
@app.function(
    image=image,
    gpu="A100",
    volumes={VOL_PATH: vol},
    timeout=36000,  # 10 hours max
)
def run_pipeline(model_name: str, skip_rdo: bool = False, skip_cones: bool = False, skip_repind: bool = False):
    """
    Run RDO + Cones + RepInd for a single model.
    
    Args:
        model_name: e.g. "Qwen2.5-3B-Instruct-ER-fullweight"
        skip_rdo: skip RDO training
        skip_cones: skip cone training
        skip_repind: skip RepInd training
    """
    import subprocess
    import torch
    
    # Reload volume to get latest files
    vol.reload()
    
    repo_dir = f"{VOL_PATH}/repo"
    model_path = f"{VOL_PATH}/models/{model_name}"
    
    # Verify files exist
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    dim_dir = f"{VOL_PATH}/results/dim_directions/{model_name}"
    assert os.path.exists(f"{dim_dir}/direction.pt"), f"DIM direction not found at {dim_dir}"
    assert os.path.exists(f"{dim_dir}/direction_metadata.json"), f"DIM metadata not found at {dim_dir}"
    assert os.path.exists(model_path), f"Model not found at {model_path}"
    print("Pre-flight checks passed")
    
    # Environment
    env = os.environ.copy()
    env.update({
        "HUGGINGFACE_CACHE_DIR": f"{VOL_PATH}/cache",
        "HF_HOME": f"{VOL_PATH}/cache",
        "TRANSFORMERS_CACHE": f"{VOL_PATH}/cache",
        "SAVE_DIR": f"{VOL_PATH}/results",
        "DIM_DIR": "dim_directions",
        "WANDB_MODE": "offline",
        "WANDB_DIR": f"{VOL_PATH}/results/wandb",
        "PYTHONPATH": repo_dir,
    })
    
    os.makedirs(f"{VOL_PATH}/results/wandb", exist_ok=True)
    
    # Write .env for rdo.py's dotenv.load_dotenv
    with open(f"{repo_dir}/.env", "w") as f:
        f.write(f'HUGGINGFACE_CACHE_DIR="{VOL_PATH}/cache"\n')
        f.write(f'SAVE_DIR="{VOL_PATH}/results"\n')
        f.write(f'DIM_DIR="dim_directions"\n')
        f.write(f'WANDB_ENTITY="mpinero-princeton-university"\n')
        f.write(f'WANDB_PROJECT="refusal_directions"\n')
        f.write(f'WANDB_MODE=offline\n')
    
    def run_cmd(cmd, label):
        print(f"\n{'=' * 60}")
        print(f"{label}")
        print(f"{'=' * 60}\n")
        result = subprocess.run(
            cmd, shell=True, cwd=repo_dir, env=env,
            capture_output=False, text=True,
        )
        # Commit after each step so results persist even if later steps fail
        vol.commit()
        if result.returncode != 0:
            print(f"WARNING: {label} exited with code {result.returncode}")
        return result.returncode
    
    # ── 1. RDO ──────────────────────────────────────────────
    if not skip_rdo:
        run_cmd(
            f"python rdo.py "
            f"--model {model_path} "
            f"--train_direction "
            f"--epochs 1 --lr 0.01 --batch_size 1 --effective_batch_size 16 "
            f"--patience 5 --n_lr_reduce 2 "
            f"--ablation_lambda 1.0 --addition_lambda 0.2 --retain_lambda 1.0",
            "[1/3] RDO direction"
        )
    
    # ── 2. Cones ────────────────────────────────────────────
    if not skip_cones:
        run_cmd(
            f"python rdo.py "
            f"--model {model_path} "
            f"--train_cone "
            f"--min_cone_dim 2 --max_cone_dim 6 "
            f"--epochs 1 --lr 0.01 --batch_size 1 --effective_batch_size 16 "
            f"--patience 5 --n_lr_reduce 2 "
            f"--ablation_lambda 1.0 --addition_lambda 0.2 --retain_lambda 1.0 "
            f"--n_sample 8 --fixed_samples 8",
            "[2/3] Cone training (dim 2-6)"
        )
    
    # ── 3. RepInd ───────────────────────────────────────────
    if not skip_repind:
        run_cmd(
            f"python rdo_repind.py "
            f"--model {model_path} "
            f"--train_independent_direction "
            f"--epochs 2 --lr 0.01 --batch_size 1 --effective_batch_size 16 "
            f"--patience 5 --n_lr_reduce 2",
            "[3/3] RepInd training"
        )
    
    print(f"\n{'=' * 60}")
    print(f"ALL COMPLETE for {model_name}")
    print(f"{'=' * 60}")


# ── Download results ─────────────────────────────────────────
@app.function(volumes={VOL_PATH: vol}, timeout=600)
def download_results(model_name: str):
    """List results for a model so you know what to download."""
    vol.reload()
    
    rdo_dir = f"{VOL_PATH}/results/rdo/{model_name}"
    if os.path.exists(rdo_dir):
        print(f"\nRDO results at {rdo_dir}:")
        for root, dirs, files in os.walk(rdo_dir):
            for f in files:
                full = os.path.join(root, f)
                size = os.path.getsize(full) / 1024
                print(f"  {os.path.relpath(full, rdo_dir)}  ({size:.1f} KB)")
    else:
        print(f"No RDO results found for {model_name}")
    
    print("\nTo download results to your local machine:")
    print(f"  modal volume get geometry-vol /results/rdo/{model_name} ./results/rdo/{model_name}")
    print(f"  modal volume get geometry-vol /results/dim_directions/{model_name} ./results/dim_directions/{model_name}")


# ── Convenience: do everything ───────────────────────────────
@app.local_entrypoint()
def main(
    model_name: str = "Qwen2.5-3B-Instruct-ER-fullweight",
    skip_rdo: bool = False,
    skip_cones: bool = False,
    skip_repind: bool = False,
):
    """
    Run the full pipeline from the command line.
    
    Usage:
      modal run modal_geometry.py --model-name Qwen2.5-3B-Instruct-ER-fullweight
      modal run modal_geometry.py --model-name Qwen2.5-3B-Instruct-ER-fullweight --skip-repind
    """
    run_pipeline.remote(
        model_name=model_name,
        skip_rdo=skip_rdo,
        skip_cones=skip_cones,
        skip_repind=skip_repind,
    )
    download_results.remote(model_name=model_name)
