"""
Evaluate all RDO/cone vectors in parallel on Modal.

Usage:
  modal run modal_eval.py --model-name Qwen2.5-3B-Instruct-ER-fullweight

This will:
  1. Discover all vectors on the volume
  2. Launch one GPU per (vector, basis_idx) pair in parallel
  3. Save results back to the volume
  4. Print a summary table
"""

import modal
import os

app = modal.App("geometry-eval")

vol = modal.Volume.from_name("geometry-vol")
VOL_PATH = "/vol"

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


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOL_PATH: vol},
    timeout=3600,
)
def eval_one_vector(model_name: str, vector_file: str, basis_idx: int = -1):
    """Evaluate a single vector (or one basis vector from a cone) on JailbreakBench."""
    import subprocess
    import json

    vol.reload()

    repo_dir = f"{VOL_PATH}/repo"
    model_path = f"{VOL_PATH}/models/{model_name}"
    vector_path = f"{VOL_PATH}/results/rdo/{model_name}/vectors/{vector_file}"
    dim_metadata_path = f"{VOL_PATH}/results/dim_directions/{model_name}/direction_metadata.json"

    # Build unique output dir
    vector_stem = vector_file.replace(".pt", "")
    suffix = f"_basis{basis_idx}" if basis_idx >= 0 else ""
    output_dir = f"{VOL_PATH}/results/rdo/{model_name}/vectors/eval_{vector_stem}{suffix}"

    # Write .env
    with open(f"{repo_dir}/.env", "w") as f:
        f.write(f'HUGGINGFACE_CACHE_DIR="{VOL_PATH}/cache"\n')
        f.write(f'SAVE_DIR="{VOL_PATH}/results"\n')
        f.write(f'DIM_DIR="dim_directions"\n')
        f.write(f'WANDB_MODE=offline\n')

    env = os.environ.copy()
    env.update({
        "HUGGINGFACE_CACHE_DIR": f"{VOL_PATH}/cache",
        "HF_HOME": f"{VOL_PATH}/cache",
        "TRANSFORMERS_CACHE": f"{VOL_PATH}/cache",
        "SAVE_DIR": f"{VOL_PATH}/results",
        "DIM_DIR": "dim_directions",
        "PYTHONPATH": f"{repo_dir}/refusal_direction",
    })

    cmd = (
        f"cd {repo_dir}/refusal_direction && "
        f"python eval_rdo_direction.py "
        f"--model_path {model_path} "
        f"--vector_path {vector_path} "
        f"--dim_metadata_path {dim_metadata_path} "
        f"--output_dir {output_dir}"
    )

    if basis_idx >= 0:
        cmd += f" --basis_idx {basis_idx}"

    label = f"{vector_file} basis={basis_idx}" if basis_idx >= 0 else vector_file
    print(f"Evaluating: {label}")

    result = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr[-1000:])  # last 1000 chars of stderr

    vol.commit()

    # Read summary if it exists
    summary_path = f"{output_dir}/rdo_eval_summary.json"
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        return {"label": label, "summary": summary}
    else:
        return {"label": label, "error": f"exit code {result.returncode}"}


@app.local_entrypoint()
def main(model_name: str = "Qwen2.5-3B-Instruct-ER-fullweight"):
    """
    Discover all vectors and launch parallel evaluations.

    Usage:
      modal run modal_eval.py --model-name Qwen2.5-3B-Instruct-ER-fullweight
    """
    import json

    # Define the vectors and their basis indices
    # Update these based on your actual files
    jobs_by_model = {
    "Qwen2.5-3B-Instruct-ER-fullweight-refusal-only": [
        ("lowest_loss_vector_609whda5.pt", 0),
        ("lowest_loss_vector_609whda5.pt", 1),
    ],
    "Qwen2.5-3B-Instruct-ER-fullweight-explanation-only": [
        ("lowest_loss_vector_nzrk11yr.pt", 0),
        ("lowest_loss_vector_nzrk11yr.pt", 1),
    ],
    "Qwen2.5-3B-Instruct-ER-fullweight-justification-only": [
        ("lowest_loss_vector_oxrtzvq6.pt", 0),
        ("lowest_loss_vector_oxrtzvq6.pt", 1),
    ],
}

    jobs = jobs_by_model.get(model_name, [])
    if not jobs:
        print(f"No jobs defined for {model_name}")
        return

    # Launch all evaluations in parallel
    results = list(eval_one_vector.starmap(
        [(model_name, vf, bi) for vf, bi in jobs]
    ))

    # Print summary table
    print("\n" + "=" * 70)
    print(f"EVALUATION SUMMARY — {model_name}")
    print("=" * 70)
    print(f"{'Label':<45} {'Ablation ASR':>12}")
    print("-" * 70)

    for r in results:
        label = r["label"]
        if "summary" in r:
            asr = r["summary"]["results"].get("ablation", "N/A")
            print(f"{label:<45} {asr:>12}")
        else:
            print(f"{label:<45} {'FAILED':>12}")

    print("=" * 70)


