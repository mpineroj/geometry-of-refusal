"""
Download all ER model variants to /scratch/network/mp3687/models/

Models downloaded:
  1. rpawar7156/qwen2.5-3b-er  →  Qwen2.5-3B-Instruct-ER-fullweight
  2. CSMaya/er-ablations-qwen2.5-3b (explanation-only)
  3. CSMaya/er-ablations-qwen2.5-3b (justification-only)
  4. CSMaya/er-ablations-qwen2.5-3b (refusal-only)

Usage:
  1. Run on Adroit LOGIN node:
       conda activate new-rdo
       python download_er_models.py

All models are saved as full-weight checkpoints with tokenizer included.
The base Qwen2.5-3B-Instruct tokenizer is used for rpawar7156/qwen2.5-3b-er.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

SAVE_ROOT = "/scratch/network/mp3687/models"
CACHE = "/scratch/network/mp3687/.cache/huggingface"
BASE_TOKENIZER = "Qwen/Qwen2.5-3B-Instruct"

os.makedirs(SAVE_ROOT, exist_ok=True)

# ── 1. Riya's full-weight ER model ──────────────────────────────────
print("=" * 60)
print("1/4  rpawar7156/qwen2.5-3b-er")
print("=" * 60)

save_path = os.path.join(SAVE_ROOT, "Qwen2.5-3B-Instruct-ER-fullweight")
if os.path.exists(save_path):
    print(f"  already exists: {save_path}, skipping.\n")
else:
    print("  downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "rpawar7156/qwen2.5-3b-er",
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE,
    )
    model.save_pretrained(save_path, safe_serialization=True)
    del model
    torch.cuda.empty_cache()

    print("  saving tokenizer (from base Qwen2.5-3B-Instruct)...")
    tok = AutoTokenizer.from_pretrained(BASE_TOKENIZER, cache_dir=CACHE)
    tok.save_pretrained(save_path)
    print(f"  saved → {save_path}\n")

# ── 2-4. Chinmaya's component ablation variants ────────────────────
REPO = "CSMaya/er-ablations-qwen2.5-3b"
VARIANTS = [
    ("explanation-only-lora", "Qwen2.5-3B-Instruct-ER-fullweight-explanation-only"),
    ("justification-only-lora", "Qwen2.5-3B-Instruct-ER-fullweight-justification-only"),
    ("refusal-only-lora", "Qwen2.5-3B-Instruct-ER-fullweight-refusal-only"),
]

for i, (subdir, name) in enumerate(VARIANTS, start=2):
    print("=" * 60)
    print(f"{i}/4  {REPO}/{subdir}")
    print("=" * 60)

    save_path = os.path.join(SAVE_ROOT, name)
    if os.path.exists(save_path):
        print(f"  already exists: {save_path}, skipping.\n")
        continue

    print("  downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        REPO,
        subfolder=subdir,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE,
    )
    model.save_pretrained(save_path, safe_serialization=True)
    del model
    torch.cuda.empty_cache()

    print("  downloading tokenizer...")
    try:
        tok = AutoTokenizer.from_pretrained(REPO, subfolder=subdir, cache_dir=CACHE)
    except Exception:
        print("  tokenizer not in repo, using base Qwen2.5-3B-Instruct tokenizer")
        tok = AutoTokenizer.from_pretrained(BASE_TOKENIZER, cache_dir=CACHE)
    tok.save_pretrained(save_path)
    print(f"  saved → {save_path}\n")

# ── Summary ─────────────────────────────────────────────────────────
print("=" * 60)
print("Summary — models in", SAVE_ROOT)
print("=" * 60)
for entry in sorted(os.listdir(SAVE_ROOT)):
    full = os.path.join(SAVE_ROOT, entry)
    if os.path.isdir(full):
        n_files = len(os.listdir(full))
        print(f"  {entry}/  ({n_files} files)")
print("\nAll done.")
