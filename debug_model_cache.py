#!/usr/bin/env python3
"""
Small debug helper to run on the head node / login node.
Checks HF cache layout, finds model snapshot path, and tries to
load the model config locally (offline).

Usage examples:
  python debug_model_cache.py --model Qwen/Qwen2.5-3B-Instruct
  python debug_model_cache.py --model rpawar7156/qwen2.5-3b-er --cache-dir /scratch/network/$USER/.cache/huggingface

The script does NOT attempt to load full weights — only local files (config/tokenizer).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

DEFAULT_CANDIDATES = [
    os.environ.get("HUGGINGFACE_CACHE_DIR"),
    os.environ.get("HF_HOME"),
    os.environ.get("TRANSFORMERS_CACHE"),
    os.path.expanduser("~/ .cache/huggingface"),
    os.path.expanduser("~/.cache/huggingface"),
    f"/scratch/network/{os.environ.get('USER', '')}/.cache/huggingface",
]


def find_cache_root(explicit: str | None) -> str | None:
    if explicit:
        if os.path.isdir(explicit):
            return explicit
        else:
            return None
    for c in DEFAULT_CANDIDATES:
        if not c:
            continue
        c = os.path.expanduser(c)
        c = c.replace("~/ ", "~/")
        if os.path.isdir(c):
            return c
    return None


def find_model_cache(cache_root: str, model: str) -> dict:
    """Return information about where model files live in cache.
    Looks for directory name 'models--org--name' and picks a snapshot.
    Also accepts when `model` is already a local path.
    """
    model_info = {"found": False, "reason": None, "model_path": None, "snapshot": None, "cache_dir": None}

    # If the user passed a local path, prefer that
    if os.path.isdir(model):
        model_info.update({"found": True, "model_path": model, "reason": "local_path"})
        return model_info

    model_dirname = f"models--{model.replace('/', '--')}"
    candidate = os.path.join(cache_root, model_dirname)
    model_info["cache_dir"] = candidate

    if not os.path.isdir(candidate):
        model_info["reason"] = f"cache-dir-missing: {candidate}"
        return model_info

    # Look for refs/main -> snapshot id, or latest snapshot directory
    refs_main = os.path.join(candidate, "refs", "main")
    if os.path.isfile(refs_main):
        try:
            snap_id = open(refs_main, "r").read().strip()
            snap_dir = os.path.join(candidate, "snapshots", snap_id)
            if os.path.isdir(snap_dir):
                model_info.update({"found": True, "model_path": snap_dir, "snapshot": snap_id})
                return model_info
        except Exception:
            pass

    # fallback: newest entry in snapshots/
    snapshots_dir = os.path.join(candidate, "snapshots")
    if os.path.isdir(snapshots_dir):
        snaps = sorted(
            [os.path.join(snapshots_dir, d) for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))],
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        if snaps:
            model_info.update({"found": True, "model_path": snaps[0], "snapshot": os.path.basename(snaps[0])})
            return model_info

    # Some caches may store files directly under the model dir (no snapshots)
    if any(os.path.exists(os.path.join(candidate, fn)) for fn in ["config.json", "tokenizer.json", "pytorch_model.bin", "pytorch_model.safetensors"]):
        model_info.update({"found": True, "model_path": candidate, "snapshot": None})
        return model_info

    model_info["reason"] = "no-snapshot-found"
    return model_info


def try_load_config(path: str) -> dict:
    out = {"loaded": False, "error": None}
    try:
        # Import lazily to provide a friendly error message
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(path, local_files_only=True)
        out.update({"loaded": True, "config": cfg.to_dict()})
    except Exception as e:
        out.update({"loaded": False, "error": traceback.format_exc()})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Model identifier (org/name) or local path")
    p.add_argument("--cache-dir", help="HuggingFace cache root (overrides env)")
    args = p.parse_args()

    cache_root = find_cache_root(args.cache_dir)
    print("Environment variables:")
    for k in ["HUGGINGFACE_CACHE_DIR", "HF_HOME", "TRANSFORMERS_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]:
        print(f"  {k}={os.environ.get(k)!r}")
    print()

    if not cache_root:
        print("No huggingface cache root found. Try passing --cache-dir or set HUGGINGFACE_CACHE_DIR.")
        sys.exit(2)

    print(f"Using cache root: {cache_root}")
    info = find_model_cache(cache_root, args.model)
    print(json.dumps(info, indent=2))

    if not info.get("found"):
        print("Model not found in cache; reason:", info.get("reason"))
        print("Cache listing (top-level):")
        try:
            for entry in sorted(os.listdir(cache_root)):
                print("  ", entry)
        except Exception:
            pass
        sys.exit(3)

    model_path = info["model_path"]
    print(f"Model path chosen: {model_path}")

    # quick file checks
    for fn in ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"]:
        print(f"  {fn}:", os.path.exists(os.path.join(model_path, fn)))

    print("\nAttempting to load config locally with Transformers (local_files_only=True)")
    load_res = try_load_config(model_path)
    if load_res["loaded"]:
        print("Loaded config keys:", list(load_res["config"].keys())[:10])
    else:
        print("Failed to load config. Error (truncated):")
        print(load_res["error"]) 

    print("\nRecommended sbatch invocation examples:")
    print(f"  MODEL=\"{args.model}\" sbatch run_rdo_test.slurm")
    print(f"  # or, when you already have a local snapshot path:\n  python rdo.py --model {model_path} --train_direction --epochs 1")


if __name__ == '__main__':
    main()
