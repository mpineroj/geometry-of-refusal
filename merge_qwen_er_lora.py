from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_PATH = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_REPO = "CSMaya/er_ablations_qwen2.5_3b"
ADAPTER_SUBFOLDER = "justification-only-lora"

CACHE_DIR = Path(".hf_cache").resolve()
SAVE_PATH = Path("models/Qwen2.5-3B-Instruct-ER-justification-only").resolve()


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        cache_dir=str(CACHE_DIR),
        low_cpu_mem_usage=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base,
        ADAPTER_REPO,
        subfolder=ADAPTER_SUBFOLDER,
        cache_dir=str(CACHE_DIR),
    )

    print("Merging adapter into base weights...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to: {SAVE_PATH}")
    merged.save_pretrained(str(SAVE_PATH), safe_serialization=True)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, cache_dir=str(CACHE_DIR))
    tokenizer.save_pretrained(str(SAVE_PATH))

    print("Done.")


if __name__ == "__main__":
    main()
