from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE = 'Qwen/Qwen2.5-3B-Instruct'
CACHE = '.hf_cache'
MODELS_DIR = '/models'

variants = [
    ('rpawar7156/qwen25-3b-instruct-er-full-lora', None, 'Qwen2.5-3B-Instruct-ER'),
    ('CSMaya/er_ablations_qwen2.5_3b', 'justification-only-lora', 'Qwen2.5-3B-Instruct-ER-justification-only'),
    ('CSMaya/er_ablations_qwen2.5_3b', 'explanation-only-lora', 'Qwen2.5-3B-Instruct-ER-explanation-only'),
    ('CSMaya/er_ablations_qwen2.5_3b', 'refusal-only-lora', 'Qwen2.5-3B-Instruct-ER-refusal-only'),
]

tokenizer = AutoTokenizer.from_pretrained(BASE, cache_dir=CACHE)

for adapter_repo, subfolder, save_name in variants:
    print(f'\n=== Merging {save_name} ===')
    save_path = f'{MODELS_DIR}/{save_name}'
    
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map='cpu', cache_dir=CACHE, low_cpu_mem_usage=True)
    
    kwargs = {'cache_dir': CACHE}
    if subfolder:
        kwargs['subfolder'] = subfolder
    model = PeftModel.from_pretrained(base, adapter_repo, **kwargs)
    merged = model.merge_and_unload()
    
    merged.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)
    print(f'Saved to {save_path}')
    
    del base, model, merged
    torch.cuda.empty_cache()

print('\nAll models merged.')