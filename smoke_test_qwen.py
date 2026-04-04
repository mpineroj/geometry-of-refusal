"""
Smoke test for Qwen2.5-3B-Instruct in the geometry-of-refusal pipeline.

Run this BEFORE running run_pipeline.py to catch environment issues early.
Usage: python smoke_test_qwen.py

Checks:
  1. Model and tokenizer load correctly
  2. Refusal token IDs match expected strings
  3. Model attribute paths match what the wrapper expects (Qwen2.5 vs Qwen1.x)
  4. Chat template tokenizes correctly
  5. Forward pass produces activations of expected shape
  6. Can extract residual stream at specific (layer, token_position) pairs
"""

import torch
import sys

MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"

# Expected architecture specs for Qwen2.5-3B
EXPECTED_HIDDEN_DIM = 2048
EXPECTED_NUM_LAYERS = 36

# From qwen_model.py — the refusal token IDs used for filtering
QWEN_REFUSAL_TOKS = [40, 2121]

# Chat template from qwen_model.py
QWEN_CHAT_TEMPLATE_WITH_SYSTEM = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

passed = 0
failed = 0
warnings = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name}")
        if detail:
            print(f"     → {detail}")
        failed += 1


def warn(name, detail=""):
    global warnings
    print(f"  ⚠️  {name}")
    if detail:
        print(f"     → {detail}")
    warnings += 1


def main():
    global passed, failed, warnings

    # =========================================================================
    # 1. Load tokenizer
    # =========================================================================
    print("\n[1/6] Loading tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        check("Tokenizer loaded", True)
    except Exception as e:
        check("Tokenizer loaded", False, str(e))
        print("\nCannot proceed without tokenizer. Exiting.")
        sys.exit(1)

    # =========================================================================
    # 2. Verify refusal token IDs
    # =========================================================================
    print("\n[2/6] Checking refusal token IDs...")

    for tok_id in QWEN_REFUSAL_TOKS:
        decoded = tokenizer.decode([tok_id])
        check(
            f"Token {tok_id} → '{decoded}'",
            decoded.strip() in ("I", "As"),
            f"Expected 'I' or 'As', got '{decoded}'. "
            "You may need to update QWEN_REFUSAL_TOKS in qwen_model.py."
        )

    # Also check what token IDs common refusal starters actually map to
    print("\n  Reference — actual token IDs for common refusal prefixes:")
    for phrase in ["I", "As", "I'm", "I cannot", "Sorry"]:
        ids = tokenizer.encode(phrase, add_special_tokens=False)
        print(f"    '{phrase}' → {ids}")

    # =========================================================================
    # 3. Load model
    # =========================================================================
    print("\n[3/6] Loading model (this may take a minute)...")
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        check("Model loaded", True)
    except Exception as e:
        check("Model loaded", False, str(e))
        print("\nCannot proceed without model. Exiting.")
        sys.exit(1)

    # =========================================================================
    # 4. Verify model architecture paths
    # =========================================================================
    print("\n[4/6] Checking model attribute paths...")

    # -- Paths the ModelBase interface uses (should work for Qwen2.5) --
    check(
        "model.model.layers exists (ModelBase path)",
        hasattr(model, "model") and hasattr(model.model, "layers"),
        "QwenModel._get_model_block_modules() will fail"
    )

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
        check(
            f"Number of layers = {num_layers} (expected {EXPECTED_NUM_LAYERS})",
            num_layers == EXPECTED_NUM_LAYERS,
        )

        block = model.model.layers[0]

        check(
            "block.self_attn exists",
            hasattr(block, "self_attn"),
            "Attention module path is wrong"
        )
        check(
            "block.self_attn.o_proj exists",
            hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"),
            "Needed for orthogonalize_qwen_weights fix"
        )
        check(
            "block.mlp exists",
            hasattr(block, "mlp"),
            "MLP module path is wrong"
        )
        check(
            "block.mlp.down_proj exists",
            hasattr(block, "mlp") and hasattr(block.mlp, "down_proj"),
            "Needed for orthogonalize_qwen_weights fix"
        )

    # -- Paths the OLD standalone functions use (Qwen 1.x — should NOT exist) --
    has_old_transformer = hasattr(model, "transformer")
    if has_old_transformer:
        warn(
            "model.transformer exists — this is Qwen 1.x layout",
            "The orthogonalize/act_add functions might work as-is, "
            "but double-check which architecture you're actually using."
        )
    else:
        check(
            "model.transformer does NOT exist (confirms Qwen2.5 layout)",
            True,
        )
        warn(
            "orthogonalize_qwen_weights() and act_add_qwen_weights() use Qwen1.x paths",
            "These functions reference model.transformer.wte and model.transformer.h, "
            "which don't exist. They will crash if called. Fix them before running RDO."
        )

    # -- Check embedding layer --
    check(
        "model.model.embed_tokens exists",
        hasattr(model.model, "embed_tokens"),
        "Embedding path for orthogonalization fix"
    )

    # =========================================================================
    # 5. Chat template and tokenization
    # =========================================================================
    print("\n[5/6] Testing chat template and tokenization...")

    test_instruction = "What is the capital of France?"
    formatted = QWEN_CHAT_TEMPLATE_WITH_SYSTEM.format(
        system=SYSTEM_PROMPT,
        instruction=test_instruction,
    )

    tokens = tokenizer(formatted, return_tensors="pt")
    input_ids = tokens["input_ids"]

    check(
        f"Tokenized prompt has {input_ids.shape[1]} tokens",
        input_ids.shape[1] > 0,
    )

    # Check that special tokens are present
    decoded_back = tokenizer.decode(input_ids[0])
    check(
        "im_start token present in tokenized output",
        "<|im_start|>" in decoded_back,
        f"Decoded: {decoded_back[:100]}..."
    )

    # Check EOI tokens (end-of-instruction, used for token position extraction)
    eoi_template = QWEN_CHAT_TEMPLATE_WITH_SYSTEM.split("{instruction}")[-1]
    eoi_template_no_system = """<|im_end|>
<|im_start|>assistant
"""
    eoi_toks = tokenizer.encode(eoi_template_no_system, add_special_tokens=False)
    print(f"  EOI tokens: {eoi_toks} → '{tokenizer.decode(eoi_toks)}'")
    check(
        "EOI tokens are non-empty",
        len(eoi_toks) > 0,
    )

    # =========================================================================
    # 6. Forward pass and activation extraction
    # =========================================================================
    print("\n[6/6] Running forward pass and extracting activations...")

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    # hidden_states is a tuple of (num_layers + 1) tensors, each (batch, seq, hidden)
    hidden_states = outputs.hidden_states

    check(
        f"Got {len(hidden_states)} hidden state layers (expected {EXPECTED_NUM_LAYERS + 1})",
        len(hidden_states) == EXPECTED_NUM_LAYERS + 1,
    )

    # Check shape of a middle layer
    mid_layer = EXPECTED_NUM_LAYERS // 2
    if mid_layer < len(hidden_states):
        h = hidden_states[mid_layer]
        check(
            f"Layer {mid_layer} activation shape: {tuple(h.shape)}",
            h.shape[-1] == EXPECTED_HIDDEN_DIM,
            f"Expected hidden dim {EXPECTED_HIDDEN_DIM}, got {h.shape[-1]}"
        )

        # Extract at last token position (position -1) — this is what DIM does
        activation_at_last = h[0, -1, :]  # shape: (hidden_dim,)
        check(
            f"Activation at (layer={mid_layer}, pos=-1): shape {tuple(activation_at_last.shape)}, "
            f"dtype {activation_at_last.dtype}, norm {activation_at_last.float().norm().item():.2f}",
            activation_at_last.shape[0] == EXPECTED_HIDDEN_DIM
            and activation_at_last.float().norm().item() > 0,
        )

    # Quick generation test
    print("\n  Quick generation test (greedy, 20 tokens)...")
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=20,
            do_sample=False,
        )
    gen_text = tokenizer.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Prompt: '{test_instruction}'")
    print(f"  Output: '{gen_text}'")
    check("Model generates non-empty output", len(gen_text.strip()) > 0)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {warnings} warnings")
    print("=" * 60)

    if failed == 0:
        print("\n  All checks passed. You're ready to run the DIM pipeline.")
        print("  Next: python pipeline/run_pipeline.py --model_path Qwen/Qwen2.5-3B-Instruct")
    else:
        print(f"\n  {failed} check(s) failed. Fix the issues above before running the pipeline.")

    if warnings > 0:
        print(f"  {warnings} warning(s) — review before proceeding to RDO.")

    print()


if __name__ == "__main__":
    main()
