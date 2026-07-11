# Optional Gemma LoRA adapter

This directory contains the released LoRA adapter for the optional local specialist model.

## Contents

- `adapter_config.json` — PEFT/LoRA configuration.
- `adapter_model.safetensors` — trained adapter weights, tracked with Git LFS.

## Important

This is **not** a standalone foundation model. It requires a compatible Gemma base model at runtime. The default loader reads the base model from `adapter_config.json`:

```text
unsloth/gemma-3-1b-it-unsloth-bnb-4bit
```

Override it when necessary:

```bash
export LOCAL_GEMMA_BASE_MODEL=<compatible-base-model-id-or-local-path>
```

## Retrieve weights after cloning

```bash
git lfs install
git lfs pull
```

Confirm that `adapter_model.safetensors` is hundreds of megabytes, not a short text file beginning with `version https://git-lfs.github.com/spec/v1`.

## Runtime status

The verifier, GUI simulator, and manual AMD-Qwen workflow do not depend on this adapter. Treat the local Gemma path as optional until it has been tested against the installed base model on the target machine.
