# Optional training utilities

This directory intentionally contains source code only. It excludes datasets, checkpoints, optimizer state, compiled caches, and experiment output.

## Files

- `Ft_data.py` — creates or cleans JSONL data from template netlists.
- `fine_tuning_script_general.py` — optional SFT/LoRA training script.

## Prepare a dataset

```bash
python training/Ft_data.py \
  --net_folder data/templates \
  --output /tmp/spice_wizard_dataset.jsonl
```

## Training environment

Training is optional and requires a GPU-specific environment. Install `requirements-training.txt` only in an appropriate CUDA or ROCm environment, then follow the platform instructions for Unsloth and PyTorch.

Do not commit generated datasets, checkpoints, caches, or training logs. They are explicitly ignored by `.gitignore`.
