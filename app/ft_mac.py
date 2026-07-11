"""Optional local Gemma LoRA generator.

The primary hackathon path is Qwen inference on AMD followed by local LTspice
verification. This module is deliberately optional and is loaded only when the
GUI's `SPICE_WIZARD_ENABLE_LOCAL_AGENT` flag is enabled.
"""

import os
from pathlib import Path

# Must be set before torch/transformers import on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalGemmaGen:
    """Load the bundled LoRA adapter over a compatible Gemma base model."""

    def __init__(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        self.adapter = repo_root / "models" / "gemma3-lora"
        adapter_weights = self.adapter / "adapter_model.safetensors"
        if not adapter_weights.is_file() or adapter_weights.stat().st_size < 1_000_000:
            raise FileNotFoundError(
                "The local Gemma adapter is unavailable. Run `git lfs pull` in the repository "
                "and confirm models/gemma3-lora/adapter_model.safetensors is a real binary file."
            )
            adapter_config = PeftConfig.from_pretrained(str(self.adapter))
            self.base_id = os.getenv("LOCAL_GEMMA_BASE_MODEL", adapter_config.base_model_name_or_path)

        if torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float16
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        print(f"Loading local Gemma base model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_id)
        base = AutoModelForCausalLM.from_pretrained(self.base_id, dtype=self.dtype)
        base.to(self.device)
        self.model = PeftModel.from_pretrained(base, str(self.adapter)).merge_and_unload()
        self.model.eval()
        print("Local Gemma adapter loaded and merged.")

    def generate_netlist(self, ic_name: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are an expert at generating LTspice netlists. Return only the netlist.",
            },
            {"role": "user", "content": f"Give me a netlist for {ic_name}."},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True)
        return self.tokenizer.decode(output[0][input_length:], skip_special_tokens=True).strip()

