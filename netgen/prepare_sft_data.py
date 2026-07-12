"""
prepare_sft_data.py
===================
Turn (instruction, design_plan, netlist) records into SFT training data for a
`prompt -> design plan -> netlist` model.

Three things this file does:
  1. Assemble structured records into the field-marked text the model is trained on.
  2. Set up COMPLETION-ONLY loss masking (loss only on plan+netlist, not the instruction).
  3. Generate design plans for netlists that lack them (back-translation) and
     CONSISTENCY-CHECK them so you don't poison the dataset.

Dependencies (install as needed):
    pip install datasets transformers trl
Optional (for back-translation):
    an LLM client of your choice (API or local). A stub `call_llm` is provided.

NOTE: validate every assembled example against your real PDK before training.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional

# Field markers the model learns to produce. Keep them stable across the whole project.
PLAN_HEADER = "### DESIGN PLAN"
NETLIST_HEADER = "### NETLIST"
INSTR_HEADER = "### INSTRUCTION"


# ----------------------------------------------------------------------------
# 1. Assemble records into prompt / completion text
# ----------------------------------------------------------------------------
def build_prompt(instruction: str) -> str:
    """The part the model CONDITIONS ON (loss is masked here)."""
    return f"{INSTR_HEADER}\n{instruction.strip()}\n\n{PLAN_HEADER}\n"


def build_completion(design_plan: str, netlist: str) -> str:
    """The part the model must PRODUCE (loss is computed here)."""
    return f"{design_plan.strip()}\n\n{NETLIST_HEADER}\n{netlist.strip()}"


def build_example_text(record: dict) -> str:
    """Full prompt+completion string (for raw-text SFT)."""
    return build_prompt(record["instruction"]) + build_completion(
        record["design_plan"], record["netlist"]
    )


def load_records(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ----------------------------------------------------------------------------
# 2. Completion-only masking with TRL
# ----------------------------------------------------------------------------
# We want loss ONLY on the completion (plan + netlist), NOT on the instruction.
# Training on instruction tokens wastes capacity and can hurt instruction-following.
#
# With trl.SFTTrainer you do this with DataCollatorForCompletionOnlyLM by giving it
# the response template that marks where the completion begins. Here, the completion
# begins right after the PLAN_HEADER (because the plan is the first thing produced).
#
# Example wiring (pseudo-real; adapt to your trl version):
#
#   from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
#   from transformers import AutoTokenizer
#   from datasets import Dataset
#
#   tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B")
#   records = load_records("training_data_format.jsonl")
#   ds = Dataset.from_list([{"text": build_example_text(r)} for r in records])
#
#   # The collator masks everything up to and including this template:
#   collator = DataCollatorForCompletionOnlyLM(
#       response_template=f"{PLAN_HEADER}\n",   # loss starts after the plan header
#       tokenizer=tok,
#   )
#
#   trainer = SFTTrainer(
#       model="Qwen/Qwen2.5-Coder-7B",
#       train_dataset=ds,
#       data_collator=collator,
#       args=SFTConfig(
#           output_dir="sft-spice-ota",
#           per_device_train_batch_size=2,
#           gradient_accumulation_steps=8,
#           num_train_epochs=3,
#           learning_rate=2e-4,            # typical LoRA LR
#           bf16=True,
#           max_seq_length=2048,
#           # LoRA/QLoRA config goes here (peft_config=LoraConfig(...)) or via PEFT.
#       ),
#   )
#   trainer.train()
#
# For QLoRA specifically: load the base model 4-bit (bitsandbytes) and pass a
# LoraConfig (r=16-32, target the attention + MLP projections). Unsloth is a
# drop-in faster path if you want 2-5x speedup / less VRAM.


# ----------------------------------------------------------------------------
# 3. Back-translation: generate a plan for a netlist that has none
# ----------------------------------------------------------------------------
# You have netlists but not design plans. Two ways to get plans:
#   (a) back-translation (this function) -- cheap, for your EXISTING netlists.
#   (b) forward RSFT -- higher quality, for NEW data (model writes plan->netlist,
#       you simulate and keep passers). Prefer (b) once the SFT model exists.
#
# CRITICAL: unverified back-translation poisons the dataset. Always run the
# consistency check below and DROP inconsistent examples.

BACKTRANSLATE_PROMPT = """You are an analog IC designer. Below is a working SPICE netlist.
Write the DESIGN PLAN that explains it: the topology, the role of each device,
how the sizing relates to the target spec, and the operating region of each device.
Do NOT restate the netlist. Be concise and technically precise.

SPEC (if known):
{spec}

NETLIST:
{netlist}
"""


def call_llm(prompt: str) -> str:
    """STUB: wire this to your LLM (API or local). Must return plain text."""
    raise NotImplementedError("Connect call_llm() to your model of choice.")


def backtranslate_plan(netlist: str, spec: str = "(not provided)") -> str:
    return call_llm(BACKTRANSLATE_PROMPT.format(spec=spec, netlist=netlist)).strip()


# ----------------------------------------------------------------------------
# Consistency check: does the plan actually match the netlist?
# ----------------------------------------------------------------------------
DEVICE_RE = re.compile(r"^\s*([MmRrCcLlDdQqVvIiXx]\w*)\s+", re.MULTILINE)
MOS_RE = re.compile(r"^\s*[Mm]\w*\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)", re.MULTILINE)


def parse_device_names(netlist: str) -> set[str]:
    return {m.group(1).upper() for m in DEVICE_RE.finditer(netlist)}


def parse_mos_models(netlist: str) -> set[str]:
    return {m.group(1).lower() for m in MOS_RE.finditer(netlist)}


@dataclass
class ConsistencyResult:
    ok: bool
    reasons: list[str]


def check_plan_netlist_consistency(plan: str, netlist: str) -> ConsistencyResult:
    """
    Lightweight structural agreement check between a (back-translated) plan and its
    netlist. This is NOT a simulator -- it just catches plans that describe a
    different circuit than the one in the netlist. Tune thresholds to taste.
    """
    reasons: list[str] = []
    plan_l = plan.lower()

    all_devices = parse_device_names(netlist)        # e.g. {"M1","M2","RD","VDD",...}
    # Only require the plan to discuss SIGNAL devices (transistors, R, C, L, D, Q,
    # subckt). Supply/stimulus sources (V*, I*) are legitimately omitted by name.
    signal_devices = {d for d in all_devices if d[0] not in ("V", "I")}
    mentioned = {d for d in signal_devices if d.lower() in plan_l}
    if signal_devices:
        coverage = len(mentioned) / len(signal_devices)
        if coverage < 0.5:
            reasons.append(
                f"plan references only {len(mentioned)}/{len(signal_devices)} signal "
                f"devices ({coverage:.0%}) -- likely describes a different circuit"
            )

    # Device-type sanity: if netlist has PMOS/NMOS, plan should not contradict it.
    models = parse_mos_models(netlist)
    has_n = any("nfet" in m or m.startswith("nmos") for m in models)
    has_p = any("pfet" in m or m.startswith("pmos") for m in models)
    if has_p and "pmos" not in plan_l and "p-mos" not in plan_l and "pfet" not in plan_l:
        reasons.append("netlist uses PMOS but plan never mentions PMOS")
    if has_n and "nmos" not in plan_l and "n-mos" not in plan_l and "nfet" not in plan_l:
        reasons.append("netlist uses NMOS but plan never mentions NMOS")

    # Flag a hallucinated compensation cap only on POSITIVE evidence (so "no
    # compensation needed" doesn't trip it). Triggers if the plan asserts a comp
    # capacitor but the netlist has no capacitor element.
    asserts_comp_cap = any(
        k in plan_l for k in ("compensation capacitor", "miller cap", "miller compensation")
    )
    if asserts_comp_cap and not re.search(r"^\s*[Cc]\w*\s", netlist, re.MULTILINE):
        reasons.append("plan claims a compensation capacitor but netlist has no capacitor")

    return ConsistencyResult(ok=(len(reasons) == 0), reasons=reasons)


# ----------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    recs = load_records("training_data_format.jsonl")
    print(f"Loaded {len(recs)} records.\n")

    # Show one assembled training example.
    print("=" * 70)
    print("ASSEMBLED EXAMPLE (prompt is masked, completion is trained):")
    print("=" * 70)
    print(build_example_text(recs[0])[:1200], "...\n")

    # Run the consistency check on the (already-paired) records as a self-test.
    print("=" * 70)
    print("CONSISTENCY CHECK (should pass for these hand-paired records):")
    print("=" * 70)
    for r in recs:
        res = check_plan_netlist_consistency(r["design_plan"], r["netlist"])
        status = "OK " if res.ok else "DROP"
        print(f"[{status}] {r['id']}  {('' if res.ok else '-> ' + '; '.join(res.reasons))}")
