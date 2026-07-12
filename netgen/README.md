# netgen — UC3 scaffolding (signal-chain synthesis roadmap)

Groundwork for the next stage beyond single-part adaptation: composing
sensor → amp → filter → ADC signal chains from generic sub-circuit blocks and
training on simulation-verified data.

| File | Purpose |
|---|---|
| `grpo_reward.py` | Reward shaping that scores generated netlists with the same LTspice measurements used by the verifier |
| `prepare_sft_data.py` | Converts verified `(spec, netlist, report)` records into SFT training rows |
| `sim_harness.py` | Standalone simulation harness variant used by the reward code |
| `training_data_format.jsonl` | Example of the training-row schema |

Status: scaffolding — not wired into the main flow yet. The main code path is
documented in the repository [README](../README.md).
