# AMD MI300X Deployment Guide

This repository demonstrates AMD usage through Qwen inference on an AMD MI300X and real LTspice verification on the local workstation. It supports both single-candidate verification and a best-of-N Candidate Arena flow.

## Goal

```text
MI300X: Qwen generates a constrained netlist adaptation
                     │
                     ▼
Mac: LTspice simulates, measures, and accepts or rejects the candidate
```

The model proposes; the simulator decides.

> **Live alternative:** [../notebooks/amd_serve_and_tunnel.ipynb](../notebooks/amd_serve_and_tunnel.ipynb)
> serves Qwen from the MI300X as a token-protected OpenAI-compatible endpoint
> through an HTTPS/443 tunnel, so the GUI chat and `generate_verify.py` call AMD
> hardware directly (no copy-paste). The manual handoff below remains the
> zero-dependency fallback and produces identical evidence.

## 1. Prepare the MI300X notebook

Open [../notebooks/amd_serve_qwen.ipynb](../notebooks/amd_serve_qwen.ipynb) on the AMD system and run its cells from top to bottom.

Before the demo:

1. Select the intended Qwen model in the notebook.
2. Confirm `torch.cuda.is_available()` and the model device output show the AMD ROCm/HIP device.
3. Run `rocm-smi` in a separate terminal.
4. Run a short inference while recording or capturing the GPU memory/utilization increase.
5. Keep secrets private. The public tunnel, if used, must require a private session token.

## 2. Generate a constrained prompt locally

```bash
python generate_verify.py AD8092 \
  --spec "non-inverting gain of 5 V/V, +/-5 V supplies, 100 mV at 1 MHz" \
  --prompt-only > /tmp/ad8092_prompt.txt
```

Copy the contents of `/tmp/ad8092_prompt.txt` into Qwen on the MI300X. Save the full response as `/tmp/ad8092_qwen_response.txt` on the Mac.

## 3. Verify the Qwen response locally

```bash
python generate_verify.py AD8092 \
  --spec "non-inverting gain of 5 V/V, +/-5 V supplies, 100 mV at 1 MHz" \
  --metric gain_db=13.98:1.0 \
  --freq 1e6 \
  --candidate /tmp/ad8092_qwen_response.txt \
  --source amd_mi300x_manual \
  --save /tmp/ad8092_gain5.net
```

A pass logs a provenance-tagged record to `data/verified_pairs.jsonl`.

## 4. Retry demonstration

If the candidate fails:

1. Copy the measured report back into the Qwen conversation.
2. Ask Qwen to preserve the template topology and adjust only values.
3. Save the new candidate.
4. Re-run the verification command.

## 5. Best-of-N Candidate Arena demonstration

For a stronger AMD demonstration, run the same constrained prompt several
times on Qwen (for example, with different sampling seeds), saving each complete
response as a separate file on the Mac:

```bash
python generate_verify.py AD8092 \
  --spec "non-inverting gain of 5 V/V, +/-5 V supplies, 100 mV at 1 MHz" \
  --metric gain_db=13.98:1.0 \
  --freq 1e6 \
  --candidates /tmp/qwen_candidate_*.txt \
  --source amd_mi300x_manual \
  --report /tmp/ad8092_candidate_arena.json \
  --save /tmp/ad8092_best.net
```

The arena rejects topology/testbench drift before simulation, runs each valid
candidate in LTspice, reports measured values, and ranks results. A candidate
does not win because its language-model text looks good: a `PASS` against the
requested metric is still mandatory. The JSON report records template and
candidate hashes, provenance, measurements, timing, and the resulting ranking.

## 6. Required evidence for the hackathon

- Notebook showing the selected Qwen model loaded on the MI300X.
- `rocm-smi` screenshot or screen recording during generation.
- Generated netlist response.
- Local PASS/FAIL verification report.
- Candidate Arena report showing rejected, failed, and/or passing alternatives when using best-of-N.
- A `verified_pairs.jsonl` record for a passing candidate.

## Recommended demo sequence

1. Show the user specification.
2. Show Qwen generating several candidates on the MI300X with `rocm-smi` visible.
3. Show the candidate responses arriving on the Mac.
4. Run Candidate Arena and show the numerical ranking and evidence JSON.
5. Load the passing winner into the GUI and show the measured report.
6. Show an intentional slew-rate failure to demonstrate why simulation is necessary.
