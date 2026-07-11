# Pre-Submission Testing Plan — AMD Hackathon Track 3

> Run everything with the project env: `/opt/anaconda3/envs/agentcir/bin/python`
> (aliased below as `$PY`).
>
> ☠️ = highest embarrassment risk if skipped (crashes live, or disqualifies).
>
> Architecture under test (two-step AMD flow):
> ```
> AMD MI300X (Jupyter, amd_serve_qwen.ipynb)          Mac (this repo)
> Qwen generates netlist on GPU  ──[.net file]──▶  LTspice verifies vs spec
>         rocm-smi proof shot                   report + data/verified_pairs.jsonl
> ```
> Sections A–E run entirely on the Mac. Section F is the AMD integration.
> Section G is the demo-recording checklist.

Track each test: `[ ]` not run · `[x]` pass · `[!]` fail (fix before submitting).
Completed-test evidence is consolidated in [TEST_RESULTS.md](TEST_RESULTS.md).

---

## A. Verifier core (UC1) — the foundation

| # | Test | How | Expect | Status |
|---|---|---|---|---|
| A1 | Known-good baseline | `$PY report_netlist.py data/templates/AD8092.net --metric gain_db=6.0:1.0` | PASS ~6.014 dB | [x] verified 2026-07-07 |
| A2 | Second part, pulse testbench | `$PY report_netlist.py data/templates/AD811.net --metric gain_db=6.02:1.0` | PASS ~6.461 dB (overshoot inflates ptp — expected) | [x] verified 2026-07-08 |
| A3 | AC sweep, 2 metrics, `meet_at_least` | AD811 AC variant: `--metric gain_db=6.02:0.5 --metric bandwidth_hz=100e6:0 --direction bandwidth_hz:meet_at_least` | Both PASS (gain 6.014, BW ~126 MHz) | [x] verified 2026-07-08 |
| A4 ☠️ | Wrong node name | Add `--out-node FOO` to A1 | Graceful N/A/FAIL listing available traces — **no stack trace** | [x] verified; [evidence](TEST_RESULTS.md#A-verifier-core-uc1) |
| A5 ☠️ | Garbage netlist | File containing random text | "FAIL (does not converge)" from syntax gate, clean message | [x] verified; [evidence](TEST_RESULTS.md#A-verifier-core-uc1) |
| A6 | Missing `.end` | Copy of AD8092.net minus last line | Caught by `parses_as_spice` | [x] verified; [evidence](TEST_RESULTS.md#A-verifier-core-uc1) |
| A7 ☠️ | Missing .lib/model | Netlist referencing `NONEXISTENT.lib` | LTspice error surfaced readably, no hang | [x] verified; [evidence](TEST_RESULTS.md#A-verifier-core-uc1) |
| A8 | Timeout | Long `.tran` + `--timeout 5` | Clean timeout error | [x] verified; [evidence](TEST_RESULTS.md#A-verifier-core-uc1) |
| A9 | Non-convergent circuit | e.g. voltage source loop | converged=False + log excerpt | [x] verified; [evidence](TEST_RESULTS.md#A-verifier-core-uc1) |
| A10 | Mixed pass/fail metrics | One passing + one failing metric | Exit code 1, both rows correct | [x] verified; [evidence](TEST_RESULTS.md#A-verifier-core-uc1) |
| A11 | Non-amp netlist | Regulator/ADC netlist, request `gain_db` | Graceful N/A, no crash | [x] verified; [evidence](TEST_RESULTS.md#A-verifier-core-uc1) |
| A12 | Unicode values | Corpus netlist using `µ` (e.g. `330µ`) | Parses and simulates | [x] verified; [evidence](TEST_RESULTS.md#A-verifier-core-uc1) |

## B. Retrieval / template index

| # | Test | How | Expect | Status |
|---|---|---|---|---|
| B1 | Canonical lookup | `$PY template_index.py AD8092` | AD8092.net listed first | [x] verified 2026-07-07 |
| B2 | Case-insensitive | `$PY template_index.py ad8092` | Same result | [x] verified; [evidence](TEST_RESULTS.md#B-retrieval--template-index) |
| B3 | Multi-template part | Any of the 337 multi-template parts | File named after part first | [x] verified; [evidence](TEST_RESULTS.md#B-retrieval--template-index) |
| B4 ☠️ | Unknown part | `$PY template_index.py XYZ999` | Clear "no template found", not KeyError | [x] verified; [evidence](TEST_RESULTS.md#B-retrieval--template-index) |

## C. Generate→Verify→Retry loop (UC2) — where the demo lives

> These run against whichever LLM backend `.env` points at. Run the suite
> once with OpenRouter (fast sanity), then re-run the key cases in the
> two-step AMD flow (Section F).

| # | Test | How | Expect | Status |
|---|---|---|---|---|
| C1 | Happy path | `$PY generate_verify.py AD8092 --spec "gain 5 V/V, ±5V, 100mV @ 1MHz test signal" --metric gain_db=13.98:1.0` | PASS, `data/verified_pairs.jsonl` appended | [x] verified 2026-07-07 (GPT-5) |
| C2 ☠️ | **Slew-rate demo case** | Same but spec pins "keep the 1V 10MHz test signal" | FAILs all attempts ~10.3 dB — THE demo moment; rehearse it | [x] reproduced 2026-07-07 |
| C3 | Impossible spec | gain 1000, single stage, wide BW | Honest FAIL after retries, exit 1, no fake PASS | [x] deterministic local loop; [evidence](TEST_RESULTS.md#C-generate--verify--retry-uc2) |
| C4 | Different part family | AD811 to gain 4 (`gain_db=12.04:1.0`) | Generalizes beyond AD8092 | [x] deterministic local candidate + real LTspice; rerun on AMD in F6; [evidence](TEST_RESULTS.md#C-generate--verify--retry-uc2) |
| C5 ☠️ | LLM obeyed constraints | Diff generated vs template: `X` line pin order unchanged, `.lib` unchanged | No topology/pinout drift | [x] template guard rejects drift; [evidence](TEST_RESULTS.md#C-generate--verify--retry-uc2) |
| C6 | Fenced/chatty LLM output | Inspect any generation | `extract_netlist` strips fences + prose (use the `[a-zA-Z]*` fence regex — 1.5B emitted ```` ```netlist ````) | [x] verified; [evidence](TEST_RESULTS.md#C-generate--verify--retry-uc2) |
| C7 ☠️ | Bad API key / network down | Wrong `LLM_API_KEY`, rerun C1 | Clean `LLMError`, not a requests traceback | [x] verified; [evidence](TEST_RESULTS.md#C-generate--verify--retry-uc2) |
| C8 | Retry feedback helps | Find a spec that fails attempt 1, passes attempt 2 | Proves the loop does something; save the log — it's a slide stat | [x] deterministic local retry; [evidence](TEST_RESULTS.md#C-generate--verify--retry-uc2) |
| C9 | verified_pairs.jsonl integrity | `$PY -c "import json;[json.loads(l) for l in open('data/verified_pairs.jsonl')]"` | Every line valid JSON, numbers match reports | [x] verified; [evidence](TEST_RESULTS.md#C-generate--verify--retry-uc2) |

## D. GUI — judges will see this; it must not flinch

| # | Test | How | Expect | Status |
|---|---|---|---|---|
| D1 | Load → simulate → plot | 3–4 different netlists | Waveforms render | [x] automated GUI integration; complete the visible rehearsal below; [evidence](TEST_RESULTS.md#D-gui-automated-state-and-integration-checks) |
| D2 ☠️ | Verify-tab garbage inputs | Empty target, `abc` tol, negative tol, duplicate metric names | Validation message, no crash | [x] verified; [evidence](TEST_RESULTS.md#D-gui-automated-state-checks) |
| D3 | Add/remove metrics repeatedly | Re-run verification after each change | No stale results | [x] verified; [evidence](TEST_RESULTS.md#D-gui-automated-state-checks) |
| D4 | Responsiveness | Click around during a long sim | GUI stays alive (async thread) | [ ] |
| D5 | Wrong in/out node in GUI fields | e.g. out node `FOO` | Same graceful N/A as A4 | [x] verified; [evidence](TEST_RESULTS.md#D-gui-automated-state-checks) |
| D6 ☠️ | Full agent flow (demo path) | ADI LLM chat → netlist → Verify Spec tab → PASS report | The exact sequence you'll record — rehearse until boring | [ ] |
| D7 | Combine Circuits tab | 2 netlists | Well-formed output; simulate it | [x] offline deterministic merge path; live LLM merge still pending; [evidence](TEST_RESULTS.md#D-gui-automated-state-and-integration-checks) |
| D8 ☠️ | State carryover | Verify twice in a row; then load a 2nd netlist and verify | Results match the *current* netlist | [x] verified; [evidence](TEST_RESULTS.md#D-gui-automated-state-checks) |

## E. Environment / submission hygiene

| # | Test | How | Expect | Status |
|---|---|---|---|---|
| E1 ☠️ | Fresh-clone test | New conda env, `pip install -r requirements.txt`, follow README | Everything imports and runs — this is what judges do | [!] preflight failed; replace machine-specific requirements before submitting; [evidence](TEST_RESULTS.md#E-environment-and-submission-hygiene) |
| E2 | Path with spaces | Run A1 from a dir with spaces | LTspice paths resolve | [x] verified; [evidence](TEST_RESULTS.md#E-environment-and-submission-hygiene) |
| E3 | No `.env` | Temporarily rename it | Warnings, not crashes | [x] verified without touching the real `.env`; [evidence](TEST_RESULTS.md#E-environment-and-submission-hygiene) |
| E4 ☠️ | **Revoke + remove hardcoded OpenRouter keys** | `experiments/test_harness.py:25` (+ commented key below it) | Keys revoked on OpenRouter (they're in git history — revoking is the only real fix), lines stripped | [!] source cleanup complete; key revocation remains required; [evidence](TEST_RESULTS.md#E-environment-and-submission-hygiene) |

## F. AMD integration (two-step flow) — the netlist generation MUST visibly come from Qwen on the MI300X

> Goal: make it undeniable in repo + video that **generation happens on AMD
> silicon**. The two-step flow is: run `amd_serve_qwen.ipynb` on the box,
> generate netlists there, bring them to the Mac, verify with UC1.

| # | Test | How | Expect | Status |
|---|---|---|---|---|
| F1 | Model loads on GPU | Notebook Step 1 with `Qwen/Qwen2.5-Coder-7B-Instruct` | `Loaded on: cuda:0` | [ ] 1.5B done; redo with 7B |
| F2 ☠️ | **Correct generation from 7B** | Generation cell, AD8092 gain-5 spec | `R2` → `4K` with `R1` unchanged (1.5B failed this: changed both to 100K, ratio still 2) | [ ] |
| F3 ☠️ | **rocm-smi proof shot** | `watch rocm-smi` in a terminal during F2 | GPU%/VRAM% spike captured as screenshot AND screen recording | [ ] |
| F4 | Generated file verifies on Mac | Download `.net`, then `$PY report_netlist.py AD8092_gain5.net --metric gain_db=13.98:1.0 --freq 1e6` | PASS ~13.9 dB | [ ] |
| F5 | Manual retry loop demo | If F4 fails, paste the measured report back into the notebook prompt, regenerate, re-verify | Shows the full loop across the two machines — narrate it in the video | [ ] |
| F6 | Second part on AMD | Repeat F2/F4 for AD811 gain 4 | Generalization, on AMD | [ ] |
| F7 | Batch generation | Generate 3–5 spec variants in one notebook run, verify all on Mac | Feeds `data/verified_pairs.jsonl`; mini flywheel evidence | [ ] |
| F8 ☠️ | README "AMD Deployment" section | Notebook path, model, rocm-smi screenshot, two-step diagram | The automated pre-screen reads the repo for AMD usage — this is what it finds | [ ] |

## G. Demo-recording checklist (the loop must be VISIBLE)

Shot list — record after C-suite and F-suite are green:

1. **Hook (Mac, GUI):** load a spec in the agent chat → ask for AD8092 gain 5.
2. **AMD proof (box):** side-by-side JupyterLab generation cell running +
   terminal with `rocm-smi` spiking. Say the words "Qwen 7B running on an
   AMD MI300X, 192 GB." Keep the instance hostname visible.
3. **The verdict (Mac):** verify the generated netlist → PASS report with
   measured 13.9 dB and margin, waveform on screen.
4. **The money shot (Mac):** the slew-rate case — GPT-5/Qwen makes the
   textbook-correct edit, simulation FAILs it at 10.3 dB, you explain the
   physics in one sentence ("the part can't slew 314 V/µs; only the
   simulator knows"). Then the corrected spec PASSes.
5. **The flywheel (Mac):** `data/verified_pairs.jsonl` scrolling — "every verified
   answer becomes training data."
6. Keep each shot ≤ 45s; total ≤ 4 min. No dead air while sims run — cut.

Submission checklist: public GitHub repo (E4 done first!), demo video,
slide deck PDF (see SLIDE_DECK_BRIEF.md), optional hosted URL.
