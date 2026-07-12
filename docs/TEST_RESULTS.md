# Verified Test Results — AMD Hackathon Track 3

**Consolidated:** 2026-07-11
**Environment:** macOS, `/opt/anaconda3/envs/agentcir/bin/python`, local LTspice  
**Scope:** Tests completed before the local-Gemma and AMD-MI300X work. This record preserves the outcomes that were previously only visible in terminal/session output.

> A deliberately failing specification test is marked **verified** when the expected failure was observed cleanly. It is not a product failure.

## A. Verifier core (UC1)

| # | Result | Evidence / observed outcome |
|---|---|---|
| A1 | Verified PASS | `AD8092`: gain = **6.014 dB** for target 6.0 dB ± 1.0 dB. |
| A2 | Verified PASS | `AD811` pulse testbench: gain = **6.461 dB** for target 6.02 dB ± 1.0 dB. The pulse overshoot increases the peak-to-peak measurement as expected. |
| A3 | Verified PASS | In-memory `AD811` AC variant: gain = **6.014 dB** and bandwidth = **125.989 MHz**; both specified metrics passed. |
| A4 | Verified expected failure | Invalid output node produced N/A/FAIL without a traceback and listed available trace names. |
| A5 | Verified expected failure | Garbage netlist was rejected cleanly by the syntax gate. |
| A6 | Verified expected failure | Netlist without `.end` was rejected cleanly by the syntax gate. |
| A7 | Verified expected failure | Missing library surfaced LTspice's `Fatal Error: Could not open library file` diagnostic. |
| A8 | Verified expected failure | Forced long simulation returned a clean timeout result. |
| A9 | Verified expected failure | Voltage-source loop returned non-convergence with the LTspice over-defined-matrix diagnostic. |
| A10 | Verified expected failure | One metric passed while unsupported transient-only bandwidth was reported N/A/FAIL; each row was correct independently. |
| A11 | Verified PASS | `AD590` ran and returned N/A for unsupported gain measurement instead of being misclassified as a malformed netlist. |
| A12 | Verified PASS | `ADP2504` with Unicode `µ` values parsed and simulated successfully. |

## B. Retrieval / template index

| # | Result | Evidence / observed outcome |
|---|---|---|
| B1 | Verified PASS | Canonical `AD8092.net` appeared first. |
| B2 | Verified PASS | Case-insensitive `ad8092` lookup returned the same canonical result. |
| B3 | Verified PASS | Corpus inspection found **337** multi-template parts; the canonical filename is prioritized. |
| B4 | Verified expected failure | `XYZ999` returned a clear no-template-found response rather than a `KeyError`. |

## C. Generate → verify → retry (UC2)

| # | Result | Evidence / observed outcome |
|---|---|---|
| C1 | Verified PASS | GPT-5 generation for the `AD8092` gain-5 specification passed LTspice verification and produced a verified-pair record. |
| C2 | Verified expected failure | The intentionally slew-limited 1 V / 10 MHz candidate measured approximately **10.305 dB**, below the approximately 13.98 dB requirement. No false PASS occurred. |
| C3 | Verified expected failure — deterministic local loop | An impossible gain target exhausted all configured retries and returned failure without a false PASS. Real LTspice was used; the scripted responses replaced only the external LLM call. |
| C4 | Verified PASS — deterministic local candidate | `AD811` gain-4 adaptation changed only `R2` from 649 Ω to 1.947 kΩ and passed real LTspice verification against 12.04 dB ± 1.0 dB. A live AMD/LLM rerun remains F6. |
| C5 | Verified PASS — constraint enforcement | Candidates that changed a subcircuit-call, `.lib`, or analysis line were rejected before simulation; a component-value-only correction was accepted. |
| C6 | Verified PASS | Netlist extraction accepted fenced `spice`/`netlist` responses and plain chatty responses without mistaking prose for a SPICE current-source line. |
| C7 | Verified expected failure | Mocked network/API failure returned clean `LLMError`; the generator CLI exited with code 2 rather than showing a `requests` traceback. |
| C8 | Verified PASS — deterministic retry | A first, slew-limited candidate failed; its measurement feedback was added to the retry prompt, and a corrected candidate passed on the next attempt. |
| C9 | Verified PASS | Existing verified-pair records parsed successfully as JSONL. |
| C10 | Verified PASS — Candidate Arena | An unchanged `AD8092` candidate measured **6.014 dB** and passed the 6.0 dB ± 1.0 dB target. Candidates that changed `.tran` to `.op` or rewired `R1` were rejected before LTspice. The JSON evidence schema is also covered by a simulator-free unit test. |

The deterministic C3–C5/C8 tests used real LTspice and mocked only the external LLM call. Their logging function was mocked, so they did not alter the persisted verified-pair dataset.

## D. GUI automated-state and integration checks

| # | Result | Evidence / observed outcome |
|---|---|---|
| D1 | Verified automated integration PASS | The GUI loaded, simulated, and rendered waveform plots for application netlists. The final regression covered `ADP2302`; earlier checks covered `AD8092`, `AD811`, and `ADP2504`. A visible on-screen rehearsal remains recommended for the demo. |
| D2 | Verified PASS | Empty targets, non-numeric tolerance, negative tolerance, and duplicate metrics were rejected without a crash. |
| D3 | Verified PASS | Repeated metric add/remove and verification did not leave stale result state. |
| D5 | Verified PASS | Invalid GUI input/output-node fields produced the same graceful N/A behavior as A4. |
| D7 | Verified partial integration PASS | Two netlists were queued, the asynchronous merge callback returned, the merged result loaded into the editor, then physically simulated and plotted. An injected deterministic offline merge agent was used; a live-backend merge remains unverified. |
| D8 | Verified PASS | Consecutive verification runs, followed by loading a second netlist, used the current netlist rather than stale state. |

The GUI tests above used the real widgets, simulation, callback path, and plot rendering. They were automated rather than a user-visible rehearsal. Manual responsiveness and the end-to-end local-agent flow remain unverified.

## E. Environment and submission hygiene

| # | Result | Evidence / observed outcome |
|---|---|---|
| E1 | Partially remediated | Portable runtime, AMD, and training requirements files replaced the former machine-specific environment export. A true clean-machine clone/install remains unverified. |
| E2 | Verified PASS | `AD8092` was copied into a temporary directory whose path contained spaces and passed real LTspice verification. |
| E3 | Verified PASS | A process with dotenv loading disabled and no LLM key returned the clean missing-key `LLMError`; core LTspice verification still passed. |
| E4 | Partially remediated | The remaining hardcoded OpenRouter fallback was removed, source/notebook scans found no credential literals, and the AMD notebook now requires an authenticated private session token. The historically exposed OpenRouter key still must be revoked by its owner before submission. |

## Test-discovered fix

`ADP2302` initially failed on macOS because its corpus netlist used a Windows-style `.lib` path. [app/simulation_runner.py](app/simulation_runner.py) now normalizes backslashes before locating the LTspice library. The fixed netlist completed a real 12.5-second simulation and rendered through the GUI plot path.

## Not yet verified

- **D4:** a human-visible responsiveness check during a long simulation.
- **D6:** full local-Gemma agent flow against the compatible base model on the target machine.
- **D7 live backend:** combine two circuits through an actual LLM backend rather than the deterministic test double.
- **E1:** a true fresh-clone install using the portable project requirements.
- **E4:** revoke the historically exposed OpenRouter key before publishing the repository.
- **F1–F8:** AMD MI300X/Qwen 7B generation and ROCm evidence.
- **Gemma loader / D6 prerequisite:** the release repository includes the real LoRA adapter through Git LFS, and the loader now resolves its compatible base-model identifier from adapter configuration. The full local-Gemma agent flow remains unverified against the compatible base model on the target machine.

## Relevant implementation

- [Verifier runner](../sim_harness.py), [waveform measurement](../measure_raw.py), [spec reporting](../spec_report.py), and [CLI](../report_netlist.py)
- [Template retrieval](../template_index.py), [LLM extraction](../llm_client.py), and [manual AMD candidate verification](../generate_verify.py)
- [Candidate Arena](../candidate_arena.py) for best-of-N selection and evidence export
- [GUI Verify Spec integration](../app/gui_main.py)
- [Local Gemma loader guard](../app/ft_mac.py)
