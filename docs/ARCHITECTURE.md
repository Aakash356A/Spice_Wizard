# Architecture

## Product boundary

Spice Wizard is a **template adaptation and verification system**. It begins with a known LTspice application circuit, permits only bounded edits, and uses LTspice results as the acceptance criterion.

```mermaid
flowchart TD
    Spec[Text specification] --> Index[template_index.py]
    Index --> Template[data/templates/*.net]
    Template --> Prompt[generate_verify.py]
    Prompt --> LLM[OpenAI-compatible LLM or manual AMD Qwen]
    LLM --> Guard[Template invariant guard]
    Guard --> Simulator[sim_harness.py]
    Simulator --> LTspice[LTspice batch run]
    LTspice --> Raw[.raw waveform]
    Raw --> Metrics[measure_raw.py]
    Metrics --> Report[spec_report.py]
    Report --> Decision{Pass?}
    Decision -->|No| Prompt
    Decision -->|Yes| Pairs[data/verified_pairs.jsonl]
```

## Runtime components

| Component | Responsibility |
|---|---|
| `template_index.py` | Finds a canonical template for a requested IC. |
| `generate_verify.py` | Builds constrained prompts, verifies candidates, applies retry feedback, and writes verified records. |
| `llm_client.py` | Calls a configurable OpenAI-compatible endpoint. |
| `sim_harness.py` | Provides the common text-netlist-to-`SimResult` interface. |
| `app/simulation_runner.py` | Locates LTspice and normalizes library paths across platforms. |
| `measure_raw.py` | Measures AC gain, bandwidth, or transient gain from LTspice raw data. |
| `spec_report.py` | Produces target, measured value, margin, and PASS/FAIL/N/A report rows. |
| `app/gui_main.py` | Tkinter editor, simulation, plotting, Verify Spec, and optional agent UI. |

## Safety constraints

Before simulation, `validate_template_constraints()` rejects a candidate that changes any of these template invariants:

- subcircuit-call lines;
- `.lib` directives;
- `.ac`, `.dc`, `.op`, or `.tran` analysis directives.

The generator may change component values, not the known-good topology or testbench structure.

## Backends

### Default local verification

The verifier runs LTspice locally. It does not require an LLM or a network connection.

### Generic LLM API

`llm_client.py` uses `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`. OpenRouter is the default compatibility path.

### AMD MI300X manual handoff

The notebook runs Qwen on AMD hardware. The generated text is brought back to the Mac and passed to `generate_verify.py --candidate`, ensuring that the local simulator remains the final authority.

## Optional local Gemma adapter

The bundled LoRA adapter is not required for the verifier. It is an optional local-specialist experiment and requires a compatible base model plus the `SPICE_WIZARD_ENABLE_LOCAL_AGENT=1` opt-in when launching the GUI.
