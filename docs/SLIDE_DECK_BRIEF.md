# Slide Deck Briefing — AMD Developer Hackathon (ACT II), Track 3

> **How to use this file:** paste the PROMPT (bottom of this file) into Claude,
> and attach/paste this entire briefing document with it. Everything Claude
> needs — story, numbers, architecture, demo moments, judging criteria, slide
> structure — is in here.
>
> **Before you generate:** search this file for `[PLANNED]` and update anything
> you've finished since 2026-07-07 (especially the AMD/vLLM port). Never let a
> slide claim something still marked planned as if it were done.

---

## 1. Project identity

- **Working name:** Spice Wizard (repo: Spice_Wizard) — feel free to let Claude
  propose a sharper product name, e.g. "VeriSPICE", "ProvenCircuit", "SpecLoop".
- **One-liner (the thesis of the whole deck):**
  > *"LLMs hallucinate circuits that look right. Ours has to prove it — every
  > netlist is simulation-verified in real SPICE before a human ever sees it."*
- **Category:** AI agent for analog/EDA circuit design with
  simulator-in-the-loop verification.
- **Track:** Track 3 — Unicorn (Open Innovation). No fixed task; judged on
  innovation, technical impressiveness, practical usefulness, and **mandatory
  demonstrated AMD compute usage**.

## 2. The problem (slide-ready framing)

1. Analog circuit design is a bottleneck: digital design has synthesis tools;
   analog still depends on scarce senior engineers reading datasheets.
2. LLMs *seem* like the answer — they happily write SPICE netlists — but their
   output is **plausible text, not verified engineering**. A netlist can be
   syntactically perfect, use the right part, follow the right formula, and
   still describe a circuit that cannot physically work.
3. Nobody ships a circuit on vibes. Without verification, LLM circuit
   generation is a demo, not a tool.
4. Killer proof point (real, reproduced in this project — see §5): GPT-5 made
   a textbook-correct change to an amplifier and was still wrong, because the
   physical part can't slew fast enough. Only a simulator catches that.

## 3. The solution — the verified generation loop

```
User spec (natural language + numeric targets)
        │
        ▼
┌─ RETRIEVE ─────────────────────────────────────────┐
│ Template index over 3,416 real ADI application     │
│ netlists (~4k .net files) → known-working circuit  │
│ for the requested part                             │
└────────────────────────────────────────────────────┘
        │
        ▼
┌─ ADAPT (LLM on AMD) ───────────────────────────────┐
│ Open-weight LLM served with vLLM on AMD MI300X     │
│ modifies ONLY component values to meet the spec;   │
│ part pinout, .lib, topology are locked             │
└────────────────────────────────────────────────────┘
        │
        ▼
┌─ VERIFY (the differentiator) ──────────────────────┐
│ Real LTspice batch simulation → parse .raw         │
│ waveforms → measure actual gain / bandwidth /      │
│ settling → PASS/FAIL vs numeric targets w/ margin  │
└────────────────────────────────────────────────────┘
        │
   FAIL ├────► feed measured numbers back to the LLM, retry (max N)
        │
   PASS ▼
Verified netlist + measured report → user
        │
        ▼
verified_pairs.jsonl  ← every pass is logged: a simulation-verified
                        dataset that grows for free with normal use
                        → fine-tune a small model on it (RSFT flywheel)
```

Key phrase for slides: **"The simulator is ground truth. The simulator is the
labeler. The simulator is the judge."**

## 4. What is actually built and working (real measured results)

| Piece | Status | Real result |
|---|---|---|
| UC1 — Verifier: netlist → LTspice sim → measured report vs spec | ✅ WORKING | AD8092 amp: target 6.0 dB, **measured 6.014 dB, PASS (+0.2% margin)** |
| `.raw` measurement library (AC gain, −3 dB bandwidth, phase, transient gain, settling time) | ✅ WORKING | `measure_raw.py` |
| Template index over the netlist corpus | ✅ WORKING | **3,416 distinct parts** indexed from ~4k ADI application netlists |
| UC2 — Generate→Verify→Retry loop | ✅ WORKING | AD8092 adapted gain 2→5: LLM changed R2 1K→4K, **measured 13.952 dB vs 13.98 dB target, PASS on attempt 1**, logged to `verified_pairs.jsonl` |
| Verified-pairs data flywheel log | ✅ WORKING | Every pass appends `(spec, netlist, measured report)` |
| Desktop GUI (netlist editor, simulation, waveform plots, verify-spec tab) | ✅ WORKING | Tkinter app, `app/gui_main.py` |
| LLM client with swappable backend (env vars only) | ✅ WORKING | OpenRouter today → vLLM/MI300X with zero code changes |
| LLM inference served on AMD MI300X via vLLM/ROCm | 🔶 [PLANNED — UPDATE BEFORE GENERATING] | Open model (e.g. Qwen2.5-Coder) on AMD Developer Cloud |
| Best-of-N parallel candidate generation on MI300X | 🔶 [PLANNED] | N candidates in parallel → simulate all → surface best pass |
| RSFT: fine-tune small model on verified_pairs on AMD | 🔶 [PLANNED] | The self-improvement chapter |
| Mini-benchmark (15–25 spec→verify cases, vs raw-LLM baseline) | 🔶 [PLANNED] | The metrics slide |

## 5. THE demo-gold story (make this its own slide — it's the emotional peak)

Real, reproduced result from 2026-07-07:

- Ask for: AD8092 non-inverting amp, gain 5 V/V (13.98 dB), with the
  template's original 1 V @ 10 MHz test signal.
- **GPT-5 does the textbook-correct thing:** changes R2 from 1K → 4K
  (gain = 1 + 4k/1k = 5). Any textbook, any professor, any code reviewer
  says: correct.
- **Simulation measures 10.3 dB (~3.3 V/V), not 13.98 dB. FAIL.**
- Why: gain 5 × 1 V at 10 MHz requires ~314 V/µs of output slew rate. The
  real AD8092 slews at ~170 V/µs. The circuit is **mathematically right and
  physically impossible.** GPT-5 never diagnosed it across 3 retries.
- With a physically sane test condition (100 mV @ 1 MHz), the same loop
  passes first try at 13.952 dB.

Slide framing: *"The LLM's answer looked perfect. Physics said no. Our system
is the only one in the room that asked physics."*

## 6. The AMD compute story (mandatory — judges disqualify without it)

AMD usage must be **load-bearing in the architecture**, not a checkbox:

1. **Inference on MI300X:** the generation LLM is an open-weight model served
   with **vLLM on ROCm** on **AMD Developer Cloud MI300X (192 GB HBM3)**.
   Closed frontier APIs (GPT-5/Claude) can't run on AMD; we don't need them —
   see point 2.
2. **The verification loop is the equalizer:** a mid-tier open model + a
   simulator oracle + retry **outperforms an unverified frontier model**,
   because wrong answers are caught and corrected. "We replaced model scale
   with ground truth."
3. **Best-of-N parallelism [PLANNED]:** MI300X's 192 GB lets us batch-generate
   N candidate netlists in parallel and simulate them all — throughput turns
   sampling into engineering. Metric to show: pass@1 vs pass@N-verified.
4. **Training on AMD [PLANNED]:** the small model is fine-tuned on
   `verified_pairs.jsonl` **on the same AMD hardware** — inference AND
   training on AMD.

## 7. The data flywheel (second-biggest differentiator)

- Every verified pass logs `(spec, netlist, measured_report)` to
  `verified_pairs.jsonl`.
- This is a **simulation-verified training dataset that grows for free from
  normal use** — no human labeling, no scraping. The simulator is the labeler.
- Fine-tune a small local model (Gemma-class) on it → the system **improves
  itself with use** (RSFT loop). Small model = cheaper tokens = more best-of-N
  candidates per dollar → more verified passes → more data. The loop compounds.

## 8. Practical usefulness (judges' third criterion)

- Built on **~4,000 real Analog Devices application netlists** — actual
  circuits engineers use, not toy examples.
- Target user: hardware engineers adapting a known part to their spec ("I know
  I want the AD8092, I need gain 5 into 2k") — today that's datasheet reading
  and manual iteration; here it's one sentence + a verified answer with
  measured numbers and margins.
- Output isn't a chat answer — it's a netlist **plus a measured report**
  (target vs measured vs margin), which is what an engineering review needs.
- GUI already exists: editor, one-click simulate, waveform plots, verify tab.

## 9. Honest limitations (have one backup slide; judges respect candor)

- Adaptation scope is component values on known-good templates — not novel
  topology synthesis from scratch (that's the roadmap, UC3: multi-part signal
  chains, scaffolding already in repo).
- Measurement library covers gain/bandwidth/phase/settling — not yet noise,
  THD, PSRR, stability margins.
- LTspice runs on CPU; AMD GPUs do all LLM work. (LTspice is the ADI-blessed
  simulator for these ADI models — that's a feature, not a shortcut.)
- Verification is only as good as the vendor SPICE models (industry-standard
  caveat).

## 10. Numbers cheat sheet (use these exact figures)

| Number | Meaning |
|---|---|
| 3,416 | distinct ICs indexed as retrievable templates |
| ~4,000 | real ADI application netlists in the corpus |
| 6.014 dB vs 6.0 target (+0.2%) | UC1 verifier accuracy on stock AD8092 |
| 13.952 dB vs 13.98 target (−0.2%) | UC2 generated-circuit pass, attempt 1 |
| 10.3 dB vs 13.98 expected | GPT-5's "textbook-correct" slew-rate failure |
| ~314 V/µs needed vs ~170 V/µs actual | the physics GPT-5 missed |
| 192 GB HBM3 | MI300X memory enabling large-model + best-of-N serving |
| 7 | generic sub-circuit blocks in the UC3 composition library (roadmap) |

## 11. Suggested deck structure (10–12 slides)

1. **Title** — product name, one-liner, team, "AMD Hackathon Track 3".
2. **Problem** — analog design bottleneck; LLMs generate plausible, unverified
   circuits; nobody ships circuits on vibes.
3. **The slew-rate story** — GPT-5's perfect-looking failure (§5). Put it
   early; it earns attention for everything after.
4. **Solution** — the loop diagram (§3): Retrieve → Adapt → Verify → Retry.
   One visual, minimal text.
5. **Live results** — the real numbers table (§4/§10): verified passes with
   measured margins. Screenshot of the CLI/GUI report if possible.
6. **AMD architecture** — MI300X + vLLM/ROCm serving diagram; why open model
   + verification beats closed frontier model unverified (§6).
7. **Best-of-N on MI300X** — sampling is cheap, verification is truth;
   pass@1 vs pass@N chart. *(drop if still unbuilt at deadline)*
8. **Data flywheel** — verified_pairs.jsonl → RSFT on AMD → better small
   model → cheaper candidates → more passes. Circular diagram.
9. **Demo** — screenshots/frames of the GUI: spec in → netlist + waveform +
   PASS report out. Point to the demo video.
10. **Roadmap** — UC3 multi-part signal chains (sensor→amp→filter→ADC),
    broader measurements, semantic retrieval.
11. **Why this wins** — recap thesis: ground truth beats model scale;
    self-improving; built on 4k real circuits; AMD compute is load-bearing.
12. **(Backup) Limitations** — §9, shown only if asked.

## 12. Design direction for the deck

- **Tone:** confident engineering, not startup fluff. Every claim has a
  measured number next to it.
- **Visual identity:** dark background works well (oscilloscope/EDA
  aesthetic) — think waveform green/amber on near-black, or clean white with
  one accent. AMD brand red (#ED1C24) as accent for AMD-related slides.
- **Recurring motifs:** PASS/FAIL stamps with margins; waveform traces;
  schematic fragments; the loop diagram repeated small as a progress marker.
- **The netlist is a prop:** show real netlist text (monospace) with the one
  changed line highlighted (R2 1K → 4K) — it makes "the LLM edits circuits"
  concrete in one glance.
- Keep per-slide word count low; this briefing is the speaker notes, not the
  slide text.

## 13. Judging criteria mapping (say these back to the judges implicitly)

| Track 3 criterion | Our answer |
|---|---|
| Innovative | Simulator-in-the-loop LLM generation; sim as labeler/judge; self-improving flywheel |
| Technically impressive | Real EDA toolchain integration (LTspice batch + .raw parsing + measurement math), retry loop, ROCm/vLLM serving |
| Practically useful | 4k real ADI circuits, real engineer workflow, measured reports with margins, working GUI |
| AMD compute usage (mandatory) | Inference (and training) on MI300X via ROCm/vLLM; best-of-N leverages MI300X throughput |

Submission checklist: GitHub repo URL (public, README with visible AMD/ROCm
section — automated pre-screen reads it), demo video, slide deck PDF, optional
hosted live demo URL (ideally running on the AMD box — double-counts).

---

---

# THE PROMPT (paste this into Claude, attach this briefing file)

```
You are designing a slide deck for the AMD Developer Hackathon (ACT II),
Track 3 "Unicorn (Open Innovation)". The full project briefing is in the
attached SLIDE_DECK_BRIEF.md — treat it as the single source of truth for
all facts, numbers, and claims. Do not invent results, metrics, or
capabilities that are not in the briefing. Items marked [PLANNED] must be
presented as roadmap/in-progress, never as completed work.

PROJECT ESSENCE: An AI agent that generates analog circuit netlists and
refuses to show them until they are PROVEN by real SPICE simulation —
"LLMs hallucinate circuits that look right; ours has to prove it." LLM
inference runs on AMD MI300X via vLLM/ROCm (mandatory judging criterion —
make AMD usage visually and verbally prominent, it must feel load-bearing,
not bolted on).

BUILD THE DECK:
- 11 slides + 1 backup limitations slide, following the structure in §11 of
  the briefing. Merge or drop slide 7 (best-of-N) if the briefing still
  marks it [PLANNED] and I haven't updated it.
- Slide 3 is the emotional peak: the slew-rate story from §5 — GPT-5 makes
  the textbook-correct resistor change, simulation measures 10.3 dB instead
  of 13.98 dB because the real part can't slew fast enough. Build this as a
  before/after reveal. Use the exact numbers.
- Use the exact figures from the §10 numbers cheat sheet wherever a
  quantitative claim appears. Every claim slide should have a number on it.
- Design direction per §12: dark EDA/oscilloscope aesthetic (waveform
  green/amber on near-black), AMD red (#ED1C24) accent on AMD slides,
  monospace netlist snippets with the changed line (R2 1K → 4K)
  highlighted, PASS/FAIL stamps with margins as a recurring motif, one big
  loop diagram (Retrieve → Adapt on AMD → Verify in SPICE → Retry/Pass)
  that reappears as a small progress marker on later slides.
- Keep slide text minimal (headlines + numbers + one visual each); put
  explanation into speaker notes. Write full speaker notes for every slide.
- Audience: technical judges who are NOT analog-electronics experts. Explain
  EDA terms in one clause the first time they appear (e.g. "netlist — the
  text description of a circuit that simulators run").
- The through-line to reinforce on every slide: ground truth from a real
  simulator beats raw model scale, and AMD hardware is what makes the
  loop's generation side fast and self-improving.

DELIVERABLE: the complete deck with, for each slide: title, on-slide text,
visual description/layout, and speaker notes. Then a closing checklist of
any facts I should verify or screenshots I need to capture (GUI report,
CLI pass output, rocm-smi, vLLM logs) before exporting to PDF.
```
