#!/usr/bin/env python3
"""
Use Case 2: single-part application-circuit generator with a
simulate-verify-retry loop (see USE_CASES_IMPLEMENTATION.md).

Flow: retrieve a known-working template netlist for the named part
(template_index) -> ask the LLM to adapt ONLY component values to the new
spec (llm_client) -> simulate for real and score against the target
metrics (sim_harness + spec_report, i.e. Use Case 1) -> on failure, feed
the measured numbers back and retry. Every PASSING (spec, netlist,
report) triple is appended to data/verified_pairs.jsonl -- a simulation-
verified dataset collected for free from normal use.

Usage:
    python generate_verify.py AD8092 \\
        --spec "non-inverting amplifier with gain of 5 V/V (~14 dB), +/-5V supply" \\
        --metric gain_db=13.98:1.0
"""

import argparse
import datetime
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

logging.basicConfig(level=logging.WARNING)  # quiet simulation_runner's DEBUG chatter

from candidate_arena import (
    CandidateInput,
    evaluate_candidate_batch,
    format_candidate_summary,
    validate_template_constraints,
    write_evidence_bundle,
)
from sim_harness import Metric, Spec, run_spice
from spec_report import report, format_report
from template_index import find_template
from llm_client import LLMError, call_llm, extract_netlist

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
VERIFIED_PAIRS_PATH = os.path.join(REPO_ROOT, "data", "verified_pairs.jsonl")

SYSTEM_PROMPT = "You are an expert analog applications engineer who writes LTspice netlists."


def build_adapt_prompt(template_netlist: str, ic_name: str, spec_text: str) -> str:
    return f"""Below is a KNOWN-WORKING LTspice netlist template for the {ic_name}.

Modify ONLY the passive component values (resistors, capacitors, source
amplitudes/frequencies) needed to meet the NEW SPEC below.
Do NOT change the {ic_name} subcircuit call line's pin order or node names.
Do NOT change the .lib reference or the analysis (.tran/.ac) directives.
Do NOT rename any nodes.

Output the complete modified netlist and nothing else.

TEMPLATE:
{template_netlist}

NEW SPEC:
{spec_text}
"""

def _constraint_feedback(violations: list[str]) -> str:
    return (
        "\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED BEFORE SIMULATION:\n- "
        + "\n- ".join(violations)
        + "\nOnly change permitted component values and output the complete netlist again."
    )

def generate_and_verify(
    ic_name: str,
    spec_text: str,
    spec: Spec,
    max_retries: int = 3,
    in_node: str = "IN",
    out_node: str = "OUT",
    freq_hz: float = 1000.0,
    timeout_s: int = 60,
    verbose: bool = True,
):
    """Returns (netlist, reports, passed). `reports` is [] if no attempt simulated."""
    template_path = find_template(ic_name)
    if template_path is None:
        raise FileNotFoundError(f"No template netlist found for '{ic_name}' in the corpus.")
    template = open(template_path, encoding="utf-8", errors="ignore").read()
    if verbose:
        print(f"Template: {template_path}")

    prompt = build_adapt_prompt(template, ic_name, spec_text)
    netlist, rep = "", []

    for attempt in range(1, max_retries + 1):
        if verbose:
            print(f"\n--- Attempt {attempt}/{max_retries}: calling LLM...")
        netlist = extract_netlist(call_llm(prompt, system_prompt=SYSTEM_PROMPT))

        violations = validate_template_constraints(template, netlist)
        if violations:
            if verbose:
                print(f"Attempt {attempt}: REJECTED TEMPLATE DRIFT\n- " + "\n- ".join(violations))
            prompt += _constraint_feedback(violations)
            continue

        if verbose:
            print("Simulating with LTspice...")
        sim = run_spice(
            netlist,
            testbench=spec.testbench,
            timeout_s=timeout_s,
            in_node=in_node,
            out_node=out_node,
            freq_hz=freq_hz,
            requested_measurements={metric.name for metric in spec.metrics},
        )

        if not sim.converged:
            if verbose:
                print(f"Attempt {attempt}: DID NOT SIMULATE\n{sim.raw_log[:500]}")
            prompt += ("\n\nYOUR PREVIOUS ATTEMPT FAILED TO SIMULATE:\n"
                       f"{sim.raw_log[:1000]}\nFix the netlist and output it in full again.")
            continue

        rep = report(sim, spec)
        if verbose:
            print(format_report(rep))

        if all(r.passed for r in rep):
            log_verified_pair(ic_name, spec_text, netlist, rep, attempt)
            if verbose:
                print(f"\nPASS on attempt {attempt} -- logged to {VERIFIED_PAIRS_PATH}")
            return netlist, rep, True

        prompt += _failure_feedback(rep)

    if verbose:
        print(f"\nFAILED after {max_retries} attempts (last attempt returned for inspection).")
    return netlist, rep, False


def verify_candidate(
    ic_name: str,
    spec_text: str,
    candidate_text: str,
    spec: Spec,
    in_node: str = "IN",
    out_node: str = "OUT",
    freq_hz: float = 1000.0,
    timeout_s: int = 60,
    attempt: int = 1,
    source: str = "manual_candidate",
    verbose: bool = True,
):
    """Verify a netlist obtained outside the API loop (for example, copied
    manually from Qwen running on the AMD MI300X).

    The candidate can be a raw netlist or the full chat response; extraction
    removes code fences and surrounding prose. A passing candidate is logged
    to data/verified_pairs.jsonl exactly like an API-generated candidate.
    """
    netlist = extract_netlist(candidate_text)
    template_path = find_template(ic_name)
    if template_path is None:
        raise FileNotFoundError(f"No template netlist found for '{ic_name}' in the corpus.")
    template = open(template_path, encoding="utf-8", errors="ignore").read()
    violations = validate_template_constraints(template, netlist)
    if violations:
        if verbose:
            print("Candidate REJECTED TEMPLATE DRIFT\n- " + "\n- ".join(violations))
        return netlist, [], False

    sim = run_spice(
        netlist,
        testbench=spec.testbench,
        timeout_s=timeout_s,
        in_node=in_node,
        out_node=out_node,
        freq_hz=freq_hz,
        requested_measurements={metric.name for metric in spec.metrics},
    )
    if not sim.converged:
        if verbose:
            print(f"Candidate DID NOT SIMULATE\n{sim.raw_log[:1000]}")
        return netlist, [], False

    rep = report(sim, spec)
    if verbose:
        print(format_report(rep))
        for diagnostic in sim.diagnostics:
            print(f"Diagnostic: {diagnostic}")

    passed = all(r.passed for r in rep)
    if passed:
        log_verified_pair(ic_name, spec_text, netlist, rep, attempt, source=source)
        if verbose:
            print(f"\nPASS -- logged to {VERIFIED_PAIRS_PATH}")
    elif verbose:
        print("\n=== RETRY FEEDBACK FOR THE GENERATOR ===")
        print(_failure_feedback(rep))
    return netlist, rep, passed


def _parse_metric(spec_str: str) -> tuple[str, float, float]:
    """Parses "name=target:tol" -> (name, target, tol). Same format as report_netlist.py."""
    try:
        name, rest = spec_str.split("=", 1)
        target_str, tol_str = rest.split(":", 1)
        return name.strip(), float(target_str), float(tol_str)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid --metric '{spec_str}'. Expected name=target:tol, e.g. gain_db=14.0:1.0"
        ) from e


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a spec-adapted netlist for a known part and verify it in LTspice.")
    parser.add_argument("ic", help="Part name with a template in the corpus, e.g. AD8092")
    parser.add_argument("--spec", required=True,
                        help='Natural-language spec, e.g. "non-inverting gain of 5 V/V, +/-5V supply"')
    parser.add_argument("--metric", action="append", default=[], metavar="name=target:tol",
                        help="Target metric, repeatable. e.g. --metric gain_db=13.98:1.0")
    parser.add_argument("--retries", type=int, default=3, help="Max generate-verify attempts.")
    parser.add_argument("--in-node", default="IN")
    parser.add_argument("--out-node", default="OUT")
    parser.add_argument("--freq", type=float, default=1000.0, help="Frequency (Hz) for AC gain measurement.")
    parser.add_argument("--timeout", type=int, default=60, help="Per-simulation timeout (s).")
    parser.add_argument("--save", default="", help="Also write the final or best-ranked netlist to this path.")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--candidate", metavar="PATH",
        help="Verify a manually generated Qwen/LLM response stored at PATH; no API call is made.",
    )
    source_group.add_argument(
        "--candidates", nargs="+", metavar="PATH",
        help=(
            "Verify and rank multiple model responses. Each candidate is checked against the "
            "same template and LTspice specification; no API call is made."
        ),
    )
    source_group.add_argument(
        "--prompt-only", action="store_true",
        help="Print the template-adaptation prompt for manual submission to Qwen; no API call is made.",
    )
    parser.add_argument(
        "--source", default="manual_candidate",
        help="Provenance label written to data/verified_pairs.jsonl for manually supplied candidates.",
    )
    parser.add_argument(
        "--report", default="", metavar="PATH",
        help="Write a JSON evidence bundle for --candidates to PATH.",
    )
    args = parser.parse_args()

    if not args.prompt_only and not args.metric:
        parser.error("At least one --metric is required (that's the whole point of verifying).")

    if args.prompt_only:
        template_path = find_template(args.ic)
        if template_path is None:
            parser.error(f"No template netlist found for '{args.ic}' in the corpus.")
        template = open(template_path, encoding="utf-8", errors="ignore").read()
        print(build_adapt_prompt(template, args.ic, args.spec))
        return 0

    if args.report and not args.candidates:
        parser.error("--report is available only with --candidates.")

    metrics = [Metric(name=n, target=t, tol=tol, direction="hit_target")
               for n, t, tol in (_parse_metric(m) for m in args.metric)]

    try:
        if args.candidates:
            template_path = find_template(args.ic)
            if template_path is None:
                raise FileNotFoundError(f"No template netlist found for '{args.ic}' in the corpus.")
            template = open(template_path, encoding="utf-8", errors="ignore").read()
            candidates = [
                CandidateInput(
                    label=Path(path).stem,
                    text=Path(path).read_text(encoding="utf-8", errors="ignore"),
                    source=args.source,
                )
                for path in args.candidates
            ]
            outcomes = evaluate_candidate_batch(
                template,
                candidates,
                Spec(metrics=metrics, testbench=""),
                in_node=args.in_node,
                out_node=args.out_node,
                freq_hz=args.freq,
                timeout_s=args.timeout,
            )
            print(format_candidate_summary(outcomes))
            for outcome in outcomes:
                if outcome.passed:
                    log_verified_pair(
                        args.ic,
                        args.spec,
                        outcome.netlist,
                        outcome.reports,
                        attempt=outcome.rank or 1,
                        source=outcome.source,
                    )
            if args.report:
                evidence_path = write_evidence_bundle(
                    args.report,
                    template_label=str(template_path),
                    template_netlist=template,
                    spec_label=args.spec,
                    spec=Spec(metrics=metrics, testbench=""),
                    in_node=args.in_node,
                    out_node=args.out_node,
                    freq_hz=args.freq,
                    outcomes=outcomes,
                )
                print(f"Evidence bundle written to {evidence_path}")
            best = outcomes[0] if outcomes else None
            netlist = best.netlist if best else ""
            rep = best.reports if best else []
            passed = any(outcome.passed for outcome in outcomes)
        elif args.candidate:
            candidate_text = open(args.candidate, encoding="utf-8", errors="ignore").read()
            netlist, rep, passed = verify_candidate(
                args.ic, args.spec, candidate_text, Spec(metrics=metrics, testbench=""),
                in_node=args.in_node, out_node=args.out_node, freq_hz=args.freq,
                timeout_s=args.timeout, source=args.source,
            )
        else:
            netlist, rep, passed = generate_and_verify(
                args.ic, args.spec, Spec(metrics=metrics, testbench=""),
                max_retries=args.retries, in_node=args.in_node, out_node=args.out_node,
                freq_hz=args.freq, timeout_s=args.timeout,
            )
    except (LLMError, FileNotFoundError, OSError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.save and netlist:
        with open(args.save, "w") as f:
            f.write(netlist)
        print(f"Netlist written to {args.save}")

    print("\n=== FINAL NETLIST ===")
    print(netlist)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
