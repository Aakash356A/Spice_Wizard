"""Rank multiple LLM circuit candidates with the same real-SPICE acceptance gate.

The Candidate Arena makes best-of-N generation auditable: every candidate is
checked against the original template invariants, simulated under the same test
conditions, ranked by measured results, and optionally exported as a compact
JSON evidence bundle. Generation can happen on AMD hardware, locally, or via
any API; the acceptance decision always remains local and reproducible.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from llm_client import extract_netlist
from sim_harness import Metric, Spec, run_spice
from spec_report import MetricReport, report as build_report


@dataclass(frozen=True)
class CandidateInput:
    """Untrusted model output submitted to the verification arena."""

    label: str
    text: str
    source: str = "candidate"


@dataclass
class CandidateOutcome:
    """The complete, inspectable result of one candidate evaluation."""

    label: str
    source: str
    candidate_sha256: str
    status: str
    netlist: str
    reports: list[MetricReport] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    error: str = ""
    score: float = 0.0
    passed_metrics: int = 0
    total_metrics: int = 0
    elapsed_sec: float = 0.0
    rank: int | None = None

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


def validate_template_constraints(template_netlist: str, candidate_netlist: str) -> list[str]:
    """Return invariant changes that disqualify a candidate before simulation.

    Candidate generation is deliberately limited to component-value adaptation.
    Element identity/connectivity, subcircuit calls, library directives, and
    analyses define the trusted topology and testbench. They must remain
    equivalent after whitespace normalization. Only the final value token of a
    passive R/C/L device and the waveform/value portion of a V/I source may
    differ.
    """
    categories = {
        "subcircuit call": lambda line: line[:1].lower() == "x",
        "library directive": lambda line: line.lower().startswith((".lib", ".include")),
        "model directive": lambda line: line.lower().startswith(".model"),
        "analysis directive": lambda line: line.lower().startswith((".ac", ".dc", ".op", ".tran")),
    }

    def protected_lines(netlist: str, predicate) -> list[str]:
        return [
            " ".join(line.split())
            for raw_line in netlist.splitlines()
            if (line := raw_line.strip()) and not line.startswith("*") and predicate(line)
        ]

    violations = []
    for category, predicate in categories.items():
        if protected_lines(template_netlist, predicate) != protected_lines(candidate_netlist, predicate):
            violations.append(
                f"{category} changed; preserve the template's {category} line(s) exactly."
            )

    def topology_signature(netlist: str) -> list[tuple[str, tuple[str, ...]]]:
        """Capture all topology-bearing tokens while ignoring editable values."""
        signatures = []
        for raw_line in netlist.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("*", ".")):
                continue
            tokens = line.split()
            if not tokens:
                continue
            kind = tokens[0][:1].upper()
            if kind in {"R", "C", "L"} and len(tokens) >= 4:
                # Keep the element name and all connectivity/model tokens;
                # only the final value is permitted to adapt.
                signatures.append(("passive", tuple(tokens[:-1])))
            elif kind in {"V", "I"} and len(tokens) >= 3:
                # A source's node connectivity is structural. The remaining
                # DC/SINE/PULSE value or waveform arguments may adapt.
                signatures.append(("source", tuple(tokens[:3])))
            else:
                signatures.append(("fixed", tuple(tokens)))
        return signatures

    if topology_signature(template_netlist) != topology_signature(candidate_netlist):
        violations.append(
            "topology changed; preserve the template's element set, order, and node connectivity."
        )
    return violations


def _normalized_metric_score(metric: Metric, result: MetricReport) -> float:
    """Return a bounded quality score that is meaningful within one metric type.

    A passing candidate always outranks a failing candidate separately. This
    score then prefers an exact hit for target metrics and greater safety margin
    for one-sided metrics. It is a ranking aid, never a replacement for PASS.
    """
    measured = result.measured
    if not math.isfinite(measured):
        return 0.0

    scale = max(abs(metric.target), abs(metric.tol), 1e-12)
    if metric.direction == "meet_at_least":
        if measured < metric.target:
            return max(0.0, measured / scale)
        return min(2.0, 1.0 + (measured - metric.target) / scale)
    if metric.direction == "stay_below":
        if measured > metric.target:
            return max(0.0, 1.0 - (measured - metric.target) / scale)
        return min(2.0, 1.0 + (metric.target - measured) / scale)

    # For target metrics, the midpoint is best; a zero-tolerance target only
    # earns credit when it is exactly met.
    if metric.tol <= 0:
        return 1.0 if measured == metric.target else 0.0
    return max(0.0, 1.0 - abs(measured - metric.target) / metric.tol)


def _score_reports(spec: Spec, reports: list[MetricReport]) -> tuple[float, int]:
    by_name = {result.name: result for result in reports}
    scores: list[float] = []
    passed_metrics = 0
    for metric in spec.metrics:
        result = by_name.get(metric.name)
        if result is None:
            scores.append(0.0)
            continue
        if result.passed:
            passed_metrics += 1
        scores.append(_normalized_metric_score(metric, result))
    return (sum(scores) / len(scores) if scores else 0.0, passed_metrics)


def _outcome_sort_key(outcome: CandidateOutcome) -> tuple[int, int, float, float, str]:
    """Order passes first, then coverage, quality, runtime, and label."""
    priority = {
        "PASS": 4,
        "FAIL": 3,
        "SIMULATION_ERROR": 2,
        "REJECTED": 1,
    }.get(outcome.status, 0)
    return (-priority, -outcome.passed_metrics, -outcome.score, outcome.elapsed_sec, outcome.label.lower())


def rank_candidate_outcomes(outcomes: Iterable[CandidateOutcome]) -> list[CandidateOutcome]:
    """Return outcomes ordered for a transparent best-of-N selection."""
    ranked = sorted(outcomes, key=_outcome_sort_key)
    for index, outcome in enumerate(ranked, start=1):
        outcome.rank = index
    return ranked


def evaluate_candidate_batch(
    template_netlist: str,
    candidates: Iterable[CandidateInput],
    spec: Spec,
    *,
    in_node: str = "IN",
    out_node: str = "OUT",
    freq_hz: float = 1000.0,
    timeout_s: int = 60,
) -> list[CandidateOutcome]:
    """Evaluate candidates serially and return ranked, simulator-backed results.

    LTspice executions intentionally run serially. This keeps the workflow
    dependable on laptops and preserves clear per-candidate logs; GPU batch
    generation can still produce all candidates before they enter this gate.
    """
    outcomes: list[CandidateOutcome] = []
    requested_measurements = {metric.name for metric in spec.metrics}

    for candidate in candidates:
        started = time.monotonic()
        netlist = extract_netlist(candidate.text)
        candidate_hash = hashlib.sha256(netlist.encode("utf-8")).hexdigest()
        violations = validate_template_constraints(template_netlist, netlist)
        if violations:
            outcomes.append(
                CandidateOutcome(
                    label=candidate.label,
                    source=candidate.source,
                    candidate_sha256=candidate_hash,
                    status="REJECTED",
                    netlist=netlist,
                    diagnostics=violations,
                    error="Template invariant violation.",
                    total_metrics=len(spec.metrics),
                    elapsed_sec=time.monotonic() - started,
                )
            )
            continue

        sim = run_spice(
            netlist,
            testbench=spec.testbench,
            timeout_s=timeout_s,
            in_node=in_node,
            out_node=out_node,
            freq_hz=freq_hz,
            requested_measurements=requested_measurements,
        )
        elapsed_sec = time.monotonic() - started
        if not sim.converged:
            outcomes.append(
                CandidateOutcome(
                    label=candidate.label,
                    source=candidate.source,
                    candidate_sha256=candidate_hash,
                    status="SIMULATION_ERROR",
                    netlist=netlist,
                    diagnostics=list(sim.diagnostics),
                    error=sim.raw_log.strip() or "LTspice did not converge.",
                    total_metrics=len(spec.metrics),
                    elapsed_sec=elapsed_sec,
                )
            )
            continue

        reports = build_report(sim, spec)
        score, passed_metrics = _score_reports(spec, reports)
        passed = bool(reports) and all(report.passed for report in reports)
        outcomes.append(
            CandidateOutcome(
                label=candidate.label,
                source=candidate.source,
                candidate_sha256=candidate_hash,
                status="PASS" if passed else "FAIL",
                netlist=netlist,
                reports=reports,
                diagnostics=list(sim.diagnostics),
                score=score,
                passed_metrics=passed_metrics,
                total_metrics=len(spec.metrics),
                elapsed_sec=elapsed_sec,
            )
        )

    return rank_candidate_outcomes(outcomes)


def format_candidate_summary(outcomes: Iterable[CandidateOutcome]) -> str:
    """Create a concise human-readable candidate tournament summary."""
    lines = ["=== CANDIDATE ARENA ==="]
    for outcome in outcomes:
        rank = f"#{outcome.rank}" if outcome.rank is not None else "-"
        line = (
            f"{rank} {outcome.label}: {outcome.status} | "
            f"metrics {outcome.passed_metrics}/{outcome.total_metrics} | "
            f"score {outcome.score:.3f} | {outcome.elapsed_sec:.1f}s"
        )
        lines.append(line)
        for metric in outcome.reports:
            measured = "N/A" if not math.isfinite(metric.measured) else f"{metric.measured:.4g}"
            lines.append(
                f"    {metric.name}: target={metric.target:.4g}, measured={measured}, "
                f"{'PASS' if metric.passed else 'FAIL'}"
            )
        if outcome.error:
            lines.append(f"    detail: {outcome.error[:400]}")
        for diagnostic in outcome.diagnostics:
            lines.append(f"    diagnostic: {diagnostic}")
    return "\n".join(lines)


def _json_safe(value):
    """Convert dataclass output to strict JSON-safe data (including NaN)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def build_evidence_bundle(
    *,
    template_label: str,
    template_netlist: str,
    spec_label: str,
    spec: Spec,
    in_node: str,
    out_node: str,
    freq_hz: float,
    outcomes: Iterable[CandidateOutcome],
    include_netlists: bool = False,
) -> dict:
    """Build a portable evidence record for a candidate selection run."""
    serialized_outcomes = []
    for outcome in outcomes:
        record = {
            "rank": outcome.rank,
            "label": outcome.label,
            "source": outcome.source,
            "candidate_sha256": outcome.candidate_sha256,
            "status": outcome.status,
            "score": outcome.score,
            "passed_metrics": outcome.passed_metrics,
            "total_metrics": outcome.total_metrics,
            "elapsed_sec": outcome.elapsed_sec,
            "reports": [asdict(report) for report in outcome.reports],
            "diagnostics": outcome.diagnostics,
            "error": outcome.error,
        }
        if include_netlists:
            record["netlist"] = outcome.netlist
        serialized_outcomes.append(record)

    return _json_safe(
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tool": "Spice Wizard Candidate Arena",
            "host_platform": platform.platform(),
            "template": {
                "label": template_label,
                "sha256": hashlib.sha256(template_netlist.encode("utf-8")).hexdigest(),
            },
            "spec": {
                "label": spec_label,
                "metrics": [asdict(metric) for metric in spec.metrics],
                "in_node": in_node,
                "out_node": out_node,
                "frequency_hz": freq_hz,
            },
            "ranking_policy": (
                "PASS status, then metrics passed, then normalized target fit; "
                "the simulator's PASS/FAIL decision remains authoritative."
            ),
            "outcomes": serialized_outcomes,
        }
    )


def write_evidence_bundle(path: str | Path, **kwargs) -> Path:
    """Write a strict JSON evidence bundle and return its resolved path."""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_evidence_bundle(**kwargs)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return output_path
