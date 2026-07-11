"""
Turns a completed SimResult (from sim_harness.run_spice) into a structured
pass/fail report against a Spec's target metrics.

Use Case 1, build step 3 (see USE_CASES_IMPLEMENTATION.md).
"""

import math
from dataclasses import dataclass

from sim_harness import SimResult, Spec

try:
    # Reuse the real scoring function if/when grpo_reward.py exists -- it
    # doesn't yet in this workspace, so the fallback below is what actually
    # runs today.
    from grpo_reward import saturating_score
except ImportError:
    def saturating_score(measured: float, target: float, tol: float, direction: str = "hit_target") -> float:
        """Fallback matching sim_harness.Metric's documented semantics:
          - "hit_target":    pass if within +/- tol of target
          - "meet_at_least": pass if measured >= target (tol ignored, per Metric's docstring)
          - "stay_below":    pass if measured <= target (tol ignored, per Metric's docstring)
        Returns 1.0 (pass) or 0.0 (fail). Swap this out if grpo_reward.py is
        added later with a smoother/weighted shape.
        """
        if direction == "meet_at_least":
            return 1.0 if measured >= target else 0.0
        if direction == "stay_below":
            return 1.0 if measured <= target else 0.0
        return 1.0 if abs(measured - target) <= tol else 0.0


@dataclass
class MetricReport:
    name: str
    target: float
    measured: float
    passed: bool
    margin_pct: float


def report(sim: SimResult, spec: Spec) -> list[MetricReport]:
    out: list[MetricReport] = []
    for m in spec.metrics:
        measured = sim.measurements.get(m.name)
        if measured is None or (isinstance(measured, float) and math.isnan(measured)):
            out.append(MetricReport(m.name, m.target, float("nan"), False, float("nan")))
            continue

        passed = saturating_score(measured, m.target, m.tol, m.direction) >= 0.999
        margin = (measured - m.target) / m.target * 100 if m.target else float("nan")
        out.append(MetricReport(m.name, m.target, measured, passed, margin))
    return out


def format_report(reports: list[MetricReport]) -> str:
    lines = []
    for r in reports:
        status = "PASS" if r.passed else "FAIL"
        measured_str = "N/A" if math.isnan(r.measured) else f"{r.measured:.3f}"
        margin_str = "N/A" if math.isnan(r.margin_pct) else f"{r.margin_pct:+.1f}%"
        lines.append(
            f"{r.name:<12} target={r.target:<8} measured={measured_str:<8} "
            f"{status:<4} (margin {margin_str})"
        )
    return "\n".join(lines)
