"""Fast, simulator-free checks for Candidate Arena policy and evidence output."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from candidate_arena import (
    CandidateOutcome,
    build_evidence_bundle,
    rank_candidate_outcomes,
    validate_template_constraints,
)
from sim_harness import Metric, Spec
from spec_report import MetricReport


class CandidateArenaTests(unittest.TestCase):
    def setUp(self):
        self.template = """* Trusted template
V1 IN 0 AC 1
R1 OUT IN 1k
XU1 IN 0 OUT VCC VEE AD8092
.lib ADI.lib
.ac dec 20 1k 1Meg
.end
"""

    def test_allows_value_only_adaptation(self):
        candidate = self.template.replace("R1 OUT IN 1k", "R1 OUT IN 4k")
        self.assertEqual(validate_template_constraints(self.template, candidate), [])

    def test_rejects_protected_analysis_change(self):
        candidate = self.template.replace(".ac dec 20 1k 1Meg", ".tran 0 10m")
        violations = validate_template_constraints(self.template, candidate)
        self.assertEqual(len(violations), 1)
        self.assertIn("analysis directive", violations[0])

    def test_rejects_topology_change(self):
        candidate = self.template.replace("R1 OUT IN 1k", "R1 OUT N001 1k")
        violations = validate_template_constraints(self.template, candidate)
        self.assertEqual(len(violations), 1)
        self.assertIn("topology changed", violations[0])

    def test_ranks_passes_before_failures_and_by_quality(self):
        outcomes = rank_candidate_outcomes(
            [
                CandidateOutcome("failed", "test", "a", "FAIL", "", score=1.0, passed_metrics=0, total_metrics=1),
                CandidateOutcome("pass-low", "test", "b", "PASS", "", score=0.5, passed_metrics=1, total_metrics=1),
                CandidateOutcome("pass-high", "test", "c", "PASS", "", score=0.9, passed_metrics=1, total_metrics=1),
            ]
        )
        self.assertEqual([outcome.label for outcome in outcomes], ["pass-high", "pass-low", "failed"])
        self.assertEqual([outcome.rank for outcome in outcomes], [1, 2, 3])

    def test_evidence_bundle_serializes_non_finite_measurements_as_null(self):
        outcome = CandidateOutcome(
            "candidate-a",
            "amd_mi300x_manual",
            "abc123",
            "FAIL",
            "* netlist\n.end\n",
            reports=[MetricReport("gain_db", 6.0, math.nan, False, math.nan)],
            total_metrics=1,
        )
        bundle = build_evidence_bundle(
            template_label="AD8092.net",
            template_netlist=self.template,
            spec_label="gain target",
            spec=Spec(metrics=[Metric("gain_db", 6.0, 1.0, "hit_target")], testbench=""),
            in_node="IN",
            out_node="OUT",
            freq_hz=1000.0,
            outcomes=[outcome],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            path.write_text(json.dumps(bundle, allow_nan=False), encoding="utf-8")
            decoded = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsNone(decoded["outcomes"][0]["reports"][0]["measured"])


if __name__ == "__main__":
    unittest.main()
