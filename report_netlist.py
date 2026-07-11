#!/usr/bin/env python3
"""
Use Case 1 CLI: simulate a netlist and report measured metrics vs. target spec.

Usage:
    python report_netlist.py AD8092.net --metric gain_db=6.0:1.0 --out-node N001
    python report_netlist.py AD8092.net --metric gain_db=6.0:1.0 --metric bandwidth_hz=1e6:2e5 \\
        --direction bandwidth_hz:meet_at_least --out-node N001
"""

import argparse
import logging
import math
import sys

# app.simulation_runner configures logging.basicConfig(level=DEBUG) at import
# time; set the root level here first so its later basicConfig() call (a
# no-op once handlers exist) doesn't flood this CLI's output.
logging.basicConfig(level=logging.WARNING)

from sim_harness import Metric, Spec, run_spice
from spec_report import report, format_report

VALID_DIRECTIONS = {"hit_target", "meet_at_least", "stay_below"}


def _parse_metric(spec_str: str) -> tuple[str, float, float]:
    """Parses "name=target:tol" -> (name, target, tol)."""
    try:
        name, rest = spec_str.split("=", 1)
        target_str, tol_str = rest.split(":", 1)
        name = name.strip()
        target = float(target_str)
        tol = float(tol_str)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid --metric '{spec_str}'. Expected name=target:tol, e.g. gain_db=6.0:1.0"
        ) from e
    if not name:
        raise argparse.ArgumentTypeError("Metric name cannot be empty.")
    if not math.isfinite(target) or not math.isfinite(tol) or tol < 0:
        raise argparse.ArgumentTypeError("Metric target and tolerance must be finite numbers; tolerance must be >= 0.")
    return name, target, tol


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate a netlist and report measured metrics vs. target spec.")
    parser.add_argument("netlist", help="Path to a .net netlist file.")
    parser.add_argument("--metric", action="append", default=[], metavar="name=target:tol",
                         help="Target metric, repeatable. e.g. --metric gain_db=6.0:1.0")
    parser.add_argument("--direction", action="append", default=[], metavar="name:direction",
                         help=f"Override direction for a metric, one of {sorted(VALID_DIRECTIONS)}. "
                              f"Default: hit_target (two-sided, within +/-tol).")
    parser.add_argument("--testbench", default="", help="Optional testbench id (not currently used to modify the netlist).")
    parser.add_argument("--timeout", type=int, default=30, help="Simulation timeout in seconds.")
    parser.add_argument("--in-node", default="IN", help="Input node for gain/bandwidth measurements.")
    parser.add_argument("--out-node", default="OUT",
                         help="Output node. Real netlists often use auto-generated names like N001 -- "
                              "run `python measure_raw.py <raw file>` to see actual trace names.")
    parser.add_argument("--freq", type=float, default=1000.0, help="Frequency (Hz) for gain measurement.")
    args = parser.parse_args()

    if not args.metric:
        parser.error("At least one --metric is required.")

    directions = {}
    for d in args.direction:
        name, _, direction = d.partition(":")
        if direction not in VALID_DIRECTIONS:
            parser.error(f"--direction '{d}': direction must be one of {sorted(VALID_DIRECTIONS)}")
        directions[name] = direction

    try:
        metrics = [
            Metric(name=name, target=target, tol=tol, direction=directions.get(name, "hit_target"))
            for name, target, tol in (_parse_metric(m) for m in args.metric)
        ]
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    with open(args.netlist, "r") as f:
        netlist_text = f.read()

    sim = run_spice(
        netlist_text,
        testbench=args.testbench,
        timeout_s=args.timeout,
        in_node=args.in_node,
        out_node=args.out_node,
        freq_hz=args.freq,
        requested_measurements={metric.name for metric in metrics},
    )

    if not sim.converged:
        print(f"FAIL (does not converge / simulation error)\n{sim.raw_log}")
        return 1

    reports = report(sim, Spec(metrics=metrics, testbench=args.testbench))
    print(format_report(reports))
    if sim.diagnostics:
        print("\nMeasurement diagnostics:")
        for diagnostic in sim.diagnostics:
            print(f"- {diagnostic}")
    return 0 if all(r.passed for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
