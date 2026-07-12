"""
sim_harness.py
==============
The simulator-in-the-loop harness -- the SPINE of the whole project.

The SAME harness is used in three places:
  - RSFT data filter (keep only netlists that simulate + pass checks)
  - RL reward     (grpo_reward.py calls run_spice / parses_as_spice / region checks)
  - Evaluation    (Pass@k against held-out specs)

This file defines the INTERFACE and a parse-level implementation. The simulate()
internals are STUBBED -- wire them to ngspice (via PySpice) or your simulator of
choice. Get this interface right early; everything plugs into it.

Suggested backends:
    pip install PySpice            # Python wrapper over ngspice
    # or shell out to `ngspice -b netlist.cir`
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional


# ----------------------------------------------------------------------------
# Spec description (what the reward compares against)
# ----------------------------------------------------------------------------
@dataclass
class Metric:
    name: str                 # "dc_gain_db", "ugf_hz", "phase_margin_deg", "power_w", ...
    target: float
    tol: float                # width for "hit_target" metrics; ignored otherwise
    direction: str            # "meet_at_least" | "stay_below" | "hit_target"
    weight: float = 1.0


@dataclass
class Spec:
    metrics: list[Metric]
    testbench: str            # analyses to run, or a testbench template id
    max_devices: int = 999
    # randomize these across RL rollouts to prevent testbench overfitting:
    corners: list[str] = field(default_factory=lambda: ["tt"])
    cm_voltage: float = 0.9
    c_load: float = 1e-12


# ----------------------------------------------------------------------------
# Simulation result (what the reward consumes)
# ----------------------------------------------------------------------------
@dataclass
class SimResult:
    converged: bool
    measurements: dict           # {"dc_gain_db": 41.2, "ugf_hz": 5.3e7, ...}
    device_regions: dict         # {"M1": "saturation", "M5": "triode", ...}
    raw_log: str = ""

    def measure(self, name: str) -> float:
        """Return a measured metric; raise if the analysis didn't produce it."""
        if name not in self.measurements:
            raise KeyError(f"metric '{name}' not measured (check testbench/analyses)")
        return self.measurements[name]


# ----------------------------------------------------------------------------
# Cheap syntax gate (real, not stubbed) -- runs before any simulation
# ----------------------------------------------------------------------------
_ELEMENT_RE = re.compile(r"^\s*[MmRrCcLlDdQqVvIiXxEeFfGgHhKkSsTt]\w*\s+\S+", re.MULTILINE)


def parses_as_spice(netlist: str) -> bool:
    """
    Fast structural check: has at least a few elements, a .end (or .ends), and no
    obviously malformed element lines. This is a pre-filter, not full validation --
    full validation is 'does ngspice accept it', which run_spice() establishes.
    """
    if not netlist or ".end" not in netlist.lower():
        return False
    elements = _ELEMENT_RE.findall(netlist)
    if len(elements) < 2:
        return False
    # No element line should have fewer than 3 whitespace tokens (name + 2 nodes min).
    for line in netlist.splitlines():
        s = line.strip()
        if not s or s.startswith("*") or s.startswith("."):
            continue
        if re.match(r"^[MmRrCcLlDdQqVvIiXx]", s) and len(s.split()) < 3:
            return False
    return True


def device_count(netlist: str) -> int:
    return len(_ELEMENT_RE.findall(netlist))


def count_floating_or_shorted_nodes(netlist: str) -> int:
    """
    STUB heuristic: count nodes that appear on only one element pin (dangling).
    Replace with a proper connectivity check. Returns a non-negative integer.
    """
    pin_counts: dict[str, int] = {}
    for line in netlist.splitlines():
        s = line.strip()
        if not s or s.startswith("*") or s.startswith("."):
            continue
        toks = s.split()
        if len(toks) < 3:
            continue
        # crude: treat tokens[1:-1] as node-ish (skip model name / values)
        for node in toks[1:]:
            if re.match(r"^[A-Za-z0-9_]+$", node) and not re.match(r"^\d+(\.\d+)?[a-zA-Z]*$", node):
                pin_counts[node] = pin_counts.get(node, 0) + 1
    return sum(1 for node, c in pin_counts.items() if c == 1 and node not in ("0", "vdd", "vss", "gnd"))


def all_signal_devices_saturated(sim: SimResult) -> bool:
    """
    Anti-reward-hack check: signal-path transistors must be in saturation.
    A circuit can converge with devices in triode/cutoff and still 'simulate'
    while not amplifying. Customize which devices are 'signal-path' for your
    topologies (here: every MOS must be in saturation).
    """
    return all(
        region == "saturation"
        for dev, region in sim.device_regions.items()
        if dev.upper().startswith("M")
    )


# ----------------------------------------------------------------------------
# The simulation call (STUBBED internals)
# ----------------------------------------------------------------------------
def run_spice(netlist: str, testbench: str, timeout_s: int = 30) -> SimResult:
    """
    Run the netlist through the simulator and return a SimResult.

    TODO (wire to ngspice/PySpice):
      1. Inject/append the testbench analyses (.op/.ac/.tran + .measure lines, or
         drive measurements from PySpice analysis objects).
      2. Run the simulator with a timeout.
      3. Detect convergence (parse log / catch PySpice errors).
      4. Extract measurements (dc_gain_db, ugf_hz, phase_margin_deg, power_w, ...).
      5. Extract per-device operating regions (from .op output / model card region flags).

    CACHING: callers should cache by hash(netlist + testbench); the policy
    regenerates near-duplicates constantly during RL.
    """
    raise NotImplementedError(
        "Implement run_spice() against ngspice/PySpice. "
        "Return SimResult(converged, measurements, device_regions, raw_log)."
    )


# Example of the shell-out skeleton (left commented; fill in measurement parsing):
#
# def run_spice(netlist, testbench, timeout_s=30):
#     full = inject_testbench(netlist, testbench)
#     with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
#         f.write(full); path = f.name
#     try:
#         out = subprocess.run(["ngspice", "-b", path], capture_output=True,
#                              text=True, timeout=timeout_s)
#         log = out.stdout + out.stderr
#         converged = "doAnalyses: TRAN:  Timestep too small" not in log and out.returncode == 0
#         measurements = parse_measure_lines(log)        # you write this
#         regions = parse_device_regions(log)            # you write this
#         return SimResult(converged, measurements, regions, log)
#     except subprocess.TimeoutExpired:
#         return SimResult(False, {}, {}, "TIMEOUT")
