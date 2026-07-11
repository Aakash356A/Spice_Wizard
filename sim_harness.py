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
from pathlib import Path
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
    diagnostics: list[str] = field(default_factory=list)

    def measure(self, name: str) -> float:
        """Return a measured metric; raise if the analysis didn't produce it."""
        if name not in self.measurements:
            raise KeyError(f"metric '{name}' not measured (check testbench/analyses)")
        return self.measurements[name]


# ----------------------------------------------------------------------------
# Cheap syntax gate (real, not stubbed) -- runs before any simulation
# ----------------------------------------------------------------------------
# LTspice application netlists commonly use unicode instance names such as
# `X§U1`. `\w*` rejects that `§`, so accept every non-whitespace suffix after
# a valid SPICE element designator instead.
_ELEMENT_RE = re.compile(r"^\s*[MmRrCcLlDdQqVvIiXxEeFfGgHhKkSsTt]\S*\s+\S+", re.MULTILINE)


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
def run_spice(
    netlist: str,
    testbench: str,
    timeout_s: int = 30,
    in_node: str = "IN",
    out_node: str = "OUT",
    freq_hz: float = 1000.0,
    requested_measurements: set[str] | None = None,
) -> SimResult:
    """
    Run the netlist through LTspice batch mode (app.simulation_runner.
    SimulationRunner) and extract measurements from the resulting .raw file.

    Unlike the original ngspice/PySpice plan, this reuses the existing
    working LTspice runner: the netlist's own .ac/.tran directives are used
    as-is (nothing is injected), so `testbench` is currently just carried
    through onto the Spec for interface compatibility rather than used to
    modify the netlist.

    `in_node`/`out_node`/`freq_hz` control the default AC gain/bandwidth
    measurements -- override them to match the real node names in your
    netlist (ADI app-note netlists commonly use auto-generated names like
    "N001" rather than "OUT"; run `python measure_raw.py <raw file>` on a
    prior simulation to see the actual trace names).

    `requested_measurements` lets a caller avoid unrelated extraction work
    and diagnostics. For example, a transient gain-only check should request
    `{"gain_db"}` rather than also reporting that AC-only bandwidth is absent.

    CACHING: callers should cache by hash(netlist + testbench); the policy
    regenerates near-duplicates constantly during RL.
    """
    if not parses_as_spice(netlist):
        return SimResult(
            converged=False,
            measurements={},
            device_regions={},
            raw_log="Netlist failed cheap syntax gate (parses_as_spice): "
                    "missing .end or too few element lines.",
        )

    # Local imports: app.simulation_runner configures logging.basicConfig()
    # at module level, which we don't want to impose on every caller of
    # this module (e.g. pure parses_as_spice() checks that never simulate).
    from app.simulation_runner import SimulationRunner
    import measure_raw

    runner = SimulationRunner()
    if not runner.is_available():
        return SimResult(
            converged=False,
            measurements={},
            device_regions={},
            raw_log="LTspice executable not found on this system.",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        net_path = Path(tmpdir) / "circuit.net"
        net_path.write_text(netlist)

        result = runner.run_batch(net_path, timeout=timeout_s)

        def _read_log() -> str:
            log_path = result.get("log_path")
            if not log_path:
                return ""
            try:
                return Path(log_path).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return ""

        if not result.get("ok"):
            # Preserve LTspice's log alongside the runner's short summary.
            # The summary alone is often just "No .raw file generated"; the
            # log contains the actionable detail (missing model, convergence
            # failure, syntax location, etc.).
            raw_log_parts = [
                str(result.get("error", "") or result.get("stderr", "")),
                _read_log(),
            ]
            raw_log = "\n\n".join(part for part in raw_log_parts if part)
            return SimResult(converged=False, measurements={}, device_regions={}, raw_log=raw_log)

        traces = measure_raw.load_traces(result["raw_path"])

        requested = set(requested_measurements or {"gain_db", "bandwidth_hz"})
        measurements: dict = {}
        diagnostics: list[str] = []
        if "gain_db" in requested:
            try:
                # Prefer a real .ac sweep if the netlist has one.
                measurements["gain_db"] = measure_raw.measure_ac_gain_db(
                    traces, in_node=in_node, out_node=out_node
                    , freq_hz=freq_hz
                )
            except Exception as ac_error:
                try:
                    # Many real ADI app netlists test gain via .tran + a SINE
                    # source instead of a .ac sweep -- fall back to a
                    # peak-to-peak amplitude ratio in that case.
                    measurements["gain_db"] = measure_raw.measure_tran_gain_db(
                        traces, in_node=in_node, out_node=out_node
                    )
                except Exception as tran_error:
                    ac_detail = str(ac_error).strip() or "no AC frequency data"
                    tran_detail = str(tran_error).strip() or "no transient waveform data"
                    diagnostics.append(
                        "gain_db unavailable: "
                        f"AC measurement failed ({ac_detail}); "
                        f"transient measurement failed ({tran_detail})"
                    )
        if "bandwidth_hz" in requested:
            try:
                measurements["bandwidth_hz"] = measure_raw.measure_bandwidth_hz(
                    traces, in_node=in_node, out_node=out_node
                )
            except Exception as error:
                detail = str(error).strip() or "no AC frequency data; bandwidth requires a .ac sweep"
                diagnostics.append(f"bandwidth_hz unavailable: {detail}")

        unsupported = requested - {"gain_db", "bandwidth_hz"}
        for name in sorted(unsupported):
            diagnostics.append(f"{name} unavailable: no extractor is implemented for this metric.")

        return SimResult(
            converged=True,
            measurements=measurements,
            device_regions={},
            raw_log=_read_log(),
            diagnostics=diagnostics,
        )
