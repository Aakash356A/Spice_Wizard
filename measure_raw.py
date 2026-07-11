"""
Parses LTspice .raw waveform files into named engineering measurements
(AC gain, -3 dB bandwidth, settled transient value, etc.).

This is the Use Case 1 measurement layer (see USE_CASES_IMPLEMENTATION.md):
it turns a completed simulation's raw output into numbers that can be
compared against a Spec. Depends only on the `ltspice` package already
pinned in requirements.txt (ltspice==1.0.6).
"""

import math

import numpy as np
import ltspice


class MeasurementError(Exception):
    """Raised when a requested trace/node can't be found or measured."""


def load_traces(raw_path: str):
    """Parse a .raw file and return the parsed Ltspice object."""
    l = ltspice.Ltspice(raw_path)
    l.parse()
    return l


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _normalize_trace_name(node: str) -> str:
    """`"out"` -> `"V(out)"`; leaves `"I(R1)"` / `"V(out)"` untouched."""
    name = node.strip()
    lower = name.lower()
    if lower.startswith("v(") or lower.startswith("i("):
        return name
    return f"V({name})"


def _get_trace(traces, node: str, case: int = 0) -> np.ndarray:
    """Lookup of a trace by node/component name.

    `ltspice.Ltspice.get_data()` is already case-insensitive internally,
    but returns `None` (rather than raising) when the name isn't found --
    so we turn that into a clear MeasurementError instead of letting a
    confusing TypeError happen later on a None value.

    Accepts either a bare node name (`"out"`) or an explicit trace name
    (`"V(out)"`, `"I(R1)"`).
    """
    wanted = _normalize_trace_name(node)
    data = traces.get_data(wanted, case=case)
    if data is not None:
        return data

    available = list(getattr(traces, "variables", []) or [])
    raise MeasurementError(
        f"Trace '{node}' (looked up as '{wanted}') not found in raw file. "
        f"Available traces: {available}"
    )


def _nearest_index(axis: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(axis - value)))


def _to_db(complex_or_real: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.abs(complex_or_real))


# ---------------------------------------------------------------------------
# public measurements
# ---------------------------------------------------------------------------

def measure_ac_gain_db(
    traces,
    freq_hz: float,
    in_node: str,
    out_node: str,
    case: int = 0,
) -> float:
    """20*log10(|Vout|/|Vin|) at the frequency bin nearest `freq_hz`.

    Requires the netlist's testbench to include a `.ac` analysis.
    """
    freq = traces.get_frequency(case=case)
    vin = _get_trace(traces, in_node, case=case)
    vout = _get_trace(traces, out_node, case=case)

    idx = _nearest_index(freq, freq_hz)
    vin_mag = abs(vin[idx])
    if vin_mag == 0:
        raise MeasurementError(
            f"Input node '{in_node}' has zero amplitude at {freq_hz} Hz; "
            "cannot compute gain (divide by zero)."
        )
    return 20.0 * math.log10(abs(vout[idx]) / vin_mag)


def measure_ac_phase_deg(
    traces,
    freq_hz: float,
    in_node: str,
    out_node: str,
    case: int = 0,
) -> float:
    """Phase of Vout relative to Vin (degrees) at the bin nearest `freq_hz`."""
    freq = traces.get_frequency(case=case)
    vin = _get_trace(traces, in_node, case=case)
    vout = _get_trace(traces, out_node, case=case)

    idx = _nearest_index(freq, freq_hz)
    phase_rad = np.angle(vout[idx]) - np.angle(vin[idx])
    return float(np.degrees(phase_rad))


def measure_bandwidth_hz(
    traces,
    in_node: str,
    out_node: str,
    ref_gain_db: float = None,
    case: int = 0,
) -> float:
    """-3 dB (from reference) bandwidth, in Hz.

    `ref_gain_db` defaults to the gain at the lowest simulated frequency
    (assumes a flat passband starting near DC, which holds for typical
    op-amp gain-block testbenches). Returns `float("inf")` if the gain
    never drops 3 dB within the simulated sweep.
    """
    freq = traces.get_frequency(case=case)
    vin = _get_trace(traces, in_node, case=case)
    vout = _get_trace(traces, out_node, case=case)

    gain_db = _to_db(vout) - _to_db(vin)
    if ref_gain_db is None:
        ref_gain_db = float(gain_db[0])
    target = ref_gain_db - 3.0

    below = np.where(gain_db <= target)[0]
    if below.size == 0:
        return float("inf")

    idx = int(below[0])
    if idx == 0:
        return float(freq[0])

    # Interpolate in log-frequency space (sweeps are typically log/dec).
    f0, f1 = float(freq[idx - 1]), float(freq[idx])
    g0, g1 = float(gain_db[idx - 1]), float(gain_db[idx])
    frac = 0.0 if g1 == g0 else (target - g0) / (g1 - g0)
    log_f = math.log10(f0) + frac * (math.log10(f1) - math.log10(f0))
    return 10.0 ** log_f


def measure_tran_gain_db(
    traces,
    in_node: str,
    out_node: str,
    settle_frac: float = 0.5,
    case: int = 0,
) -> float:
    """AC gain (dB) estimated from a *transient* sine-wave simulation, via
    the peak-to-peak amplitude ratio of Vout/Vin over the later portion of
    the run (skipping the first `settle_frac` fraction to avoid startup
    transients).

    Many real ADI application netlists test gain with `.tran` + a SINE
    source at a fixed test frequency rather than a `.ac` sweep -- use this
    when `measure_ac_gain_db` isn't applicable (no AC analysis present).
    """
    vin = _get_trace(traces, in_node, case=case)
    vout = _get_trace(traces, out_node, case=case)

    start = int(len(vin) * settle_frac)
    vin_pp = float(np.ptp(vin[start:]))
    vout_pp = float(np.ptp(vout[start:]))
    if vin_pp == 0:
        raise MeasurementError(
            f"Input node '{in_node}' has zero peak-to-peak amplitude in the "
            "steady-state window; cannot compute gain."
        )
    return 20.0 * math.log10(vout_pp / vin_pp)


def measure_tran_final_value(traces, node: str, case: int = 0) -> float:
    """Last sample of a transient trace (e.g. a settled DC output)."""
    data = _get_trace(traces, node, case=case)
    last = data[-1]
    return float(last.real) if np.iscomplexobj(data) else float(last)


def measure_tran_settling_time_s(
    traces,
    node: str,
    final_value: float = None,
    tolerance_pct: float = 1.0,
    case: int = 0,
) -> float:
    """Time at which the transient trace enters and stays within
    `tolerance_pct` % of its final value. `float("inf")` if it never
    settles within the simulated window.
    """
    time = traces.get_time(case=case)
    data = _get_trace(traces, node, case=case)
    data = data.real if np.iscomplexobj(data) else data

    if final_value is None:
        final_value = float(data[-1])

    band = abs(final_value) * (tolerance_pct / 100.0) or (tolerance_pct / 100.0)
    within = np.abs(data - final_value) <= band
    outside = np.where(~within)[0]
    if outside.size == 0:
        return float(time[0])
    last_outside = int(outside[-1])
    if last_outside + 1 >= len(time):
        return float("inf")
    return float(time[last_outside + 1])


if __name__ == "__main__":
    # Quick manual sanity check, independent of sim_harness/SimulationRunner:
    #   python measure_raw.py path/to/some.raw
    import sys

    if len(sys.argv) != 2:
        print("Usage: python measure_raw.py <path-to-.raw-file>")
        sys.exit(1)

    t = load_traces(sys.argv[1])
    print("Available traces:", list(getattr(t, "variables", [])))
