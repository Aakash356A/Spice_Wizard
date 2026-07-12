"""
grpo_reward.py
==============
Reward function for RL (GRPO/RLVR) fine-tuning of a SPICE-netlist generator.

DESIGN PRINCIPLE: gated -> shaped -> saturated.
  - GATE on cheap binary checks (format, parse, converge) before spending
    simulation on metrics.
  - SHAPE metric rewards (continuous) so RL has a gradient to climb.
  - SATURATE each metric at 1.0 so no single metric can be over-satisfied to
    buy off another failing one.

This consumes the harness in sim_harness.py. Wire run_spice() there to a real
simulator before using this in a training loop.

------------------------------------------------------------------------------
REWARD-HACKING TRAPS (this is where the project is won or lost):

 1. UNBOUNDED METRICS -> single-metric maximization. If "gain" gives unlimited
    reward, the policy maximizes gain and lets power/bandwidth fail.
    FIX: saturate every metric at 1.0 (saturating_score below).

 2. "SIMULATES" != "WORKS". A circuit can converge with transistors in
    triode/cutoff -- it doesn't amplify but passes a naive "did it simulate" gate.
    FIX: operating-region check (all_signal_devices_saturated) as a penalty.
    This is the gap between ~75% simulation-validity and ~45% spec-fulfillment.

 3. DC-ONLY REWARD -> correct bias, zero AC function.
    FIX: always include a metric from the functional analysis (gain/UGF from .ac).

 4. TESTBENCH OVERFITTING. A fixed narrow testbench gets gamed.
    FIX: randomize across rollouts -- vary common-mode, load, ideally PVT corners
    (Spec.cm_voltage / Spec.c_load / Spec.corners).

 5. CoT LENGTH/FORMAT HACKING. If verbose plans earn reward, the model writes
    impressive plans decoupled from the netlist.
    FIX: keep the format reward tiny; reward correctness, not eloquence.

 6. "ALL-WRONG GROUP" (GRPO-specific). GRPO normalizes advantage WITHIN a group
    of samples for the same prompt. If every sample hits the floor, variance = 0
    -> gradient = 0 -> no learning.
    FIX: curriculum. Start RL from the SFT checkpoint on EASY specs where some
    samples already pass, then ramp difficulty.

 7. SURROGATE BLIND SPOTS. If you use a learned surrogate instead of SPICE in the
    inner loop, the policy WILL exploit its inaccuracies.
    FIX: periodically re-validate winners with real SPICE; retrain the surrogate
    on the policy's current distribution.

ALWAYS hand-sanity-check this reward (see __main__) BEFORE any RL run: confirm
known-good netlists score higher than known-bad ones. Many RL runs silently fail
because the reward quietly ranks things wrong.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import math

from sim_harness import (
    Spec,
    Metric,
    run_spice,
    parses_as_spice,
    all_signal_devices_saturated,
    device_count,
    count_floating_or_shorted_nodes,
)

# Field markers must match prepare_sft_data.py
PLAN_HEADER = "### DESIGN PLAN"
NETLIST_HEADER = "### NETLIST"


# ----------------------------------------------------------------------------
# Output parsing
# ----------------------------------------------------------------------------
def has_plan_and_netlist(text: str) -> bool:
    return PLAN_HEADER in text and NETLIST_HEADER in text


def split_plan_netlist(text: str) -> tuple[str, str]:
    """Return (plan, netlist). Assumes has_plan_and_netlist(text) is True."""
    plan_part = text.split(PLAN_HEADER, 1)[1]
    plan, netlist = plan_part.split(NETLIST_HEADER, 1)
    return plan.strip(), netlist.strip()


# ----------------------------------------------------------------------------
# Per-metric shaping (saturated)
# ----------------------------------------------------------------------------
def saturating_score(x: float, target: float, tol: float, direction: str) -> float:
    """
    Map one measured metric to [0, 1], saturating at 1.0 when the target is met.

    direction:
      "meet_at_least" -> ratio x/target, capped at 1   (gain, UGF, phase margin)
      "stay_below"    -> ratio target/x, capped at 1   (power, area)
      "hit_target"    -> smooth quadratic decay around target, width = tol
    """
    eps = 1e-12
    if direction == "meet_at_least":
        return max(0.0, min(1.0, x / (target + eps)))
    if direction == "stay_below":
        return max(0.0, min(1.0, target / (x + eps)))
    if direction == "hit_target":
        return math.exp(-(((x - target) / (tol + eps)) ** 2))
    raise ValueError(f"unknown direction: {direction}")


# ----------------------------------------------------------------------------
# The reward
# ----------------------------------------------------------------------------
def compute_reward(output_text: str, spec: Spec) -> float:
    """
    Returns a scalar reward. Valid, spec-meeting circuits approach +1.0;
    unusable outputs hit negative floors. Tune the floor magnitudes to your
    GRPO setup (they set how strongly the policy avoids each failure mode).
    """
    # 0. Format gate -- small weight, just keep the CoT structure intact.
    if not has_plan_and_netlist(output_text):
        return -1.0
    plan, netlist = split_plan_netlist(output_text)

    # 1. Syntax gate.
    if not parses_as_spice(netlist):
        return -0.8

    # 2. Simulation gate.
    sim = run_spice(netlist, spec.testbench)        # cache by hash(netlist+testbench)
    if not sim.converged:
        return -0.5

    # 3. Operating-region check (anti-hack trap #2).
    region_penalty = 0.0 if all_signal_devices_saturated(sim) else -0.3

    # 4. Per-spec metric reward -- the actual learning signal.
    total_w = sum(m.weight for m in spec.metrics) or 1.0
    metric_score = 0.0
    for m in spec.metrics:
        try:
            measured = sim.measure(m.name)
        except KeyError:
            # analysis didn't produce this metric -> treat as failed, not crash.
            continue
        metric_score += m.weight * saturating_score(measured, m.target, m.tol, m.direction)
    metric_score /= total_w                          # in [0, 1]

    # 5. Soft structural constraints (small penalties).
    metric_score -= 0.05 * max(0, device_count(netlist) - spec.max_devices)
    metric_score -= 0.10 * count_floating_or_shorted_nodes(netlist)

    return metric_score + region_penalty             # roughly [-0.4, 1.0] on valid circuits


# ----------------------------------------------------------------------------
# Hand sanity-check (run this BEFORE any RL run)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # A fake harness so this file runs standalone for the ordering sanity-check.
    # In real use, sim_harness.run_spice hits ngspice and you delete this shim.
    import sim_harness

    def _fake_run(netlist, testbench, timeout_s=30):
        # Pretend: a "good" netlist (contains 'M4' mirror + diff pair) meets spec
        # and is saturated; a "bad" one converges but sits in triode and underperforms.
        good = "M1" in netlist and "M4" in netlist
        if good:
            return sim_harness.SimResult(
                converged=True,
                measurements={"dc_gain_db": 42.0, "ugf_hz": 6.0e7, "power_w": 36e-6},
                device_regions={"M1": "saturation", "M2": "saturation",
                                "M3": "saturation", "M4": "saturation", "M5": "saturation"},
            )
        return sim_harness.SimResult(
            converged=True,
            measurements={"dc_gain_db": 8.0, "ugf_hz": 9.0e6, "power_w": 120e-6},
            device_regions={"M1": "triode", "M5": "saturation"},
        )

    # NOTE: compute_reward looks up `run_spice` in THIS module's globals (it was
    # imported via `from sim_harness import run_spice`). So we rebind the global
    # here, not sim_harness.run_spice. Delete this whole shim in real use.
    run_spice = _fake_run  # noqa: F811  (demo-only override)

    spec = Spec(
        metrics=[
            Metric("dc_gain_db", target=40.0, tol=5.0, direction="meet_at_least", weight=1.0),
            Metric("ugf_hz", target=50e6, tol=10e6, direction="meet_at_least", weight=1.0),
            Metric("power_w", target=50e-6, tol=10e-6, direction="stay_below", weight=1.0),
        ],
        testbench="ac_ota_v1",
        max_devices=8,
    )

    good_out = (
        f"{PLAN_HEADER}\n5T OTA, NMOS pair, PMOS mirror load...\n\n"
        f"{NETLIST_HEADER}\nM1 n1 inp tail 0 nfet\nM2 out inn tail 0 nfet\n"
        f"M3 n1 n1 vdd vdd pfet\nM4 out n1 vdd vdd pfet\nM5 tail vb 0 0 nfet\n.end"
    )
    bad_out = (
        f"{PLAN_HEADER}\nsingle device...\n\n"
        f"{NETLIST_HEADER}\nM1 out gate 0 0 nfet\nRD vdd out 1k\n.end"
    )
    malformed = "no markers here, just text"

    print(f"good circuit  : {compute_reward(good_out, spec):+.3f}   (expect highest; <1.0 here due to dangling-node penalty in the toy netlist)")
    print(f"bad circuit   : {compute_reward(bad_out, spec):+.3f}   (expect low: triode devices + missed specs)")
    print(f"malformed     : {compute_reward(malformed, spec):+.3f}   (expect -1.0: no plan/netlist markers)")
    print("\nSanity check passes if good > bad > malformed. If not, fix the reward "
          "BEFORE training.")
