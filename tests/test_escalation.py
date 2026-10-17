"""Tests for the Revision 4 escalation-policy copy.

The legacy-default check is the only re-check performed on the old
Revision 3 predictions (reference_data/rev3_real_pred_full.npz).
"""
import collections
import os

import numpy as np

from mdp_escalation import AlertEscalationMDP
import mdp_escalation_original

OLD_NPZ = os.path.join(os.path.dirname(__file__), "..", "reference_data",
                       "rev3_real_pred_full.npz")


def _counts(alerts):
    s = collections.Counter(a["state"] for a in alerts)
    a_ = collections.Counter(a["action"] for a in alerts)
    return ([s[k] for k in ("GREEN", "YELLOW", "ORANGE", "RED")],
            [a_[k] for k in ("No-Action", "Notify", "Escalate", "ICU-Code")])


def test_legacy_default_reproduces_revision3_on_old_predictions():
    d = np.load(OLD_NPZ)
    y, p, s = d["yte"], d["p_mc"], d["std"]
    al, _, _, rw = AlertEscalationMDP().process_sequence(p, s, y)
    al_o, _, _, rw_o = mdp_escalation_original.AlertEscalationMDP(2).process_sequence(p, s, y)
    states, actions = _counts(al)
    print(f"\nstates G/Y/O/R={states}; actions NA/N/E/ICU={actions}; total={rw.sum()}; "
          f"identical to original module per stay={al == al_o}")
    assert states == [48, 8, 4, 16]
    assert actions == [54, 4, 2, 16]
    assert rw.sum() == -500.0
    assert al == al_o and np.array_equal(rw, rw_o)


def test_alpha_parameter_default_and_zero():
    d = np.load(OLD_NPZ)
    y, p, s = d["yte"], d["p_mc"], d["std"]
    st0, _ = _counts(AlertEscalationMDP(alpha=0.0).process_sequence(p, s, y)[0])
    print(f"\nalpha=0 states={st0} (revision3_results alpha_sensitivity alpha=0: 48/8/4/16)")
    assert st0 == [48, 8, 4, 16]


def test_per_stay_mode_is_order_invariant_and_never_reaches_threshold():
    d = np.load(OLD_NPZ)
    y, p, s = d["yte"], d["p_mc"], d["std"]
    ids = np.arange(len(y))
    m = AlertEscalationMDP()
    al, _, _, rw = m.process_sequence(p, s, y, stay_ids=ids)
    rng = np.random.default_rng(0)
    totals = set()
    for _ in range(200):
        o = rng.permutation(len(y))
        totals.add(float(m.process_sequence(p[o], s[o], y[o], stay_ids=ids[o])[3].sum()))
    _, actions = _counts(al)
    print(f"\nper-stay total={rw.sum()}; totals over 200 permutations={sorted(totals)}; actions={actions}")
    assert totals == {float(rw.sum())}
    assert actions[2] == 0  # Escalate needs >=2 consecutive predictions of one stay
    assert not any(a["state"] == "YELLOW" and a["action"] == "Notify" for a in al)


def test_per_stay_mode_applies_hysteresis_within_a_stay():
    """Rolling predictions of two interleaved stays: counters stay separate."""
    m = AlertEscalationMDP()
    p = np.array([0.2, 0.05, 0.2, 0.05, 0.4])
    s = np.zeros(5); y = np.array([1, 0, 1, 0, 1])
    ids = np.array(["A", "B", "A", "B", "A"])
    acts = [a["action"] for a in m.process_sequence(p, s, y, stay_ids=ids)[0]]
    legacy = [a["action"] for a in m.process_sequence(p, s, y)[0]]
    print(f"\nper-stay actions={acts}; legacy actions={legacy}")
    assert acts == ["No-Action", "No-Action", "Notify", "No-Action", "Escalate"]
    assert legacy == ["No-Action", "No-Action", "No-Action", "No-Action", "Notify"]
