"""Escalation-policy analysis for one predictions file (Revision 4).

Simulated stay-level rule with ONE prediction per stay; not longitudinal
bedside alerting and not clinically validated.

Reports:
  - reward by cell (action taken / not taken x true label) for each variant
  - legacy stream: one consecutive counter across the stays in saved order
  - per-stay: counter kept per stay (order-invariant)
  - direct state->action mapping: consecutive_threshold = 1
  - shuffled-order distribution of the legacy stream (10,000 orders,
    np.random.default_rng(20261015))

Usage:
    python scripts/escalation_analysis.py runs/seed42_a/predictions.npz \
        --p-key p_mc_corrected --std-key std_corrected --out results/seed42_escalation_corrected.json
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
REV4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REV4, "sepsisnet", "src"))
from mdp_escalation import AlertEscalationMDP  # noqa: E402

N_ORDERS = 10000
ORDER_SEED = 20261015
R = {("no_action", 1): -500, ("no_action", 0): 10, ("action", 1): 100, ("action", 0): -50}


def summarise(alerts, rewards):
    st = collections.Counter(a["state"] for a in alerts)
    ac = collections.Counter(a["action"] for a in alerts)
    cells = collections.Counter(("no_action" if a["action"] == "No-Action" else "action", a["true_label"])
                                for a in alerts)
    cell_rows = {f"{k[0]}|y={k[1]}": {"n": cells[k], "reward": cells[k] * R[k]} for k in R}
    assert sum(v["reward"] for v in cell_rows.values()) == float(rewards.sum())
    return {"states_GYOR": [st[k] for k in ("GREEN", "YELLOW", "ORANGE", "RED")],
            "actions_NA_N_E_ICU": [ac[k] for k in ("No-Action", "Notify", "Escalate", "ICU-Code")],
            "reward_by_cell": cell_rows, "total_reward": float(rewards.sum()),
            "missed_positives": [{"stay_index": i, "state": a["state"], "p": round(a["sepsis_prob"], 4)}
                                 for i, a in enumerate(alerts)
                                 if a["true_label"] == 1 and a["action"] == "No-Action"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred"); ap.add_argument("--p-key", required=True)
    ap.add_argument("--std-key", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; refusing to overwrite")
    d = np.load(a.pred)
    y, p, s = d["yte"].astype(int), d[a.p_key], d[a.std_key]
    ids = np.arange(len(y))
    res = {"source": os.path.abspath(a.pred), "p_key": a.p_key, "std_key": a.std_key,
           "alpha": a.alpha, "note": "Simulated stay-level rule, one prediction per stay; "
                                     "not longitudinal bedside alerting; not clinically validated."}
    m2 = AlertEscalationMDP(consecutive_threshold=2, alpha=a.alpha)
    al, _, _, rw = m2.process_sequence(p, s, y)
    res["legacy_stream_saved_order"] = summarise(al, rw)
    al, _, _, rw = m2.process_sequence(p, s, y, stay_ids=ids)
    res["per_stay"] = summarise(al, rw)
    m1 = AlertEscalationMDP(consecutive_threshold=1, alpha=a.alpha)
    al, _, _, rw = m1.process_sequence(p, s, y)
    res["state_to_action_threshold1"] = summarise(al, rw)

    rng = np.random.default_rng(ORDER_SEED)
    tot = np.empty(N_ORDERS); fn = np.empty(N_ORDERS, dtype=int); per_stay_tot = set()
    for k in range(N_ORDERS):
        o = rng.permutation(len(y))
        al_o, _, _, r_o = m2.process_sequence(p[o], s[o], y[o])
        tot[k] = r_o.sum()
        fn[k] = sum(1 for x in al_o if x["true_label"] == 1 and x["action"] == "No-Action")
        if k < 500:
            per_stay_tot.add(float(m2.process_sequence(p[o], s[o], y[o], stay_ids=ids[o])[3].sum()))
    leg = res["legacy_stream_saved_order"]["total_reward"]
    res["shuffled_order_legacy_stream"] = {
        "n_orders": N_ORDERS, "rng_seed": ORDER_SEED,
        "reward_min": float(tot.min()), "reward_p2.5": float(np.percentile(tot, 2.5)),
        "reward_median": float(np.median(tot)), "reward_p97.5": float(np.percentile(tot, 97.5)),
        "reward_max": float(tot.max()),
        "share_orders_reward_le_saved_order": float((tot <= leg).mean()),
        "missed_positives_distribution": {int(k): int(v) for k, v in sorted(collections.Counter(fn).items())},
    }
    res["per_stay_order_invariance_check"] = {"orders_checked": 500,
                                              "distinct_totals": sorted(per_stay_tot)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
