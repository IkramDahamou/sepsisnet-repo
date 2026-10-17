"""Revision 4, Stage 2 Part B: summarize the 10-seed sweep from saved files only.

Per seed: scripts/evaluate_methods.py -> runs/sweep/seed_<n>/metrics.json/.md
(every metric recomputed from that seed's predictions.npz), and escalation on
the corrected-MC predictions (per-stay reward, direct state->action mapping,
legacy-stream median over 1,000 shuffled orders, rng 20261015).

Across seeds (ALL EXPLORATORY):
  - per-method mean, SD, t-based 95% CI (df = n-1)
  - paired comparisons: Wilcoxon signed-rank (two-sided, scipy default), n+/n-,
    paired mean difference with t-based 95% CI; Holm within each family
      family "discrimination": corrected-det, corrected-legacy, det-BiLSTM,
                               corrected-BiLSTM, each for AUROC and AUPRC (8 tests)
      family "error_detection": sigma minus -|p-0.5| baseline, corrected and legacy (2 tests)
  - Table 7 check: deterministic and BiLSTM-only per-seed AUROC/AUPRC vs
    reference_data/rev3_multiseed_results_10seed_deterministic.json (4-dp values),
    plus bitwise prediction checks vs reference_data/determinism_check_seed{13,21}_a.npz

Replicates: runs/sweep_a is the REPORTED result (decided before running);
runs/sweep_b only measures reproducibility. Per seed: max |delta| on all
prediction arrays and weight datasets (a vs b), and the tolerance criterion:
a seed PASSES if AUROC, AUPRC, Brier, ECE, variance ratio, error-detection
AUROC (sigma and baseline) and G/Y/O/R state counts are identical at 3
decimals for every method. Failing seeds are flagged and kept.

Usage:
  PYTHONDONTWRITEBYTECODE=1 python scripts/summarize_sweep.py --out results/sweep_summary.json --md results/sweep_summary.md
"""
import argparse
import json
import os
import subprocess
import sys

import h5py
import numpy as np
from scipy import stats

sys.dont_write_bytecode = True
REV4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.dirname(REV4)
sys.path.insert(0, os.path.join(REV4, "sepsisnet", "src"))
from mdp_escalation import AlertEscalationMDP  # noqa: E402

SWEEP = os.path.join(REV4, "runs", "sweep_a")
SWEEP_B = os.path.join(REV4, "runs", "sweep_b")
TOL_KEYS = ["AUROC", "AUPRC", "Brier", "ECE_10bin", "variance_ratio_sepsis_over_normal",
            "AUROC_error_given_sigma", "AUROC_error_baseline_neg_abs_p_minus_0.5"]


def evaluate(d):
    mj = os.path.join(d, "metrics.json")
    if not os.path.exists(mj):
        subprocess.run([sys.executable, "-I", os.path.join(REV4, "scripts", "evaluate_methods.py"),
                        os.path.join(d, "predictions.npz"), "--out", mj, "--md", os.path.join(d, "metrics.md")],
                       check=True, capture_output=True)
    return json.load(open(mj))["methods"]


def h5_arrays(p):
    out = {}
    with h5py.File(p, "r") as f:
        f.visititems(lambda n, o: out.__setitem__(n, o[()]) if isinstance(o, h5py.Dataset) else None)
    return out


def repro_ab(s):
    da, db = os.path.join(SWEEP, f"seed_{s}"), os.path.join(SWEEP_B, f"seed_{s}")
    if not os.path.exists(os.path.join(db, "manifest.json")):
        return {"status": "run b missing"}
    pa, pb = np.load(os.path.join(da, "predictions.npz")), np.load(os.path.join(db, "predictions.npz"))
    pred = {k: float(np.abs(pa[k].astype(float) - pb[k].astype(float)).max()) for k in pa.files}
    w = {}
    for f in ("cnn_bilstm.weights.h5", "bilstm_only.weights.h5"):
        A, B = h5_arrays(os.path.join(da, f)), h5_arrays(os.path.join(db, f))
        w[f] = max(float(np.abs(A[k].astype(float) - B[k].astype(float)).max()) for k in A if np.issubdtype(A[k].dtype, np.number))
    Ma, Mb = evaluate(da), evaluate(db)
    mism = []
    for m in METHODS:
        for k in TOL_KEYS:
            x, y = Ma[m].get(k), Mb[m].get(k)
            if (x is None) != (y is None) or (x is not None and round(x, 3) != round(y, 3)):
                mism.append({"method": m, "metric": k, "a": x, "b": y})
        if Ma[m]["states_GYOR_alpha0.10"] != Mb[m]["states_GYOR_alpha0.10"]:
            mism.append({"method": m, "metric": "states", "a": Ma[m]["states_GYOR_alpha0.10"], "b": Mb[m]["states_GYOR_alpha0.10"]})
    return {"status": "PASS" if not mism else "FAIL (flagged, kept)", "bitwise_identical": max(pred.values()) == 0 and max(w.values()) == 0,
            "max_abs_diff_predictions": max(pred.values()), "max_abs_diff_predictions_by_array": pred,
            "max_abs_diff_weights": w, "metric_mismatches_at_3dp": mism}
METHODS = ["deterministic", "legacy_mc", "corrected_mc", "bilstm_only"]
METRICS = ["AUROC", "AUPRC", "Brier", "ECE_10bin", "calibration_in_the_large",
           "variance_ratio_sepsis_over_normal", "AUROC_error_given_sigma",
           "AUROC_error_baseline_neg_abs_p_minus_0.5", "AUROC_error_sigma_minus_baseline"]
N_ORDERS = 1000
ORDER_SEED = 20261015


def tci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); sd = x.std(ddof=1) if n > 1 else float("nan")
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": float(m), "sd": float(sd), "t_ci95": [float(m - h), float(m + h)],
            "min": float(x.min()), "max": float(x.max())}


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def escalation(pred):
    d = np.load(pred); y = d["yte"].astype(int); p = d["p_mc_corrected"]; s = d["std_corrected"]
    ids = np.arange(len(y))
    per_stay = float(AlertEscalationMDP(2).process_sequence(p, s, y, stay_ids=ids)[3].sum())
    thr1 = float(AlertEscalationMDP(1).process_sequence(p, s, y)[3].sum())
    saved = float(AlertEscalationMDP(2).process_sequence(p, s, y)[3].sum())
    rng = np.random.default_rng(ORDER_SEED); m2 = AlertEscalationMDP(2); tot = []
    for _ in range(N_ORDERS):
        o = rng.permutation(len(y)); tot.append(m2.process_sequence(p[o], s[o], y[o])[3].sum())
    tot = np.array(tot)
    missed = int(sum(1 for a in AlertEscalationMDP(2).process_sequence(p, s, y, stay_ids=ids)[0]
                     if a["true_label"] == 1 and a["action"] == "No-Action"))
    return {"per_stay_reward": per_stay, "state_to_action_threshold1_reward": thr1,
            "legacy_stream_saved_order_reward": saved,
            "legacy_stream_shuffled_median": float(np.median(tot)),
            "legacy_stream_shuffled_p2.5_p97.5": [float(np.percentile(tot, 2.5)), float(np.percentile(tot, 97.5))],
            "per_stay_missed_positives": missed}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--md", required=True)
    a = ap.parse_args()
    for f in (a.out, a.md):
        if os.path.exists(f):
            sys.exit(f"{f} exists; refusing to overwrite")
    t7 = {r["seed"]: r for r in json.load(open(os.path.join(REV4, "reference_data", "rev3_multiseed_results_10seed_deterministic.json")))}
    seeds = list(t7)
    res = {"exploratory": True, "seeds_planned": seeds, "seeds": {}, "missing_or_failed": []}
    per = {m: {k: [] for k in METRICS} for m in METHODS}; esc = []
    for s in seeds:
        d = os.path.join(SWEEP, f"seed_{s}")
        if not os.path.exists(os.path.join(d, "manifest.json")):
            res["missing_or_failed"].append(s); continue
        M = evaluate(d)
        e = escalation(os.path.join(d, "predictions.npz")); esc.append(e)
        cfg = json.load(open(os.path.join(d, "config.json")))
        res["seeds"][s] = {"metrics": {m: {k: M[m].get(k) for k in METRICS + ["errors_at_0.5", "FN", "FP"]} for m in METHODS},
                           "escalation_corrected": e, "scheme": cfg["scheme"], "environment": cfg["environment"],
                           "elapsed_sec": cfg["elapsed_sec"]}
        for m in METHODS:
            for k in METRICS:
                if M[m].get(k) is not None:
                    per[m][k].append(M[m][k])
    done = list(res["seeds"])
    res["reproducibility_a_vs_b"] = {s: repro_ab(s) for s in done}
    res["reproducibility_rule"] = ("seed passes if AUROC, AUPRC, Brier, ECE, variance ratio, error-detection AUROC "
                                   "(sigma and baseline) and state counts are identical at 3 decimals for all methods; "
                                   "run a is reported, run b measures reproducibility only")
    res["per_method_summary"] = {m: {k: tci(v) for k, v in per[m].items() if len(v) == len(done) and v} for m in METHODS}

    # ---- Table 7 check ----
    t7c = {}
    for s in done:
        row = {}
        for m_new, m_old in (("deterministic", "CNN_LSTM_deterministic"), ("bilstm_only", "LSTM_only"), ("legacy_mc", "CNN_LSTM_MC")):
            for k in ("AUROC", "AUPRC"):
                new = res["seeds"][s]["metrics"][m_new][k]; old = t7[s][m_old][k]
                row[f"{m_new}_{k}"] = {"new": new, "table7": old, "match_at_4dp": round(new, 4) == old,
                                       "abs_diff": abs(round(new, 4) - old)}
        t7c[s] = row
    bit = {}
    for s in (13, 21):
        if s in done:
            o = np.load(os.path.join(REV4, "reference_data", f"determinism_check_seed{s}_a.npz"))
            n = np.load(os.path.join(SWEEP, f"seed_{s}", "predictions.npz"))
            bit[s] = {"p_det_bitwise": bool(np.array_equal(n["p_det"], o["p_det"])),
                      "p_det_max_abs_diff": float(np.abs(n["p_det"].astype(float) - o["p_det"].astype(float)).max()),
                      "p_lstm_bitwise": bool(np.array_equal(n["p_lstm"], o["p_lstm"])),
                      "p_lstm_max_abs_diff": float(np.abs(n["p_lstm"].astype(float) - o["p_lstm"].astype(float)).max())}
    res["table7_check"] = {"note": "legacy_mc rows are NOT expected to match: Revision 3 MC used unseeded/unreset Dropout "
                                   "generator states; Revision 4 resets them from the seed. Deterministic and BiLSTM-only are the check.",
                           "per_seed": t7c, "bitwise_vs_revision3_determinism_check": bit,
                           "deterministic_and_bilstm_all_match_4dp": all(v["match_at_4dp"] for r in t7c.values()
                                                                          for kk, v in r.items() if not kk.startswith("legacy"))}

    # ---- paired comparisons ----
    def paired(x1, x2):
        dlt = np.asarray(x1, float) - np.asarray(x2, float)
        w = stats.wilcoxon(dlt) if np.any(dlt != 0) else None
        c = tci(dlt)
        return {"mean_diff": c["mean"], "mean_diff_t_ci95": c["t_ci95"], "sd_diff": c["sd"],
                "n_plus": int((dlt > 0).sum()), "n_minus": int((dlt < 0).sum()), "n_zero": int((dlt == 0).sum()),
                "wilcoxon_W": None if w is None else float(w.statistic), "p_raw": 1.0 if w is None else float(w.pvalue)}
    fam1 = []
    for m1, m2 in (("corrected_mc", "deterministic"), ("corrected_mc", "legacy_mc"),
                   ("deterministic", "bilstm_only"), ("corrected_mc", "bilstm_only")):
        for k in ("AUROC", "AUPRC"):
            fam1.append({"comparison": f"{m1} - {m2}", "metric": k, **paired(per[m1][k], per[m2][k])})
    fam2 = []
    for m in ("corrected_mc", "legacy_mc"):
        fam2.append({"comparison": f"{m}: sigma - baseline", "metric": "AUROC(error)",
                     **paired(per[m]["AUROC_error_given_sigma"], per[m]["AUROC_error_baseline_neg_abs_p_minus_0.5"])})
    for fam in (fam1, fam2):
        adj = holm(np.array([r["p_raw"] for r in fam]))
        for r, p in zip(fam, adj):
            r["p_holm"] = float(p)
    res["paired_comparisons"] = {"family_discrimination": fam1, "family_error_detection": fam2,
                                 "label": "exploratory; Holm within family; n = number of completed seeds"}

    # ---- escalation across seeds ----
    res["escalation_across_seeds"] = {k: tci([e[k] for e in esc]) for k in
                                      ("per_stay_reward", "state_to_action_threshold1_reward",
                                       "legacy_stream_shuffled_median", "per_stay_missed_positives")}
    res["limitations"] = ["10 training seeds on one synthetic generator; data and split vary with the seed exactly as in Revision 3 "
                          "(generator and split are seeded by the same seed), so seed-to-seed variation mixes training randomness "
                          "and cohort/split resampling from ONE synthetic generator, not population variability",
                          "76 test stays with 18 positives per seed: per-seed metrics are noisy",
                          "all statistical comparisons exploratory; non-significant results do not show equivalence"]
    json.dump(res, open(a.out, "w"), indent=2)

    # ---- markdown ----
    L = [f"Seeds completed (run a, reported): {done}; missing/failed: {res['missing_or_failed']}", "",
         "| Seed | a vs b | bitwise | max abs diff predictions | max abs diff weights (CNN-BiLSTM / BiLSTM) | 3-dp mismatches |", "|---|---|---|---|---|---|"]
    for s_, r in res["reproducibility_a_vs_b"].items():
        if "max_abs_diff_predictions" in r:
            ww = r["max_abs_diff_weights"]
            L.append(f"| {s_} | {r['status']} | {r['bitwise_identical']} | {r['max_abs_diff_predictions']:.2e} | "
                     f"{ww['cnn_bilstm.weights.h5']:.2e} / {ww['bilstm_only.weights.h5']:.2e} | {len(r['metric_mismatches_at_3dp'])} |")
        else:
            L.append(f"| {s_} | {r['status']} | | | | |")
    L += ["",
         "| Metric (mean ± SD [t 95% CI]) | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for k in METRICS:
        cells = []
        for m in METHODS:
            v = res["per_method_summary"][m].get(k)
            cells.append("n/a" if v is None else f"{v['mean']:.3f} ± {v['sd']:.3f} [{v['t_ci95'][0]:.3f}, {v['t_ci95'][1]:.3f}]")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L += ["", "| Comparison | Metric | Mean diff [t 95% CI] | n+/n-/0 | W | p raw | p Holm |", "|---|---|---|---|---|---|---|"]
    for r in fam1 + fam2:
        L.append(f"| {r['comparison']} | {r['metric']} | {r['mean_diff']:+.3f} [{r['mean_diff_t_ci95'][0]:+.3f}, {r['mean_diff_t_ci95'][1]:+.3f}] | "
                 f"{r['n_plus']}/{r['n_minus']}/{r['n_zero']} | {r['wilcoxon_W']} | {r['p_raw']:.4f} | {r['p_holm']:.4f} |")
    L += ["", "| Seed | det AUROC new/T7 | det AUPRC new/T7 | BiLSTM AUROC new/T7 | BiLSTM AUPRC new/T7 |", "|---|---|---|---|---|"]
    for s, r in t7c.items():
        L.append(f"| {s} | " + " | ".join(f"{r[c]['new']:.4f}/{r[c]['table7']:.4f}{'' if r[c]['match_at_4dp'] else ' ≠'}"
                                          for c in ("deterministic_AUROC", "deterministic_AUPRC", "bilstm_only_AUROC", "bilstm_only_AUPRC")) + " |")
    L += ["", "| Seed | per-stay reward | threshold-1 reward | legacy stream shuffled median | per-stay missed positives |", "|---|---|---|---|---|"]
    for s in done:
        e = res["seeds"][s]["escalation_corrected"]
        L.append(f"| {s} | {e['per_stay_reward']:.0f} | {e['state_to_action_threshold1_reward']:.0f} | {e['legacy_stream_shuffled_median']:.0f} | {e['per_stay_missed_positives']} |")
    open(a.md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
