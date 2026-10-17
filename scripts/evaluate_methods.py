"""Compute every Section 5C metric from a saved predictions.npz only.

Usage:
    python scripts/evaluate_methods.py runs/seed42_a/predictions.npz \
        --out results/seed42_metrics.json --md results/seed42_metrics.md

Methods: deterministic CNN-BiLSTM, legacy MC (BN batch stats), corrected MC
(BN inference, primary), BiLSTM-only. Bootstrap: 2,000 resamples of the test
stays with replacement (np.random.default_rng(42)), resamples with a single
class skipped (same rule as run_authoritative.py); the SAME resample indices
are used for every method, so differences are paired.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)

N_BOOT = 2000
BOOT_SEED = 42
ALPHA = 0.10
THR = 0.5


def ece10(y, p, bins=10):
    e = np.linspace(0, 1, bins + 1); ece = 0.0; n = len(y)
    for i in range(bins):
        lo, hi = e[i], e[i + 1]
        m = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if m.sum():
            ece += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def states(p, s):
    a = p + ALPHA * s
    return {"GREEN": int((a < 0.10).sum()), "YELLOW": int(((a >= 0.10) & (a < 0.30)).sum()),
            "ORANGE": int(((a >= 0.30) & (a < 0.60)).sum()), "RED": int((a >= 0.60).sum())}


def ci(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred"); ap.add_argument("--out", required=True); ap.add_argument("--md")
    a = ap.parse_args()
    for f in (a.out, a.md):
        if f and os.path.exists(f):
            sys.exit(f"{f} exists; refusing to overwrite")
    d = np.load(a.pred)
    y = d["yte"].astype(int)
    M = {
        "deterministic": (d["p_det"], None),
        "legacy_mc": (d["p_mc_legacy"], d["std_legacy"]),
        "corrected_mc": (d["p_mc_corrected"], d["std_corrected"]),
        "bilstm_only": (d["p_lstm"], None),
    }
    rng = np.random.default_rng(BOOT_SEED); N = len(y)
    boots = []
    while len(boots) < N_BOOT:   # count only valid resamples
        i = rng.integers(0, N, N)
        if 0 < y[i].sum() < N:
            boots.append(i)
    # NB: run_authoritative.py drew 2000 resamples and skipped invalid ones;
    # with 18/76 positives invalid resamples are essentially impossible.

    res = {"source": os.path.abspath(a.pred), "n_test": int(N), "n_positive": int(y.sum()),
           "bootstrap": {"n": N_BOOT, "rng_seed": BOOT_SEED, "paired": True},
           "alpha": ALPHA, "threshold": THR, "methods": {}, "paired_differences": {}}
    boot_auc = {}; boot_ap = {}
    for name, (p, s) in M.items():
        p = p.astype(np.float64)
        yp = (p >= THR).astype(int); err = (yp != y).astype(int)
        bau = [roc_auc_score(y[i], p[i]) for i in boots]
        bap = [average_precision_score(y[i], p[i]) for i in boots]
        boot_auc[name], boot_ap[name] = np.array(bau), np.array(bap)
        r = {"AUROC": float(roc_auc_score(y, p)), "AUROC_CI95": ci(bau),
             "AUPRC": float(average_precision_score(y, p)), "AUPRC_CI95": ci(bap),
             "Brier": float(brier_score_loss(y, p)), "ECE_10bin": ece10(y, p),
             "calibration_in_the_large": float(p.mean() - y.mean()),
             "errors_at_0.5": int(err.sum()),
             "FN": int(((yp == 0) & (y == 1)).sum()), "FP": int(((yp == 1) & (y == 0)).sum()),
             "mean_p": float(p.mean())}
        base = -np.abs(p - 0.5)
        r["AUROC_error_baseline_neg_abs_p_minus_0.5"] = (float(roc_auc_score(err, base))
                                                          if 0 < err.sum() < N else None)
        if s is not None:
            s = s.astype(np.float64); v = s ** 2
            r["variance_sepsis"] = float(v[y == 1].mean())
            r["variance_normal"] = float(v[y == 0].mean())
            r["variance_ratio_sepsis_over_normal"] = r["variance_sepsis"] / r["variance_normal"]
            r["mean_std"] = float(s.mean())
            r["spearman_std_vs_p(1-p)"] = float(spearmanr(s, p * (1 - p)).statistic)
            if 0 < err.sum() < N:
                r["AUROC_error_given_sigma"] = float(roc_auc_score(err, s))
                diffs, a1, a2 = [], [], []
                for i in boots:
                    e = err[i]
                    if 0 < e.sum() < N:
                        x1, x2 = roc_auc_score(e, s[i]), roc_auc_score(e, base[i])
                        a1.append(x1); a2.append(x2); diffs.append(x1 - x2)
                r["AUROC_error_given_sigma_CI95"] = ci(a1)
                r["AUROC_error_baseline_CI95"] = ci(a2)
                r["AUROC_error_sigma_minus_baseline"] = r["AUROC_error_given_sigma"] - r["AUROC_error_baseline_neg_abs_p_minus_0.5"]
                r["AUROC_error_sigma_minus_baseline_CI95"] = ci(diffs)
                r["error_detection_bootstrap_valid_resamples"] = len(diffs)
        r["states_GYOR_alpha0.10"] = states(p, s if s is not None else np.zeros_like(p))
        r["states_sigma_used"] = s is not None
        res["methods"][name] = r

    pairs = [("corrected_mc", "deterministic"), ("legacy_mc", "deterministic"),
             ("corrected_mc", "legacy_mc"), ("corrected_mc", "bilstm_only"),
             ("deterministic", "bilstm_only")]
    for m1, m2 in pairs:
        p1, p2 = M[m1][0].astype(np.float64), M[m2][0].astype(np.float64)
        da = boot_auc[m1] - boot_auc[m2]; dp = boot_ap[m1] - boot_ap[m2]
        res["paired_differences"][f"{m1} - {m2}"] = {
            "dAUROC": res["methods"][m1]["AUROC"] - res["methods"][m2]["AUROC"], "dAUROC_CI95": ci(da),
            "dAUPRC": res["methods"][m1]["AUPRC"] - res["methods"][m2]["AUPRC"], "dAUPRC_CI95": ci(dp),
            "mean_abs_dp": float(np.abs(p1 - p2).mean()), "max_abs_dp": float(np.abs(p1 - p2).max()),
            "pearson_r_p": float(np.corrcoef(p1, p2)[0, 1]),
            "n_stays_classification_changes_at_0.5": int(((p1 >= THR) != (p2 >= THR)).sum()),
        }
    sofa, qs = d["sofa"], d["qsofa"]
    res["reference_scores"] = {"SOFA_AUROC": float(roc_auc_score(y, sofa)),
                               "qSOFA_AUROC": float(roc_auc_score(y, qs))}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)

    def f(x, k=3):
        return "n/a" if x is None else (f"{x:.{k}f}" if isinstance(x, float) else str(x))

    def fci(c):
        return "n/a" if c is None else f"[{c[0]:.3f}, {c[1]:.3f}]"
    names = list(M)
    rows = [("AUROC", lambda r: f(r["AUROC"])), ("AUROC 95% CI", lambda r: fci(r["AUROC_CI95"])),
            ("AUPRC", lambda r: f(r["AUPRC"])), ("AUPRC 95% CI", lambda r: fci(r["AUPRC_CI95"])),
            ("Brier", lambda r: f(r["Brier"])), ("ECE (10 bins)", lambda r: f(r["ECE_10bin"])),
            ("Calibration-in-the-large", lambda r: f(r["calibration_in_the_large"], 4)),
            ("Variance ratio sepsis/normal", lambda r: f(r.get("variance_ratio_sepsis_over_normal"), 2)),
            ("AUROC(error | sigma)", lambda r: f(r.get("AUROC_error_given_sigma"))),
            ("  95% CI", lambda r: fci(r.get("AUROC_error_given_sigma_CI95"))),
            ("AUROC(error | -abs(p-0.5)) baseline", lambda r: f(r.get("AUROC_error_baseline_neg_abs_p_minus_0.5"))),
            ("  95% CI", lambda r: fci(r.get("AUROC_error_baseline_CI95"))),
            ("sigma - baseline", lambda r: f(r.get("AUROC_error_sigma_minus_baseline"))),
            ("  95% CI", lambda r: fci(r.get("AUROC_error_sigma_minus_baseline_CI95"))),
            ("Spearman(sigma, p(1-p))", lambda r: f(r.get("spearman_std_vs_p(1-p)"))),
            ("Errors at 0.5 (FN / FP)", lambda r: f"{r['errors_at_0.5']} ({r['FN']} / {r['FP']})"),
            ("G/Y/O/R (alpha 0.10)", lambda r: "/".join(str(v) for v in r["states_GYOR_alpha0.10"].values())
             + ("" if r["states_sigma_used"] else " (sigma=0)"))]
    md = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for lab, fn in rows:
        md.append(f"| {lab} | " + " | ".join(fn(res["methods"][n]) for n in names) + " |")
    md += ["", "| Paired difference | dAUROC [95% CI] | dAUPRC [95% CI] | mean/max abs dp | r | class changes at 0.5 |",
           "|---|---|---|---|---|---|"]
    for k, v in res["paired_differences"].items():
        md.append(f"| {k} | {v['dAUROC']:+.3f} {fci(v['dAUROC_CI95'])} | {v['dAUPRC']:+.3f} {fci(v['dAUPRC_CI95'])} | "
                  f"{v['mean_abs_dp']:.3f} / {v['max_abs_dp']:.3f} | {v['pearson_r_p']:.3f} | {v['n_stays_classification_changes_at_0.5']} |")
    md.append(f"\nReference (same cohort): SOFA AUROC {res['reference_scores']['SOFA_AUROC']:.3f}, "
              f"qSOFA AUROC {res['reference_scores']['qSOFA_AUROC']:.3f}")
    text = "\n".join(md)
    if a.md:
        open(a.md, "w", encoding="utf-8").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
