"""Revision 4, Stage 3: detailed single-run analyses on the OFFICIAL seed-42 run
(runs/seed42_pinned_a; 30 epochs, 100 MC passes). Illustrative only; the
paper's primary results are the 10-seed sweep.

Algorithms are copied from Revision 3 run_authoritative.py
(bootstrap, calibration, fixed-specificity operating points, error detection,
risk-coverage, alpha sweep, onset stratification), applied to the PRIMARY
method (corrected MC Dropout). Class-conditional variance is also reported for
legacy MC (comparison requested by Reviewer 2).

Usage:
  python scripts/seed42_detailed.py --run runs/seed42_pinned_a --out results/seed42_pinned_detailed.json
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, log_loss, roc_auc_score)

SEED = 42


def op(y, pr, thr):
    yp = (pr >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    return (tp / (tp + fn) if tp + fn else 0.0, tn / (tn + fp) if tn + fp else 0.0,
            tp / (tp + fp) if tp + fp else 0.0, tn / (tn + fn) if tn + fn else 0.0)


def bootstrap(y, pr, n=2000, thr=0.5):
    rng = np.random.default_rng(SEED); N = len(y)
    acc = {k: [] for k in ["AUROC", "AUPRC", "Sens", "Spec", "PPV", "NPV"]}
    for _ in range(n):
        idx = rng.integers(0, N, N); yb, pb = y[idx], pr[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        acc["AUROC"].append(roc_auc_score(yb, pb)); acc["AUPRC"].append(average_precision_score(yb, pb))
        s, sp, pp, nv = op(yb, pb, thr)
        acc["Sens"].append(s); acc["Spec"].append(sp); acc["PPV"].append(pp); acc["NPV"].append(nv)
    return {k: {"mean": float(np.mean(v)), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}
            for k, v in acc.items()}


def calib(y, pr, bins=10):
    e = np.linspace(0, 1, bins + 1); rows = []; ece = 0.0; N = len(y)
    for i in range(bins):
        lo, hi = e[i], e[i + 1]
        m = (pr >= lo) & (pr < hi) if i < bins - 1 else (pr >= lo) & (pr <= hi)
        if m.sum() == 0:
            rows.append([float((lo + hi) / 2), None, 0]); continue
        conf = float(pr[m].mean()); ac = float(y[m].mean()); c = int(m.sum())
        rows.append([conf, ac, c]); ece += (c / N) * abs(ac - conf)
    return {"reliability": rows, "ECE": float(ece), "Brier": float(brier_score_loss(y, pr)),
            "NLL": float(log_loss(y, np.clip(pr, 1e-7, 1 - 1e-7))), "cal_in_large": float(pr.mean() - y.mean())}


def operating_points(y, p):
    out = {}
    for target in [0.80, 0.90, 0.95]:
        cands = []
        for t in np.unique(p):
            yp = (p >= t).astype(int)
            tn = ((yp == 0) & (y == 0)).sum(); fp = ((yp == 1) & (y == 0)).sum()
            tp = ((yp == 1) & (y == 1)).sum(); fn = ((yp == 0) & (y == 1)).sum()
            spec = tn / (tn + fp) if tn + fp else 0; sens = tp / (tp + fn) if tp + fn else 0
            ppv = tp / (tp + fp) if tp + fp else 0
            if spec >= target:
                cands.append((float(spec), float(sens), float(ppv), float(t), int(tp + fp), int(tp), int(fp)))
        if cands:
            cands.sort(key=lambda c: (-c[1], c[0]))
            s_, sn_, pp_, thr_, al_, tp_, fp_ = cands[0]
            out[str(target)] = dict(achieved_spec=s_, sens=sn_, ppv=pp_, thr=thr_, alerts=al_, tp=tp_, fp=fp_,
                                    alerts_per_100=100 * al_ / len(y), false_alert_fraction=fp_ / al_ if al_ else None)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; refusing to overwrite")
    d = np.load(os.path.join(a.run, "predictions.npz"))
    cfg = json.load(open(os.path.join(a.run, "config.json"))); chk = json.load(open(os.path.join(a.run, "run_checks.json")))
    y = d["yte"].astype(int); p = d["p_mc_corrected"].astype(np.float64); std = d["std_corrected"].astype(np.float64); var = std ** 2
    sl = d["std_legacy"].astype(np.float64); vl = sl ** 2
    R = {"run": a.run, "settings": cfg["settings"], "method": "corrected MC Dropout (bn_mode='inference')",
         "test_set": {"n_stays": int(len(y)), "n_positive": int(y.sum()), "prevalence": float(y.mean())}}
    R["ablation"] = {
        "BiLSTM_only": {"AUROC": float(roc_auc_score(y, d["p_lstm"])), "AUPRC": float(average_precision_score(y, d["p_lstm"])), "params": chk["params_bilstm_only"]},
        "CNN_BiLSTM_deterministic": {"AUROC": float(roc_auc_score(y, d["p_det"])), "AUPRC": float(average_precision_score(y, d["p_det"])), "params": chk["params_cnn_bilstm"]},
        "CNN_BiLSTM_legacy_MC": {"AUROC": float(roc_auc_score(y, d["p_mc_legacy"])), "AUPRC": float(average_precision_score(y, d["p_mc_legacy"])), "params": chk["params_cnn_bilstm"]},
        "CNN_BiLSTM_corrected_MC": {"AUROC": float(roc_auc_score(y, p)), "AUPRC": float(average_precision_score(y, p)), "params": chk["params_cnn_bilstm"]},
    }
    R["bootstrap_ci"] = bootstrap(y, p)
    R["calibration"] = calib(y, p)
    R["operating_points"] = operating_points(y, p)
    R["variance_by_class"] = {
        "corrected": {"sepsis": float(var[y == 1].mean()), "normal": float(var[y == 0].mean()), "ratio": float(var[y == 1].mean() / var[y == 0].mean()),
                      "mean_std_sepsis": float(std[y == 1].mean()), "mean_std_normal": float(std[y == 0].mean())},
        "legacy": {"sepsis": float(vl[y == 1].mean()), "normal": float(vl[y == 0].mean()), "ratio": float(vl[y == 1].mean() / vl[y == 0].mean())}}
    yp = (p >= 0.5).astype(int); correct = yp == y; err = (~correct).astype(int)
    order = np.argsort(std); rc = []
    for c in np.arange(0.10, 1.001, 0.05):
        k = max(1, int(np.round(c * len(std)))); idx = order[:k]
        rc.append({"coverage": float(c), "accuracy": float((yp[idx] == y[idx]).mean()), "n_retained": int(k)})
    base = -np.abs(p - 0.5)
    R["error_detection"] = {"n_correct": int(correct.sum()), "n_wrong": int(err.sum()),
                            "FN": int(((yp == 0) & (y == 1)).sum()), "FP": int(((yp == 1) & (y == 0)).sum()),
                            "mean_variance_correct": float(var[correct].mean()), "mean_variance_wrong": float(var[~correct].mean()),
                            "wrong_over_correct_ratio": float(var[~correct].mean() / var[correct].mean()),
                            "AUROC_error_from_variance": float(roc_auc_score(err, var)),
                            "AUROC_error_baseline_neg_abs_p_minus_0.5": float(roc_auc_score(err, base)),
                            "risk_coverage": rc}
    alpha = []
    for al in [0.0, 0.05, 0.10, 0.20, 0.50, 1.00]:
        pa = np.clip(p + al * std, 0, 1)
        alpha.append({"alpha": al, "AUROC": float(roc_auc_score(y, pa)), "AUPRC": float(average_precision_score(y, pa)),
                      "n_green": int((pa < 0.10).sum()), "n_yellow": int(((pa >= 0.10) & (pa < 0.30)).sum()),
                      "n_orange": int(((pa >= 0.30) & (pa < 0.60)).sum()), "n_red": int((pa >= 0.60).sum())})
    R["alpha_sensitivity"] = alpha
    on = d["onset_te"][y == 1]; pp = p[y == 1]; med = float(np.median(on)); e_, l_ = on <= med, on > med
    R["onset_stratification"] = {"median_onset_timesteps": med, "n_early": int(e_.sum()), "n_late": int(l_.sum()),
                                 "mean_p_early_onset": float(pp[e_].mean()), "mean_p_late_onset": float(pp[l_].mean()),
                                 "detected_05_early": int((pp[e_] >= 0.5).sum()), "detected_05_late": int((pp[l_] >= 0.5).sum())}
    R["sofa_qsofa_same_cohort"] = {"SOFA": {"AUROC": float(roc_auc_score(y, d["sofa"])), "AUPRC": float(average_precision_score(y, d["sofa"]))},
                                   "qSOFA": {"AUROC": float(roc_auc_score(y, d["qsofa"])), "AUPRC": float(average_precision_score(y, d["qsofa"]))}}
    json.dump(R, open(a.out, "w"), indent=2)
    print(json.dumps({k: R[k] for k in ("ablation", "operating_points", "variance_by_class", "onset_stratification", "sofa_qsofa_same_cohort")}, indent=1))
    print("bootstrap:", {k: (round(v["lo"], 3), round(v["hi"], 3)) for k, v in R["bootstrap_ci"].items()})
    print("calibration:", {k: round(v, 4) for k, v in R["calibration"].items() if k != "reliability"})
    print("error:", {k: v for k, v in R["error_detection"].items() if k != "risk_coverage"})
    print("alpha:", [(x["alpha"], round(x["AUROC"], 3), round(x["AUPRC"], 3), x["n_green"], x["n_yellow"], x["n_orange"], x["n_red"]) for x in alpha])


if __name__ == "__main__":
    main()
