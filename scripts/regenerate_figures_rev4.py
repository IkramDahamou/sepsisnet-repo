"""Revision 4: regenerate every output-dependent figure from saved files.

Inputs (read-only):
  runs/seed42_pinned_a/predictions.npz        official seed-42 run (illustrative)
  results/seed42_pinned_detailed.json         seed-42 analyses (scripts/seed42_detailed.py)
  results/sweep_summary.json, runs/sweep_a/   10-seed sweep (primary results)
Outputs: figures/fig{4a,4b,5,6,7,8,9}_*.{png,pdf}
(fig2_cnn_lstm and fig3_petri_net are unchanged copies of the Revision 3 files.)

Usage:  python scripts/regenerate_figures_rev4.py
"""
import json
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

REV4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(REV4, "figures")
os.makedirs(FIG, exist_ok=True)
mpl.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.titlesize": 12,
                     "axes.labelsize": 11, "legend.frameon": False, "legend.fontsize": 10, "figure.dpi": 130,
                     "savefig.dpi": 220, "savefig.bbox": "tight"})
BLUE, ORANGE, GRAY, GREEN, RED, PURPLE = "#2b6ba7", "#e07b3f", "#8a8a8a", "#3f8f66", "#c14a4a", "#7a5aa6"

d = np.load(os.path.join(REV4, "runs", "seed42_pinned_a", "predictions.npz"))
R = json.load(open(os.path.join(REV4, "results", "seed42_pinned_detailed.json")))
S = json.load(open(os.path.join(REV4, "results", "sweep_summary.json")))
y = d["yte"].astype(int); p = d["p_mc_corrected"].astype(float); std = d["std_corrected"].astype(float); var = std ** 2
pl = d["p_mc_legacy"].astype(float); sl = d["std_legacy"].astype(float)
TAG = "seed 42, single illustrative run"


def save(name):
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(); print("wrote", name)


# ---- ROC ----
fpr, tpr, _ = roc_curve(y, p); au = roc_auc_score(y, p)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1.2, label="Random")
ax.fill_between(fpr, tpr, alpha=0.20, color=BLUE)
ax.plot(fpr, tpr, "-", color=BLUE, lw=2.2, label=f"Corrected MC Dropout  AUROC = {au:.3f}")
for target, mkr in zip(["0.8", "0.9", "0.95"], ["o", "s", "D"]):
    o = R["operating_points"][target]
    ax.plot(1 - o["achieved_spec"], o["sens"], mkr, ms=9, color=ORANGE, markeredgecolor="#7a3f18", markeredgewidth=1.2,
            label=f"Target spec {target}: sens={o['sens']:.2f}, spec={o['achieved_spec']:.2f}")
ax.set_xlabel("False Positive Rate (1 − Specificity)"); ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.set_title(f"ROC curve ({TAG})", fontsize=11)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal"); ax.legend(loc="lower right", fontsize=8.5)
save("fig4a_roc_curve")

# ---- PR ----
prec, rec, _ = precision_recall_curve(y, p); ap = average_precision_score(y, p); prev = float(y.mean())
fig, ax = plt.subplots(figsize=(6, 5))
ax.hlines(prev, 0, 1, colors=GRAY, linestyles="--", lw=1.2, label=f"Random baseline (prevalence = {prev:.3f})")
ax.fill_between(rec, prec, alpha=0.20, color=ORANGE, step="post")
ax.step(rec, prec, "-", where="post", color=ORANGE, lw=2.2, label=f"Corrected MC Dropout  AUPRC = {ap:.3f}")
ax.set_xlabel("Recall (Sensitivity)"); ax.set_ylabel("Precision (PPV)")
ax.set_title(f"Precision-Recall curve ({TAG})", fontsize=11)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.05); ax.legend(loc="lower left", fontsize=9)
save("fig4b_pr_curve")

# ---- Uncertainty by class: corrected vs legacy ----
vb = R["variance_by_class"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, (pp, ss, lab, key) in zip(axes, [(p, std, "Corrected MC (BN inference mode; primary)", "corrected"),
                                         (pl, sl, "Legacy MC (BN batch statistics; Revision 3)", "legacy")]):
    for m, c, n in ((y == 0, BLUE, "Non-sepsis"), (y == 1, ORANGE, "Sepsis")):
        ax.scatter(pp[m], ss[m], s=48, alpha=0.75, color=c, edgecolor="white", linewidth=0.6, label=f"{n} ({int(m.sum())})")
    ax.set_title(f"{lab}\nvariance ratio sepsis/normal = {vb[key]['ratio']:.2f}", fontsize=10.5)
    ax.set_xlabel("Mean predicted P(sepsis) over 100 MC passes"); ax.set_xlim(-0.02, 1.02)
axes[0].set_ylabel("Predictive standard deviation σ"); axes[0].legend(loc="upper right", fontsize=9)
plt.suptitle(f"MC Dropout predictive σ by class, same weights ({TAG})", fontsize=12, y=1.02)
plt.tight_layout(); save("fig5_uncertainty")

# ---- Error detection: sigma vs baseline ----
E = R["error_detection"]; yp = (p >= 0.5).astype(int); correct = yp == y
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]; data = [var[correct], var[~correct]]
bp = ax.boxplot(data, tick_labels=[f"Correct\n(n={int(correct.sum())})", f"Wrong\n(n={int((~correct).sum())})"], widths=0.55,
                patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=1.6))
for patch, c in zip(bp["boxes"], [GREEN, RED]):
    patch.set_facecolor(c); patch.set_alpha(0.55)
rng = np.random.default_rng(1)
for i, (arr, c) in enumerate(zip(data, [GREEN, RED])):
    ax.scatter(np.full(len(arr), i + 1.0) + rng.normal(0, 0.05, len(arr)), arr, s=24, alpha=0.65, color=c, edgecolor="white", linewidth=0.4)
ax.set_ylabel("Predictive variance σ²")
ax.set_title(f"AUROC(error | σ²) = {E['AUROC_error_from_variance']:.3f};  baseline AUROC(error | −|p−0.5|) = "
             f"{E['AUROC_error_baseline_neg_abs_p_minus_0.5']:.3f}", fontsize=10)
ax = axes[1]
order_s = np.argsort(std); order_b = np.argsort(np.abs(p - 0.5))[::-1]
covs = np.arange(0.10, 1.001, 0.05)
def curve(order):
    return [float((yp[order[:max(1, int(np.round(c * len(y))))]] == y[order[:max(1, int(np.round(c * len(y))))]]).mean()) for c in covs]
ax.plot(covs, curve(order_s), "-o", ms=5, color=BLUE, lw=1.8, label="Retain lowest-σ stays")
ax.plot(covs, curve(order_b), "-s", ms=5, color=PURPLE, lw=1.6, label="Retain stays farthest from p = 0.5 (baseline)")
ax.hlines(float(correct.mean()), 0, 1, colors=GRAY, linestyles="--", lw=1.0, label=f"Accuracy at full coverage = {float(correct.mean()):.3f}")
ax.set_xlabel("Coverage (fraction of stays retained)"); ax.set_ylabel("Accuracy on retained stays")
ax.set_title("Risk-coverage: σ versus the −|p−0.5| baseline", fontsize=10.5); ax.set_xlim(0, 1.02); ax.legend(loc="lower left", fontsize=8.5)
plt.suptitle(f"Error detection with corrected MC Dropout σ ({TAG})", fontsize=12, y=1.02)
plt.tight_layout(); save("fig6_error_detection")

# ---- alpha sensitivity ----
A = R["alpha_sensitivity"]; al = [r["alpha"] for r in A]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
ax.plot(al, [r["AUROC"] for r in A], "-o", ms=7, color=BLUE, lw=2, label="AUROC")
ax.plot(al, [r["AUPRC"] for r in A], "-s", ms=7, color=ORANGE, lw=2, label="AUPRC")
ax.axvline(0.10, ls=":", color="#444", lw=1.2); ax.text(0.11, 0.95, "default α = 0.10", fontsize=8.5, color="#444", va="top")
ax.set_xlabel("Uncertainty coefficient α in p + α·σ"); ax.set_ylabel("Discrimination (test split)")
ax.set_ylim(0.5, 1.0); ax.set_xlim(-0.03, 1.05); ax.legend(loc="lower right")
ax = axes[1]; xs = np.arange(len(al))
g, yy, o, r_ = (np.array([r[k] for r in A]) for k in ("n_green", "n_yellow", "n_orange", "n_red"))
ax.bar(xs, g, color=GREEN, edgecolor="white", label="GREEN"); ax.bar(xs, yy, bottom=g, color="#e6c945", edgecolor="white", label="YELLOW")
ax.bar(xs, o, bottom=g + yy, color=ORANGE, edgecolor="white", label="ORANGE"); ax.bar(xs, r_, bottom=g + yy + o, color=RED, edgecolor="white", label="RED")
ax.set_xticks(xs); ax.set_xticklabels([f"α={v:g}" for v in al]); ax.set_ylabel(f"Number of test stays (out of {len(y)})")
ax.set_ylim(0, 92); ax.legend(loc="upper center", fontsize=8.5, ncol=4)
plt.suptitle(f"Sensitivity of the escalation score to α ({TAG})", fontsize=12, y=1.02)
plt.tight_layout(); save("fig7_alpha_sensitivity")

# ---- calibration + bootstrap ----
rel = R["calibration"]["reliability"]; ci = R["bootstrap_ci"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]; ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1.2, label="Perfectly calibrated")
pts = [(r[0], r[1], r[2]) for r in rel if r[1] is not None]
ax.scatter([a for a, _, _ in pts], [b for _, b, _ in pts], s=[max(30, c * 60) for _, _, c in pts], color=BLUE, alpha=0.72, edgecolor="white", zorder=3)
ax.plot([a for a, _, _ in pts], [b for _, b, _ in pts], "-", color=BLUE, alpha=0.35)
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency of sepsis")
ax.set_title(f"Reliability  |  ECE = {R['calibration']['ECE']:.3f}  |  Brier = {R['calibration']['Brier']:.3f}", fontsize=10.5)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal"); ax.legend(loc="upper left", fontsize=9)
ax = axes[1]; mo = ["AUROC", "AUPRC", "Sens", "Spec", "PPV", "NPV"]
mn = np.array([ci[m]["mean"] for m in mo]); lo = np.array([ci[m]["lo"] for m in mo]); hi = np.array([ci[m]["hi"] for m in mo])
ax.errorbar(mn, np.arange(6), xerr=[mn - lo, hi - mn], fmt="o", ms=9, color=BLUE, ecolor=GRAY, elinewidth=2.2, capsize=5)
for i in range(6):
    ax.text(min(hi[i] + 0.02, 1.0), i, f"{mn[i]:.2f} [{lo[i]:.2f}, {hi[i]:.2f}]", va="center", fontsize=9.5)
ax.set_yticks(range(6)); ax.set_yticklabels(mo); ax.set_xlabel("Value (95% bootstrap CI, n = 2000; threshold 0.5)")
ax.set_xlim(0, 1.3); ax.invert_yaxis()
plt.suptitle(f"Calibration and bootstrap CIs, corrected MC Dropout ({TAG})", fontsize=12, y=1.02)
plt.tight_layout(); save("fig8_calibration_ci")

# ---- NEW fig9: 10-seed per-seed results (primary) ----
seeds = [str(s) for s in S["seeds_planned"]]
M = {s: json.load(open(os.path.join(REV4, "runs", "sweep_a", f"seed_{s}", "metrics.json")))["methods"] for s in seeds}
meth = [("bilstm_only", "BiLSTM-only"), ("deterministic", "CNN-BiLSTM\ndeterministic"), ("legacy_mc", "Legacy MC"), ("corrected_mc", "Corrected MC")]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, k, lab in ((axes[0], "AUROC", "AUROC"), (axes[1], "AUPRC", "AUPRC")):
    for s in seeds:
        ax.plot(range(4), [M[s][m][k] for m, _ in meth], "-", color=GRAY, alpha=0.4, lw=1)
    for i, (m, _) in enumerate(meth):
        v = [M[s][m][k] for s in seeds]; ax.scatter([i] * len(v), v, s=26, color=BLUE, zorder=3)
        ax.errorbar(i + 0.18, np.mean(v), yerr=np.std(v, ddof=1), fmt="D", color=ORANGE, capsize=4, zorder=4)
    ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in meth], fontsize=9); ax.set_ylabel(lab); ax.set_title(f"{lab} per seed (lines join the same seed)", fontsize=10.5)
ax = axes[2]
for i, (m, lab) in enumerate((("legacy_mc", "Legacy MC"), ("corrected_mc", "Corrected MC"))):
    dsig = [M[s][m]["AUROC_error_given_sigma"] for s in seeds]; dbase = [M[s][m]["AUROC_error_baseline_neg_abs_p_minus_0.5"] for s in seeds]
    for a_, b_ in zip(dsig, dbase):
        ax.plot([2 * i, 2 * i + 1], [a_, b_], "-", color=GRAY, alpha=0.4, lw=1)
    ax.scatter([2 * i] * 10, dsig, s=26, color=BLUE, zorder=3); ax.scatter([2 * i + 1] * 10, dbase, s=26, color=PURPLE, zorder=3)
ax.axhline(0.5, ls=":", color=GRAY)
ax.set_xticks(range(4)); ax.set_xticklabels(["Legacy\nσ", "Legacy\nbaseline", "Corrected\nσ", "Corrected\nbaseline"], fontsize=9)
ax.set_ylabel("AUROC(error)"); ax.set_title("Error detection: σ vs −|p−0.5| baseline", fontsize=10.5)
plt.suptitle("10-seed sweep (run a; 15 epochs, 50 MC passes); orange: mean ± SD", fontsize=12, y=1.02)
plt.tight_layout(); save("fig9_multiseed")
