"""
Regenerate every figure in the manuscript from real prediction arrays
saved by run_authoritative.py. Run:  python3 regenerate_figures.py
Produces PNG + PDF for each figure into ./figures/.
"""
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score

# ---- Style ---------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 130,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})
BLUE   = "#2b6ba7"
ORANGE = "#e07b3f"
GRAY   = "#8a8a8a"
LIGHT  = "#dfe8f2"
LIGHTO = "#f7e2d3"
GREEN  = "#3f8f66"
RED    = "#c14a4a"

# ---- Load data -----------------------------------------------------------
d = np.load("real_pred_full.npz")
yte    = d["yte"];       onset_te = d["onset_te"]
p_det  = d["p_det"];     p_mc     = d["p_mc"]
std    = d["std"];       var      = d["var"]
p_lstm = d["p_lstm"];    sofa     = d["sofa"];  qsofa = d["qsofa"]
with open("revision3_results.json") as f:
    R = json.load(f)

au_full = roc_auc_score(yte, p_mc)
print(f"Loaded: {len(yte)} stays, {int(yte.sum())} positive, AUROC={au_full:.4f}")

# ==========================================================================
# Figure 1 (fig2_cnn_lstm) — CORRECT architecture diagram
# 3 CNN blocks (64/128/256) + 2 BiLSTM (128/64) + 2 Dense (256/128), dropout 0.4
# ==========================================================================
def draw_architecture():
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

    def box(x, y, w, h, label, sub, color, edge="#1a1a1a"):
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.10,rounding_size=0.6",
                           lw=1.4, facecolor=color, edgecolor=edge)
        ax.add_patch(p)
        ax.text(x + w/2, y + h/2 + 1.0, label,
                ha="center", va="center", fontsize=10.5, weight="bold", color="#111")
        ax.text(x + w/2, y + h/2 - 1.8, sub,
                ha="center", va="center", fontsize=8.5, color="#333")

    def arrow(x0, x1, y=23):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#333"))

    def group_label(x_center, y_top, text, color):
        ax.text(x_center, y_top, text, ha="center", va="bottom",
                fontsize=10.5, weight="bold", color=color)

    # Positions
    y0, h = 16, 12
    # Group tags
    group_label(6,   34, "Input",           "#1f9d55")
    group_label(30,  34, "1D CNN encoder",  "#2b6ba7")
    group_label(56,  34, "BiLSTM encoder",  "#7b3fa9")
    group_label(78,  34, "Dense head",      "#c07f00")
    group_label(94,  34, "MC Dropout",      "#a8390a")

    # Input
    box(2,  y0, 8, h, "Input", "T=240, F=20", "#c6e9d5")
    arrow(10, 15)
    # CNN 1
    box(15, y0, 8, h, "Conv1D", "64f, k=3\nBN + Drop 0.4", "#dfe8f2")
    arrow(23, 28)
    # CNN 2
    box(28, y0, 8, h, "Conv1D", "128f, k=3\nBN + Drop 0.4", "#c9dcef")
    arrow(36, 41)
    # CNN 3
    box(41, y0, 8, h, "Conv1D", "256f, k=3\nBN + Drop 0.4", "#b6d1eb")
    arrow(49, 54)
    # BiLSTM 1
    box(54, y0, 8, h, "BiLSTM", "128 units\nreturn_seq\nDrop 0.4", "#e3d5f2")
    arrow(62, 65)
    # BiLSTM 2
    box(65, y0, 8, h, "BiLSTM", "64 units\nDrop 0.4", "#d3bee9")
    arrow(73, 76)
    # Dense 1
    box(76, y0, 7, h, "Dense", "256 units\nDrop 0.4", "#faedcc")
    arrow(83, 85)
    # Dense 2
    box(85, y0, 7, h, "Dense", "128 units\nDrop 0.4", "#f4dfa8")
    arrow(92, 94)
    # Output / MC
    box(94, y0, 5, h, "Sigmoid", "P(sepsis)\n(training=T\nat inference)", "#f5cfc0")

    # Shape trace below
    shapes = [(6, "240×20"), (19, "240×64"), (32, "240×128"),
              (45, "240×256"), (58, "240×256"), (69, "128"),
              (79.5, "256"),   (88.5, "128"),  (96.5, "1")]
    for x, s in shapes:
        ax.text(x, 12.5, s, ha="center", va="top",
                fontsize=8.5, color="#555", style="italic")

    # Parameter total
    ax.text(50, 4, "Total trainable parameters: 753,601   |   "
                   "MC Dropout: 100 stochastic forward passes with training=True",
            ha="center", va="center", fontsize=10.5, weight="bold", color="#1a1a1a")

    plt.title("SepsisNet — CNN-BiLSTM architecture with MC Dropout",
              fontsize=13.5, weight="bold", pad=14)
    for ext in ("png", "pdf"):
        plt.savefig(f"figures/fig2_cnn_lstm.{ext}")
    plt.close()

draw_architecture()
print("✓ fig2_cnn_lstm")

# ==========================================================================
# Figure 2 (fig4a) — ROC curve (real predictions)
# ==========================================================================
fpr, tpr, _ = roc_curve(yte, p_mc)
au = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1.2, label="Random")
ax.fill_between(fpr, tpr, alpha=0.20, color=BLUE)
ax.plot(fpr, tpr, "-", color=BLUE, lw=2.2, label=f"SepsisNet  AUROC = {au:.3f}")
# Mark operating points
for target, mkr in zip(["0.8", "0.9", "0.95"], ["o", "s", "D"]):
    op = R["operating_points"][target]
    thr = op["thr"]; ypred = (p_mc >= thr).astype(int)
    tp = int(((ypred == 1) & (yte == 1)).sum())
    fp = int(((ypred == 1) & (yte == 0)).sum())
    tn = int(((ypred == 0) & (yte == 0)).sum())
    fn = int(((ypred == 0) & (yte == 1)).sum())
    sens = tp / (tp + fn); spec = tn / (tn + fp)
    ax.plot(1 - spec, sens, mkr, ms=9, color=ORANGE,
            markeredgecolor="#7a3f18", markeredgewidth=1.2,
            label=f"Target spec {target}: sens={sens:.2f}, spec={spec:.2f}")
ax.set_xlabel("False Positive Rate (1 − Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.set_title("ROC curve — CNN-BiLSTM + MC Dropout (synthetic test split)")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
ax.legend(loc="lower right", fontsize=8.5)
for ext in ("png", "pdf"):
    plt.savefig(f"figures/fig4a_roc_curve.{ext}")
plt.close()
print(f"✓ fig4a_roc_curve  (AUROC={au:.4f})")

# ==========================================================================
# Figure 3 (fig4b) — Precision-Recall
# ==========================================================================
prec, rec, _ = precision_recall_curve(yte, p_mc)
prevalence = float(yte.mean())
from sklearn.metrics import average_precision_score
ap = average_precision_score(yte, p_mc)

fig, ax = plt.subplots(figsize=(6, 5))
ax.hlines(prevalence, 0, 1, colors=GRAY, linestyles="--", lw=1.2,
          label=f"Random baseline (prevalence = {prevalence:.3f})")
ax.fill_between(rec, prec, alpha=0.20, color=ORANGE, step="post")
ax.step(rec, prec, "-", where="post", color=ORANGE, lw=2.2,
        label=f"SepsisNet  AUPRC = {ap:.3f}")
ax.set_xlabel("Recall (Sensitivity)")
ax.set_ylabel("Precision (PPV)")
ax.set_title("Precision-Recall curve — CNN-BiLSTM + MC Dropout")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.05)
ax.legend(loc="lower left", fontsize=9)
for ext in ("png", "pdf"):
    plt.savefig(f"figures/fig4b_pr_curve.{ext}")
plt.close()
print(f"✓ fig4b_pr_curve  (AUPRC={ap:.4f})")

# ==========================================================================
# Figure 4 (fig5) — Uncertainty scatter (real MC variance)
# ==========================================================================
ratio = R["uncertainty_variance"]["variance_ratio"]
sv = R["uncertainty_variance"]["sepsis_variance"]
nv = R["uncertainty_variance"]["normal_variance"]

fig, ax = plt.subplots(figsize=(7.5, 5.2))
mask_pos = yte == 1; mask_neg = yte == 0
ax.scatter(p_mc[mask_neg], std[mask_neg], s=48, alpha=0.72,
           color=BLUE,   edgecolor="white", linewidth=0.6,
           label=f"Non-sepsis ({int(mask_neg.sum())})")
ax.scatter(p_mc[mask_pos], std[mask_pos], s=64, alpha=0.85,
           color=ORANGE, edgecolor="white", linewidth=0.6,
           label=f"Sepsis ({int(mask_pos.sum())})")
ax.axhline(np.sqrt(nv), ls=":", color=BLUE, alpha=0.85,
           label=f"Mean σ non-sepsis = {np.sqrt(nv):.3f}")
ax.axhline(np.sqrt(sv), ls=":", color=ORANGE, alpha=0.85,
           label=f"Mean σ sepsis      = {np.sqrt(sv):.3f}")
ax.set_xlabel("Mean predicted P(sepsis) over 100 MC Dropout passes")
ax.set_ylabel("Predictive standard deviation σ (uncertainty)")
ax.set_title(f"MC Dropout uncertainty by class  —  variance ratio = {ratio:.2f}×")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, max(std.max()*1.08, 0.35))
ax.legend(loc="upper right", fontsize=9)
for ext in ("png", "pdf"):
    plt.savefig(f"figures/fig5_uncertainty.{ext}")
plt.close()
print(f"✓ fig5_uncertainty  (ratio={ratio:.4f})")

# ==========================================================================
# Figure 5 (fig6) — NEW: Uncertainty error-detection + risk-coverage
# ==========================================================================
yp05 = (p_mc >= 0.5).astype(int)
correct = (yp05 == yte)
ec = R["uncertainty_error_detection"]
rc = ec["risk_coverage"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

# Left: box plot of variance by correctness
ax = axes[0]
data = [var[correct], var[~correct]]
bp = ax.boxplot(data, labels=[f"Correct\n(n={int(correct.sum())})",
                              f"Wrong\n(n={int((~correct).sum())})"],
                widths=0.55, patch_artist=True,
                medianprops=dict(color="black", lw=1.6),
                boxprops=dict(edgecolor="black", lw=1.1),
                whiskerprops=dict(color="black", lw=1.0),
                capprops=dict(color="black", lw=1.0),
                showfliers=False)
for patch, c in zip(bp["boxes"], [GREEN, RED]):
    patch.set_facecolor(c); patch.set_alpha(0.55)
# Overlay scatter
rng = np.random.default_rng(1)
for i, (arr, c) in enumerate(zip(data, [GREEN, RED])):
    x = np.full_like(arr, i+1, dtype=float) + rng.normal(0, 0.05, size=len(arr))
    ax.scatter(x, arr, s=24, alpha=0.65, color=c, edgecolor="white", linewidth=0.4)
ax.set_ylabel("MC Dropout predictive variance σ²")
ax.set_title(f"Uncertainty is higher on wrong predictions\n"
             f"AUROC(error | variance) = {ec['AUROC_error_from_variance']:.3f} "
             f"(ratio = {ec['wrong_over_correct_ratio']:.2f}×)")

# Right: risk-coverage curve
ax = axes[1]
covs = [r["coverage"] for r in rc]
accs = [r["accuracy"] for r in rc]
ax.fill_between(covs, [prevalence]*len(covs), accs, alpha=0.18, color=BLUE)
ax.plot(covs, accs, "-o", ms=5, color=BLUE, lw=1.8,
        label="Retain the k least-uncertain stays")
ax.hlines(float(correct.mean()), 0, 1, colors=GRAY, linestyles="--", lw=1.0,
          label=f"Accuracy at full coverage = {float(correct.mean()):.3f}")
ax.set_xlabel("Coverage (fraction of stays retained)")
ax.set_ylabel("Accuracy on retained stays")
ax.set_title("Risk-coverage: accuracy improves as coverage shrinks")
ax.set_xlim(0, 1.02); ax.set_ylim(0.80, 1.02)
ax.legend(loc="lower left", fontsize=9)

plt.suptitle("Error-detection analysis of MC Dropout uncertainty",
             fontsize=13, weight="bold", y=1.02)
plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(f"figures/fig6_error_detection.{ext}")
plt.close()
print("✓ fig6_error_detection")

# ==========================================================================
# Figure 6 (fig7) — NEW: Coefficient-alpha sensitivity + escalation states
# ==========================================================================
alphas    = [row["alpha"]   for row in R["alpha_sensitivity"]]
au_arr    = [row["AUROC"]   for row in R["alpha_sensitivity"]]
ap_arr    = [row["AUPRC"]   for row in R["alpha_sensitivity"]]
ng        = [row["n_green"] for row in R["alpha_sensitivity"]]
ny        = [row["n_yellow"] for row in R["alpha_sensitivity"]]
no        = [row["n_orange"] for row in R["alpha_sensitivity"]]
nr        = [row["n_red"]    for row in R["alpha_sensitivity"]]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

# Left: AUROC / AUPRC vs alpha
ax = axes[0]
ax.plot(alphas, au_arr, "-o", ms=7, color=BLUE,   lw=2, label="AUROC")
ax.plot(alphas, ap_arr, "-s", ms=7, color=ORANGE, lw=2, label="AUPRC")
ax.axvline(0.10, ls=":", color="#444", lw=1.2, alpha=0.85)
ax.text(0.11, 0.94, "released\ndefault α = 0.10",
        fontsize=8.5, color="#444", ha="left", va="top")
ax.set_xlabel("Uncertainty coefficient α in p + α·σ")
ax.set_ylabel("Discrimination on synthetic test split")
ax.set_title("Sensitivity of the uncertainty coefficient")
ax.set_ylim(0.5, 1.0); ax.set_xlim(-0.03, 1.05)
ax.legend(loc="lower right", fontsize=10)

# Right: escalation state distribution (stacked bar per alpha)
ax = axes[1]
xs = np.arange(len(alphas))
p_g = np.array(ng); p_y = np.array(ny); p_o = np.array(no); p_r = np.array(nr)
b1 = ax.bar(xs, p_g, color=GREEN,  edgecolor="white", label="GREEN")
b2 = ax.bar(xs, p_y, bottom=p_g, color="#e6c945", edgecolor="white", label="YELLOW")
b3 = ax.bar(xs, p_o, bottom=p_g+p_y, color=ORANGE, edgecolor="white", label="ORANGE")
b4 = ax.bar(xs, p_r, bottom=p_g+p_y+p_o, color=RED, edgecolor="white", label="RED")
ax.set_xticks(xs); ax.set_xticklabels([f"α={a:g}" for a in alphas])
ax.set_xlabel("Uncertainty coefficient α in p + α·σ")
ax.set_ylabel("Number of test stays (out of 76)")
ax.set_title("Escalation-state distribution by α (Green/Yellow/Orange/Red)")
ax.set_ylim(0, 92)
ax.legend(loc="upper center", fontsize=8.5, ncol=4, bbox_to_anchor=(0.5, 1.00))

plt.suptitle("Uncertainty-adjusted escalation policy — sensitivity to α",
             fontsize=13, weight="bold", y=1.02)
plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(f"figures/fig7_alpha_sensitivity.{ext}")
plt.close()
print("✓ fig7_alpha_sensitivity")

# ==========================================================================
# Figure 7 (fig8) — Calibration reliability + bootstrap CIs
# ==========================================================================
rel = R["calibration"]["reliability"]
ece = R["calibration"]["ECE"]
brier = R["calibration"]["Brier"]
ci = R["bootstrap_ci"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

# Left: reliability diagram
ax = axes[0]
ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1.2, label="Perfectly calibrated")
xs, ys, sizes = [], [], []
for r in rel:
    if r[1] is not None:
        xs.append(r[0]); ys.append(r[1]); sizes.append(max(30, r[2] * 60))
ax.scatter(xs, ys, s=sizes, color=BLUE, alpha=0.72,
           edgecolor="white", linewidth=1.2, zorder=3)
ax.plot(xs, ys, "-", color=BLUE, alpha=0.35, zorder=2)
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed frequency of sepsis")
ax.set_title(f"Reliability  |  ECE = {ece:.3f}  |  Brier = {brier:.3f}")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
ax.legend(loc="upper left", fontsize=9)

# Right: bootstrap CIs
ax = axes[1]
metrics_order = ["AUROC", "AUPRC", "Sens", "Spec", "PPV", "NPV"]
means = [ci[m]["mean"] for m in metrics_order]
lo    = [ci[m]["lo"]   for m in metrics_order]
hi    = [ci[m]["hi"]   for m in metrics_order]
xs = np.arange(len(metrics_order))
ax.errorbar(means, xs, xerr=[np.array(means)-np.array(lo), np.array(hi)-np.array(means)],
            fmt="o", ms=9, color=BLUE, ecolor=GRAY, elinewidth=2.2, capsize=5)
for i, m in enumerate(metrics_order):
    ax.text(hi[i] + 0.02, i, f"{means[i]:.2f} [{lo[i]:.2f}, {hi[i]:.2f}]",
            va="center", fontsize=9.5, color="#333")
ax.set_yticks(xs); ax.set_yticklabels(metrics_order)
ax.set_xlabel("Value (95 % bootstrap CI, n = 2000)")
ax.set_title("Bootstrap 95 % confidence intervals")
ax.set_xlim(0, 1.15); ax.invert_yaxis()

plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(f"figures/fig8_calibration_ci.{ext}")
plt.close()
print(f"✓ fig8_calibration_ci  (ECE={ece:.4f})")

print("\nAll figures written to ./figures/")
