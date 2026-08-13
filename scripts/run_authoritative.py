"""
SepsisNet — Authoritative single-seed run for Revision 3.
Reproduces revision_results_real.json and saves all prediction arrays
needed to regenerate every figure and run the new analyses:
  - Uncertainty error-detection (R2#4)
  - Coefficient-alpha sensitivity of p+alpha*sigma (R2#4)
  - End-to-end escalation-policy simulation (R2#3, ex-post on stay scores)
Seed = 42.
"""
import os, sys, json, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sepsisnet", "src"))

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from data_loader import MIMICDataGenerator, MIMICPreprocessor
from models import SepsisNetModel
from uncertainty import MonteCarloDropout
from mdp_escalation import AlertEscalationMDP, AlertState, AlertAction
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, brier_score_loss, log_loss,
                             roc_curve, precision_recall_curve)

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

# ==== Data ====
gen = MIMICDataGenerator(500, 240, 20, 0.236, random_seed=SEED)
X, y, onset = gen.generate()
pre = MIMICPreprocessor(random_seed=SEED)
Xtr, Xval, Xte, ytr, yval, yte = pre.split(X, y)
Xtr_n, Xval_n, Xte_n = pre.normalize(Xtr, Xval, Xte)

# Recover onset for test-set positives by replaying the preprocessor's exact
# split, including its (subtly non-0.5) val_frac derived from imprecise floats.
from sklearn.model_selection import train_test_split
val_frac_exact = pre.val_ratio / (pre.val_ratio + pre.test_ratio)
Xtr_, Xtmp_, ytr_, ytmp_, otr_, otmp_ = train_test_split(
    X, y, onset, train_size=pre.train_ratio,
    random_state=pre.random_seed, stratify=y)
Xv_, Xte_, yv_, yte_, ov_, ote_ = train_test_split(
    Xtmp_, ytmp_, otmp_, train_size=val_frac_exact,
    random_state=pre.random_seed, stratify=ytmp_)
assert np.array_equal(yte, yte_), "test-set alignment failed"
onset_te = ote_
print(f"[data] Test: {len(yte)} stays, {int(yte.sum())} positive")
print(f"[data] Test onset (positives): min={onset_te[yte==1].min()}, "
      f"max={onset_te[yte==1].max()}, median={int(np.median(onset_te[yte==1]))}")

es = lambda: callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                     monitor="val_loss")

def report(p, tag):
    au = roc_auc_score(yte, p)
    ap = average_precision_score(yte, p)
    print(f"  [{tag}] AUROC={au:.4f}  AUPRC={ap:.4f}")
    return float(au), float(ap)

# ==== Full model (CNN+LSTM) — deterministic + MC ====
print("\n== Full model (CNN-BiLSTM) ==")
tf.random.set_seed(SEED); np.random.seed(SEED)
full = SepsisNetModel.build_model(input_shape=(240, 20))
print(f"  Params: {full.count_params()}")
full.fit(Xtr_n, ytr, validation_data=(Xval_n, yval),
         epochs=30, batch_size=32, callbacks=[es()], verbose=0)
p_det = full.predict(Xte_n, verbose=0).ravel()
au_det, ap_det = report(p_det, "deterministic")

mcd = MonteCarloDropout(full, n_iterations=100)
p_mc, std, var = mcd.predict_with_uncertainty(Xte_n)
au_mc, ap_mc = report(p_mc, "MC Dropout")
unc = mcd.compute_calibration_metrics(var, yte)
print(f"  variance ratio (sepsis/normal) = {unc['variance_ratio']:.4f}")

# ==== LSTM-only ablation ====
print("\n== LSTM-only ablation ==")
tf.random.set_seed(SEED); np.random.seed(SEED)
lstm = SepsisNetModel.build_model(input_shape=(240, 20), conv_filters=())
print(f"  Params: {lstm.count_params()}")
lstm.fit(Xtr_n, ytr, validation_data=(Xval_n, yval),
         epochs=30, batch_size=32, callbacks=[es()], verbose=0)
p_lstm = lstm.predict(Xte_n, verbose=0).ravel()
au_l, ap_l = report(p_lstm, "LSTM-only")

# ==== Bootstrap CIs on the full model ====
def op(y, pr, thr):
    yp = (pr >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    return (tp/(tp+fn) if tp+fn else 0.0,
            tn/(tn+fp) if tn+fp else 0.0,
            tp/(tp+fp) if tp+fp else 0.0,
            tn/(tn+fn) if tn+fn else 0.0)

def bootstrap(y, pr, n=2000, thr=0.5):
    rng = np.random.default_rng(SEED); N = len(y)
    acc = {k: [] for k in ["AUROC", "AUPRC", "Sens", "Spec", "PPV", "NPV"]}
    for _ in range(n):
        idx = rng.integers(0, N, N); yb, pb = y[idx], pr[idx]
        if yb.sum() == 0 or yb.sum() == len(yb): continue
        acc["AUROC"].append(roc_auc_score(yb, pb))
        acc["AUPRC"].append(average_precision_score(yb, pb))
        s, sp, pp, nv = op(yb, pb, thr)
        acc["Sens"].append(s); acc["Spec"].append(sp)
        acc["PPV"].append(pp); acc["NPV"].append(nv)
    return {k: {"mean": float(np.mean(v)),
                "lo": float(np.percentile(v, 2.5)),
                "hi": float(np.percentile(v, 97.5))} for k, v in acc.items()}

# ==== Calibration ====
def calib(y, pr, bins=10):
    e = np.linspace(0, 1, bins+1); rows = []; ece = 0.0; N = len(y)
    for i in range(bins):
        lo, hi = e[i], e[i+1]
        m = (pr >= lo) & (pr < hi) if i < bins-1 else (pr >= lo) & (pr <= hi)
        if m.sum() == 0:
            rows.append([float((lo+hi)/2), None, 0]); continue
        conf = float(pr[m].mean()); ac = float(y[m].mean()); c = int(m.sum())
        rows.append([conf, ac, c]); ece += (c/N)*abs(ac-conf)
    return {"reliability": rows, "ECE": float(ece),
            "Brier": float(brier_score_loss(y, pr)),
            "NLL": float(log_loss(y, np.clip(pr, 1e-7, 1-1e-7))),
            "cal_in_large": float(pr.mean()-y.mean())}

print("\n== Bootstrap + calibration ==")
ci = bootstrap(yte, p_mc)
cal = calib(yte, p_mc)
print(f"  AUROC CI: [{ci['AUROC']['lo']:.3f}, {ci['AUROC']['hi']:.3f}]")
print(f"  ECE={cal['ECE']:.4f}  Brier={cal['Brier']:.4f}")

# ==== Operating points at fixed specificity ====
op_points = {}
for target in [0.80, 0.90, 0.95]:
    cands = []
    for t in np.unique(p_mc):
        yp = (p_mc >= t).astype(int)
        tn = ((yp==0)&(yte==0)).sum(); fp = ((yp==1)&(yte==0)).sum()
        tp = ((yp==1)&(yte==1)).sum(); fn = ((yp==0)&(yte==1)).sum()
        spec = tn/(tn+fp) if tn+fp else 0
        sens = tp/(tp+fn) if tp+fn else 0
        ppv = tp/(tp+fp) if tp+fp else 0
        if spec >= target: cands.append((spec, sens, ppv, float(t), int(tp+fp)))
    if cands:
        cands.sort(key=lambda c: (-c[1], c[0]))
        s_, sn_, pp_, thr_, alerts_ = cands[0]
        op_points[str(target)] = dict(achieved_spec=s_, sens=sn_, ppv=pp_,
                                       thr=thr_, alerts=alerts_)

# ==== SOFA / qSOFA on same cohort ====
Xte_raw = Xte
last = Xte_raw[:, -1, :]
def sofa_score(L):
    s = np.zeros(len(L))
    pf = L[:,13]; s += np.select([pf<100,pf<200,pf<300,pf<400],[4,3,2,1],0)
    pl = L[:,12]; s += np.select([pl<20,pl<50,pl<100,pl<150],[4,3,2,1],0)
    bi = L[:,11]; s += np.select([bi>=12,bi>=6,bi>=2,bi>=1.2],[4,3,2,1],0)
    mp = L[:,3]; vp = L[:,19]; s += np.select([vp>0.2,vp>0,mp<70],[3,2,1],0)
    gc = L[:,7]; s += np.select([gc<6,gc<10,gc<13,gc<15],[4,3,2,1],0)
    cr = L[:,10]; s += np.select([cr>=5,cr>=3.5,cr>=2.0,cr>=1.2],[4,3,2,1],0)
    return s
def qsofa_score(L):
    return ((L[:,4]>=22).astype(int) + (L[:,1]<=100).astype(int)
            + (L[:,7]<15).astype(int))
sofa = sofa_score(last); qsofa = qsofa_score(last)
sofa_res = {"SOFA": {"AUROC": float(roc_auc_score(yte, sofa)),
                     "AUPRC": float(average_precision_score(yte, sofa))},
            "qSOFA":{"AUROC": float(roc_auc_score(yte, qsofa)),
                     "AUPRC": float(average_precision_score(yte, qsofa))}}
print(f"  SOFA AUROC={sofa_res['SOFA']['AUROC']:.4f}  "
      f"qSOFA AUROC={sofa_res['qSOFA']['AUROC']:.4f}")

# ==== NEW: Uncertainty error-detection analysis (R2#4) ====
print("\n== Uncertainty error-detection analysis ==")
yp05 = (p_mc >= 0.5).astype(int)
correct = (yp05 == yte)
mean_var_correct = float(var[correct].mean())
mean_var_wrong = float(var[~correct].mean())
mean_std_correct = float(std[correct].mean())
mean_std_wrong = float(std[~correct].mean())
# AUROC of predicting "error" using variance as a score
error_labels = (~correct).astype(int)
if error_labels.sum() > 0 and error_labels.sum() < len(error_labels):
    err_auroc_var = float(roc_auc_score(error_labels, var))
    err_auroc_std = float(roc_auc_score(error_labels, std))
else:
    err_auroc_var = err_auroc_std = None

# Risk-coverage: at each coverage level (retain the k least-uncertain), what is the accuracy?
order = np.argsort(std)
cov_grid = np.arange(0.10, 1.001, 0.05)
risk_curve = []
for c in cov_grid:
    k = max(1, int(np.round(c * len(std))))
    idx = order[:k]
    acc = float((yp05[idx] == yte[idx]).mean())
    risk_curve.append({"coverage": float(c), "accuracy": acc, "n_retained": int(k)})
print(f"  mean variance | correct: {mean_var_correct:.5f}   wrong: {mean_var_wrong:.5f}   ratio: {mean_var_wrong/mean_var_correct:.2f}x")
print(f"  AUROC(error | variance): {err_auroc_var:.4f}")

# ==== NEW: Coefficient-alpha sensitivity of p + alpha*sigma (R2#4) ====
print("\n== Sensitivity of p+alpha*sigma ==")
alpha_grid = [0.0, 0.05, 0.10, 0.20, 0.50, 1.00]
alpha_sens = []
for a in alpha_grid:
    p_adj = p_mc + a * std
    p_adj = np.clip(p_adj, 0, 1)
    au = float(roc_auc_score(yte, p_adj))
    ap = float(average_precision_score(yte, p_adj))
    # State distribution at the fixed thresholds 0.10/0.30/0.60
    n_green  = int(((p_adj < 0.10)).sum())
    n_yellow = int(((p_adj >= 0.10) & (p_adj < 0.30)).sum())
    n_orange = int(((p_adj >= 0.30) & (p_adj < 0.60)).sum())
    n_red    = int((p_adj >= 0.60).sum())
    # PPV / sens at threshold 0.30 (Orange onset) and 0.60 (Red onset)
    yp_orange = (p_adj >= 0.30).astype(int)
    yp_red    = (p_adj >= 0.60).astype(int)
    def op_(yp):
        tp = int(((yp==1)&(yte==1)).sum()); fp = int(((yp==1)&(yte==0)).sum())
        tn = int(((yp==0)&(yte==0)).sum()); fn = int(((yp==0)&(yte==1)).sum())
        return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                    sens = tp/(tp+fn) if tp+fn else 0.0,
                    spec = tn/(tn+fp) if tn+fp else 0.0,
                    ppv  = tp/(tp+fp) if tp+fp else 0.0)
    alpha_sens.append({
        "alpha": a, "AUROC": au, "AUPRC": ap,
        "n_green": n_green, "n_yellow": n_yellow,
        "n_orange": n_orange, "n_red": n_red,
        "orange_thr": op_(yp_orange), "red_thr": op_(yp_red),
    })
    print(f"  alpha={a:.2f}: AUROC={au:.4f} AUPRC={ap:.4f}  "
          f"G/Y/O/R = {n_green}/{n_yellow}/{n_orange}/{n_red}")

# ==== NEW: End-to-end escalation policy simulation on stay-level scores ====
# NB: this is retrospective, stay-level; the policy is applied ex-post to
# per-stay (p, sigma) pairs, treating the cohort as a sequence of stays.
print("\n== Escalation policy simulation (stay-level, retrospective) ==")
mdp = AlertEscalationMDP(consecutive_threshold=2)
alerts, states, actions, rewards = mdp.process_sequence(p_mc, std, yte)
state_counts = {"GREEN": 0, "YELLOW": 0, "ORANGE": 0, "RED": 0}
action_counts = {"No-Action": 0, "Notify": 0, "Escalate": 0, "ICU-Code": 0}
for a in alerts:
    state_counts[a["state"]] += 1
    action_counts[a["action"]] += 1
print(f"  states  G/Y/O/R = {state_counts['GREEN']}/{state_counts['YELLOW']}"
      f"/{state_counts['ORANGE']}/{state_counts['RED']}")
print(f"  actions NA/N/E/ICU = {action_counts['No-Action']}/{action_counts['Notify']}"
      f"/{action_counts['Escalate']}/{action_counts['ICU-Code']}")
print(f"  total reward: {float(rewards.sum()):.1f}")

# ==== NEW: Onset-stratified analysis (Path A feasibility teaser) ====
print("\n== Onset-stratified analysis (positives only) ==")
pos_mask = (yte == 1)
onset_pos = onset_te[pos_mask]
p_pos = p_mc[pos_mask]
median_onset = float(np.median(onset_pos))
early = onset_pos <= median_onset  # early onset -> more post-onset signal
late  = onset_pos >  median_onset
mean_p_early = float(p_pos[early].mean()) if early.any() else float("nan")
mean_p_late  = float(p_pos[late].mean())  if late.any()  else float("nan")
detect_rate_05_early = float((p_pos[early] >= 0.5).mean()) if early.any() else float("nan")
detect_rate_05_late  = float((p_pos[late]  >= 0.5).mean()) if late.any()  else float("nan")
print(f"  median onset = {median_onset:.0f} timesteps")
print(f"  mean p(sepsis) — early-onset positives: {mean_p_early:.4f} "
      f"({early.sum()} pts, detect@0.5 = {detect_rate_05_early:.3f})")
print(f"  mean p(sepsis) — late-onset positives:  {mean_p_late:.4f} "
      f"({late.sum()} pts, detect@0.5 = {detect_rate_05_late:.3f})")

# ==== Persist everything ====
results = {
    "seed": SEED,
    "test_set": {"n_stays": int(len(yte)), "n_positive": int(yte.sum()),
                 "prevalence": float(yte.mean())},
    "full_model_params": int(full.count_params()),
    "lstm_only_params": int(lstm.count_params()),
    "ablation": {
        "LSTM_only":              {"AUROC": round(au_l,4),  "AUPRC": round(ap_l,4),  "params": int(lstm.count_params())},
        "CNN_LSTM_deterministic": {"AUROC": round(au_det,4),"AUPRC": round(ap_det,4),"params": int(full.count_params())},
        "CNN_LSTM_MC_full":       {"AUROC": round(au_mc,4), "AUPRC": round(ap_mc,4), "params": int(full.count_params())},
    },
    "bootstrap_ci": ci, "calibration": cal, "operating_points": op_points,
    "uncertainty_variance": unc, "sofa_qsofa_same_cohort": sofa_res,
    "uncertainty_error_detection": {
        "mean_variance_correct": mean_var_correct,
        "mean_variance_wrong": mean_var_wrong,
        "mean_std_correct": mean_std_correct,
        "mean_std_wrong": mean_std_wrong,
        "wrong_over_correct_ratio": mean_var_wrong/mean_var_correct if mean_var_correct>0 else None,
        "AUROC_error_from_variance": err_auroc_var,
        "AUROC_error_from_std": err_auroc_std,
        "n_correct": int(correct.sum()), "n_wrong": int((~correct).sum()),
        "risk_coverage": risk_curve,
    },
    "alpha_sensitivity": alpha_sens,
    "escalation_policy_simulation": {
        "states": state_counts, "actions": action_counts,
        "total_reward": float(rewards.sum()),
        "n_stays": int(len(yte)),
    },
    "onset_stratification": {
        "median_onset_timesteps": median_onset,
        "onset_range": [int(onset_pos.min()), int(onset_pos.max())],
        "mean_p_early_onset": mean_p_early,
        "mean_p_late_onset":  mean_p_late,
        "detect_rate_05_early": detect_rate_05_early,
        "detect_rate_05_late":  detect_rate_05_late,
        "n_early": int(early.sum()), "n_late": int(late.sum()),
    },
}
json.dump(results, open("revision3_results.json", "w"), indent=2)
print("\n[saved] revision3_results.json")

# Save prediction arrays for figure generation and downstream analysis
np.savez("real_pred_full.npz",
         yte=yte, onset_te=onset_te,
         p_det=p_det, p_mc=p_mc, std=std, var=var,
         p_lstm=p_lstm, sofa=sofa, qsofa=qsofa)
print("[saved] real_pred_full.npz")
