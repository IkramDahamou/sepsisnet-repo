"""Compare the deterministic vs non-deterministic 10-seed reruns.

For each of the 10 seeds and each of the 3 models, computes
|non-det AUROC - det AUROC| and |non-det AUPRC - det AUPRC|, then reports
mean/max of these absolute differences overall and per model. This
quantifies how much of the earlier "seed variability" (from
run_multiseed_10seed.py, non-deterministic) was actually run-to-run
training noise rather than genuine seed sensitivity, by comparing against
the bit-for-bit reproducible deterministic reruns of the SAME seeds.

Saves multiseed_deterministic_vs_nondeterministic_comparison.json.
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DET_PATH = os.path.join(HERE, "..", "multiseed_results_10seed_deterministic.json")
NONDET_PATH = os.path.join(HERE, "..", "multiseed_results_10seed_nondeterministic.json")
OUT_PATH = os.path.join(HERE, "..", "multiseed_deterministic_vs_nondeterministic_comparison.json")

MODELS = ["CNN_LSTM_deterministic", "CNN_LSTM_MC", "LSTM_only"]


def main():
    det = json.load(open(DET_PATH))
    nondet = json.load(open(NONDET_PATH))

    det_by_seed = {r["seed"]: r for r in det}
    nondet_by_seed = {r["seed"]: r for r in nondet}
    assert set(det_by_seed) == set(nondet_by_seed) == {7, 13, 21, 42, 55, 67, 79, 91, 103, 115}
    seeds = sorted(det_by_seed)

    rows = []
    diffs_auroc, diffs_auprc = [], []
    per_model_diffs = {m: {"AUROC": [], "AUPRC": []} for m in MODELS}

    header = f"{'seed':>5} {'model':28s} {'det_AUROC':>10} {'nondet_AUROC':>13} {'|diff|':>8} | {'det_AUPRC':>10} {'nondet_AUPRC':>13} {'|diff|':>8}"
    print(header)
    for s in seeds:
        dr, nr = det_by_seed[s], nondet_by_seed[s]
        for m in MODELS:
            d_au, nd_au = dr[m]["AUROC"], nr[m]["AUROC"]
            d_ap, nd_ap = dr[m]["AUPRC"], nr[m]["AUPRC"]
            diff_au = abs(d_au - nd_au)
            diff_ap = abs(d_ap - nd_ap)
            per_model_diffs[m]["AUROC"].append(diff_au)
            per_model_diffs[m]["AUPRC"].append(diff_ap)
            diffs_auroc.append(diff_au)
            diffs_auprc.append(diff_ap)
            rows.append({
                "seed": s, "model": m,
                "det_AUROC": d_au, "nondet_AUROC": nd_au, "abs_diff_AUROC": round(diff_au, 4),
                "det_AUPRC": d_ap, "nondet_AUPRC": nd_ap, "abs_diff_AUPRC": round(diff_ap, 4),
            })
            print(f"{s:>5} {m:28s} {d_au:>10.4f} {nd_au:>13.4f} {diff_au:>8.4f} | {d_ap:>10.4f} {nd_ap:>13.4f} {diff_ap:>8.4f}")

    print("\n=== Overall (all 30 seed-model pairs) ===")
    print(f"AUROC |diff|: mean={np.mean(diffs_auroc):.4f}  max={np.max(diffs_auroc):.4f}")
    print(f"AUPRC |diff|: mean={np.mean(diffs_auprc):.4f}  max={np.max(diffs_auprc):.4f}")

    print("\n=== Per-model ===")
    for m in MODELS:
        a, p = per_model_diffs[m]["AUROC"], per_model_diffs[m]["AUPRC"]
        print(f"{m:28s} AUROC mean={np.mean(a):.4f} max={np.max(a):.4f}  AUPRC mean={np.mean(p):.4f} max={np.max(p):.4f}")

    out = {
        "description": ("Absolute differences between deterministic (op-determinism enabled) "
                         "and non-deterministic (default TF ops) reruns of the SAME 10 seeds "
                         "under the same fast-sweep config (epochs=15, MC=50). Quantifies how "
                         "much of the seed-to-seed variability reported under the "
                         "non-deterministic run was actually run-to-run training noise rather "
                         "than genuine seed sensitivity."),
        "det_file": "multiseed_results_10seed_deterministic.json",
        "nondet_file": "multiseed_results_10seed_nondeterministic.json",
        "per_seed_model_diffs": rows,
        "overall": {
            "AUROC_abs_diff_mean": round(float(np.mean(diffs_auroc)), 4),
            "AUROC_abs_diff_max": round(float(np.max(diffs_auroc)), 4),
            "AUPRC_abs_diff_mean": round(float(np.mean(diffs_auprc)), 4),
            "AUPRC_abs_diff_max": round(float(np.max(diffs_auprc)), 4),
        },
        "per_model": {
            m: {
                "AUROC_abs_diff_mean": round(float(np.mean(per_model_diffs[m]["AUROC"])), 4),
                "AUROC_abs_diff_max": round(float(np.max(per_model_diffs[m]["AUROC"])), 4),
                "AUPRC_abs_diff_mean": round(float(np.mean(per_model_diffs[m]["AUPRC"])), 4),
                "AUPRC_abs_diff_max": round(float(np.max(per_model_diffs[m]["AUPRC"])), 4),
            } for m in MODELS
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved comparison to {OUT_PATH}")


if __name__ == "__main__":
    main()
