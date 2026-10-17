"""Bitwise comparison of two Revision 4 run directories.

Usage:
    python scripts/compare_runs.py runs/seed42_a runs/seed42_b --out results/seed42_reproducibility.json
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np


def h5_arrays(path):
    out = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = obj[()]
        f.visititems(visit)
    return out


def compare_arrays(a, b):
    keys = sorted(set(a) | set(b))
    rows = {}
    for k in keys:
        if k not in a or k not in b:
            rows[k] = {"present_in_both": False}
            continue
        x, y = np.asarray(a[k]), np.asarray(b[k])
        same_shape = x.shape == y.shape
        bitwise = bool(same_shape and x.dtype == y.dtype and x.tobytes() == y.tobytes())
        diff = (float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max())
                if same_shape and x.size and np.issubdtype(x.dtype, np.number) else None)
        rows[k] = {"bitwise_equal": bitwise, "max_abs_diff": diff, "shape": list(x.shape)}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a"); ap.add_argument("run_b"); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; refusing to overwrite")
    A, B = args.run_a, args.run_b
    res = {"run_a": A, "run_b": B}
    for w in ("cnn_bilstm.weights.h5", "bilstm_only.weights.h5"):
        res[w] = compare_arrays(h5_arrays(os.path.join(A, w)), h5_arrays(os.path.join(B, w)))
    pa, pb = np.load(os.path.join(A, "predictions.npz")), np.load(os.path.join(B, "predictions.npz"))
    res["predictions.npz"] = compare_arrays({k: pa[k] for k in pa.files}, {k: pb[k] for k in pb.files})
    for j in ("history_cnn_bilstm.json", "history_bilstm_only.json", "run_checks.json"):
        res[j] = {"identical": json.load(open(os.path.join(A, j))) == json.load(open(os.path.join(B, j)))}
    ma, mb = json.load(open(os.path.join(A, "manifest.json"))), json.load(open(os.path.join(B, "manifest.json")))
    res["file_sha256_equal"] = {k: ma.get(k) == mb.get(k) for k in sorted(set(ma) | set(mb))}

    def all_bitwise(d):
        return all(v.get("bitwise_equal", False) for v in d.values())
    res["summary"] = {
        "cnn_bilstm_weights_bitwise_equal": all_bitwise(res["cnn_bilstm.weights.h5"]),
        "bilstm_only_weights_bitwise_equal": all_bitwise(res["bilstm_only.weights.h5"]),
        "predictions_bitwise_equal": all_bitwise(res["predictions.npz"]),
        "histories_identical": res["history_cnn_bilstm.json"]["identical"] and res["history_bilstm_only.json"]["identical"],
        "max_abs_diff_any_prediction": max((v["max_abs_diff"] or 0.0) for v in res["predictions.npz"].values()),
        "max_abs_diff_any_weight": max((v["max_abs_diff"] or 0.0) for w in ("cnn_bilstm.weights.h5", "bilstm_only.weights.h5")
                                       for v in res[w].values()),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res["summary"], indent=2))
    print("weights-file sha256 equal:", {k: v for k, v in res["file_sha256_equal"].items()})


if __name__ == "__main__":
    main()
