"""Revision 4, Stage 2 Part B: 10-seed sweep (resumable).

Settings are those of the Revision 3 deterministic multi-seed sweep:
seeds from reference_data/rev3_multiseed_results_10seed_deterministic.json, max_epochs 15,
batch 32, EarlyStopping(patience=5, restore_best_weights=True, val_loss),
50 MC passes, oneDNN off, op determinism, 1 thread. Per seed, on the SAME
CNN-BiLSTM weights: deterministic, legacy MC (bn_mode="batch") and corrected
MC (bn_mode="inference", primary), MC seed = training seed; plus BiLSTM-only.

Seeding scheme (--scheme), chosen from Stage 2 A1:
  np_tf     : np.random.seed + tf.random.set_seed (Revision 3 seeds 7-42)
  np_tf_py  : random.seed + np.random.seed + tf.random.set_seed
              (Revision 3 run_multiseed_resume.py, seeds 55-115)
  keras_set_random_seed : keras.utils.set_random_seed(seed) (Python, NumPy,
              TF + Keras global SeedGenerator reset) -- APPROVED for Part B
Thread pinning OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 must be set in the
environment before the interpreter starts (asserted).
applied at the same three points as Revision 3: before data, before the
CNN-BiLSTM build, before the BiLSTM-only build.

Resumable: a seed whose folder holds manifest.json is skipped; a folder
without manifest.json (interrupted attempt) is renamed to
seed_<n>_incomplete_<timestamp> (never deleted) and the seed is rerun with
identical settings; every such event is logged in sweep_events.json.

Replicate runs (approved Stage 2 rule): --tag a (REPORTED result) and --tag b
(reproducibility measurement only) write to runs/sweep_a/ and runs/sweep_b/.
--seeds optionally restricts a worker to a subset (separate events file).

Usage:
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1       python scripts/run_sweep.py --scheme keras_set_random_seed --tag a [--seeds 7 13]
"""
import os
import sys

assert os.environ.get("PYTHONHASHSEED") == "0"
assert os.environ.get("OMP_NUM_THREADS") == "1" and os.environ.get("MKL_NUM_THREADS") == "1",     "Set OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 before starting python (approved Stage 2 setting)."
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import platform
import random
import time
import warnings

warnings.filterwarnings("ignore")
REV4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.dirname(REV4)
sys.path.insert(0, os.path.join(REV4, "sepsisnet", "src"))

import numpy as np
import sklearn
import tensorflow as tf
import keras
from tensorflow.keras import callbacks

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()

from data_loader import MIMICDataGenerator, MIMICPreprocessor
from models import SepsisNetModel
from uncertainty import MonteCarloDropout
from sklearn.model_selection import train_test_split


def sofa_score(L):   # formula copied from Revision 3 run_authoritative.py
    v = np.zeros(len(L))
    pf = L[:, 13]; v += np.select([pf < 100, pf < 200, pf < 300, pf < 400], [4, 3, 2, 1], 0)
    pl = L[:, 12]; v += np.select([pl < 20, pl < 50, pl < 100, pl < 150], [4, 3, 2, 1], 0)
    bi = L[:, 11]; v += np.select([bi >= 12, bi >= 6, bi >= 2, bi >= 1.2], [4, 3, 2, 1], 0)
    mp = L[:, 3]; vp = L[:, 19]; v += np.select([vp > 0.2, vp > 0, mp < 70], [3, 2, 1], 0)
    gc = L[:, 7]; v += np.select([gc < 6, gc < 10, gc < 13, gc < 15], [4, 3, 2, 1], 0)
    cr = L[:, 10]; v += np.select([cr >= 5, cr >= 3.5, cr >= 2.0, cr >= 1.2], [4, 3, 2, 1], 0)
    return v


def qsofa_score(L):
    return (L[:, 4] >= 22).astype(int) + (L[:, 1] <= 100).astype(int) + (L[:, 7] < 15).astype(int)

SETTINGS = dict(max_epochs=15, batch_size=32, es_patience=5, es_monitor="val_loss",
                restore_best_weights=True, mc_passes=50, n_patients=500,
                n_timesteps=240, n_features=20, sepsis_rate=0.236)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def run_seed(s, scheme, out, tag):
    S = SETTINGS; seeds_set = []

    def seed_point(label):
        if scheme == "keras_set_random_seed":
            keras.utils.set_random_seed(s)
        else:
            if scheme == "np_tf_py":
                random.seed(s)
            np.random.seed(s); tf.random.set_seed(s)
        seeds_set.append({"point": label, "scheme": scheme,
                          "python_random": scheme in ("np_tf_py", "keras_set_random_seed"),
                          "numpy": s, "tf_global": s, "keras_global_seedgen_reset": scheme == "keras_set_random_seed"})

    def es():
        return callbacks.EarlyStopping(patience=S["es_patience"], restore_best_weights=S["restore_best_weights"],
                                       monitor=S["es_monitor"])

    def fit(m, Xtr, ytr, Xv, yv):
        cb = es()
        h = m.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=S["max_epochs"], batch_size=S["batch_size"],
                  callbacks=[cb], verbose=0)
        d = {k: [float(x) for x in v] for k, v in h.history.items()}
        d.update(_epochs_run=len(d["loss"]), _best_epoch=int(cb.best_epoch), _stopped_epoch=int(cb.stopped_epoch))
        return d

    t0 = time.time()
    seed_point("before data")
    X, y, onset = MIMICDataGenerator(S["n_patients"], S["n_timesteps"], S["n_features"], S["sepsis_rate"],
                                     random_seed=s).generate()
    pre = MIMICPreprocessor(random_seed=s)
    Xtr, Xval, Xte, ytr, yval, yte = pre.split(X, y)
    Xtr_n, Xval_n, Xte_n = pre.normalize(Xtr, Xval, Xte)
    vf = pre.val_ratio / (pre.val_ratio + pre.test_ratio)
    i_tr, i_tmp = train_test_split(np.arange(len(y)), train_size=pre.train_ratio, random_state=s, stratify=y)
    i_val, i_te = train_test_split(i_tmp, train_size=vf, random_state=s, stratify=y[i_tmp])
    assert np.array_equal(y[i_te], yte) and np.array_equal(X[i_te], Xte)
    last = Xte[:, -1, :]

    seed_point("before CNN-BiLSTM build")
    full = SepsisNetModel.build_model(input_shape=(240, 20))
    h_full = fit(full, Xtr_n, ytr, Xval_n, yval)
    full.save_weights(os.path.join(out, "cnn_bilstm.weights.h5"))
    p_det = full.predict(Xte_n, verbose=0).ravel()
    mcd = MonteCarloDropout(full, n_iterations=S["mc_passes"])
    p_c, s_c, v_c, smp_c = mcd.predict_with_uncertainty(Xte_n, bn_mode="inference", seed=s, return_samples=True)
    p_l, s_l, v_l, smp_l = mcd.predict_with_uncertainty(Xte_n, bn_mode="batch", seed=s, return_samples=True)
    det_after = full.predict(Xte_n, verbose=0).ravel()

    seed_point("before BiLSTM-only build")
    lstm = SepsisNetModel.build_model(input_shape=(240, 20), conv_filters=())
    h_lstm = fit(lstm, Xtr_n, ytr, Xval_n, yval)
    lstm.save_weights(os.path.join(out, "bilstm_only.weights.h5"))
    p_lstm = lstm.predict(Xte_n, verbose=0).ravel()

    np.savez(os.path.join(out, "predictions.npz"), yte=yte, onset_te=onset[i_te],
             idx_train=i_tr, idx_val=i_val, idx_test=i_te,
             p_det=p_det, p_mc_corrected=p_c, std_corrected=s_c, var_corrected=v_c, samples_corrected=smp_c,
             p_mc_legacy=p_l, std_legacy=s_l, var_legacy=v_l, samples_legacy=smp_l,
             p_lstm=p_lstm, sofa=sofa_score(last), qsofa=qsofa_score(last))
    json.dump(h_full, open(os.path.join(out, "history_cnn_bilstm.json"), "w"), indent=2)
    json.dump(h_lstm, open(os.path.join(out, "history_bilstm_only.json"), "w"), indent=2)
    cfg = {"seed": s, "tag": tag, "role": "reported" if tag == "a" else "reproducibility check only", "scheme": scheme, "settings": S, "mc_seed": s, "seeds_set": seeds_set,
           "settings_source": "Revision 3 run_multiseed_10seed_deterministic.py settings",
           "check_det_identical_after_mc": bool(np.array_equal(p_det, det_after)),
           "environment": {k: os.environ.get(k) for k in ("PYTHONHASHSEED", "TF_ENABLE_ONEDNN_OPTS", "TF_DETERMINISTIC_OPS",
                                                          "TF_CUDNN_DETERMINISTIC", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                          "CUDA_VISIBLE_DEVICES")},
           "threads": 1, "enable_op_determinism": True,
           "versions": {"python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__,
                        "numpy": np.__version__, "scikit-learn": sklearn.__version__},
           "elapsed_sec": round(time.time() - t0, 1)}
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    man = {f: sha256(os.path.join(out, f)) for f in sorted(os.listdir(out))}
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=2)   # written last = completion marker
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=["np_tf", "np_tf_py", "keras_set_random_seed"])
    ap.add_argument("--tag", required=True, choices=["a", "b"])
    ap.add_argument("--seeds", type=int, nargs="*")
    a = ap.parse_args()
    seeds = [r["seed"] for r in json.load(open(os.path.join(REV4, "reference_data", "rev3_multiseed_results_10seed_deterministic.json")))]
    if a.seeds:
        assert set(a.seeds) <= set(seeds), "unknown seed"
        seeds = [x for x in seeds if x in a.seeds]
    SWEEP = os.path.join(REV4, "runs", f"sweep_{a.tag}")
    os.makedirs(SWEEP, exist_ok=True)
    ev_path = os.path.join(SWEEP, f"sweep_events_{'_'.join(map(str, seeds))}.json")
    events = json.load(open(ev_path)) if os.path.exists(ev_path) else []
    for s in seeds:
        out = os.path.join(SWEEP, f"seed_{s}")
        if os.path.exists(os.path.join(out, "manifest.json")):
            events.append({"seed": s, "event": "skipped (complete)", "time": time.strftime("%Y-%m-%d %H:%M:%S")})
            continue
        if os.path.exists(out):
            moved = f"{out}_incomplete_{time.strftime('%Y%m%d_%H%M%S')}"
            os.rename(out, moved)
            events.append({"seed": s, "event": f"incomplete folder moved to {os.path.basename(moved)}; rerun with identical settings"})
        os.makedirs(out)
        events.append({"seed": s, "event": "start", "time": time.strftime("%Y-%m-%d %H:%M:%S"), "scheme": a.scheme})
        json.dump(events, open(ev_path, "w"), indent=2)
        try:
            cfg = run_seed(s, a.scheme, out, a.tag)
            events.append({"seed": s, "event": "done", "elapsed_sec": cfg["elapsed_sec"]})
            print(f"[seed {s}] done in {cfg['elapsed_sec']} s", flush=True)
        except Exception as e:  # report, do not substitute
            events.append({"seed": s, "event": "FAILED", "error": repr(e)})
            print(f"[seed {s}] FAILED: {e!r}", flush=True)
        json.dump(events, open(ev_path, "w"), indent=2)


if __name__ == "__main__":
    main()
