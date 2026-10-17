"""Revision 4, Stage 1: deterministic seed-42 run with the three-method
comparison on the same trained weights.

This is a NEW training run, not a reproduction of the Revision 3 seed-42 run
(that run was non-deterministic and its weights were not retained).

Usage (PYTHONHASHSEED must be set before the interpreter starts):
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 python scripts/run_seed42.py --out runs/seed42_a

Outputs (all inside --out, which must be a new directory under revision4/):
    config.json            settings, seeds, environment, library versions
    history_cnn_bilstm.json, history_bilstm_only.json
    cnn_bilstm.weights.h5, bilstm_only.weights.h5
    predictions.npz        labels, split indices, every method's predictions
    run_checks.json        in-run integrity checks (weights/BN unchanged by MC)
    manifest.json          sha256 of every file above
"""
import os
import sys

assert os.environ.get("OMP_NUM_THREADS") == "1" and os.environ.get("MKL_NUM_THREADS") == "1", \
    "Set OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 before starting python."
assert os.environ.get("PYTHONHASHSEED") == "0", \
    "Run with PYTHONHASHSEED=0 set in the environment before python starts."
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import platform
import time
import warnings

warnings.filterwarnings("ignore")
REV4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REV4, "sepsisnet", "src"))

import numpy as np
import sklearn
import tensorflow as tf
import keras
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()

from data_loader import MIMICDataGenerator, MIMICPreprocessor
from models import SepsisNetModel
from uncertainty import MonteCarloDropout, batchnorm_layers

SETTINGS = dict(seed=42, mc_seed=42, n_patients=500, n_timesteps=240,
                n_features=20, sepsis_rate=0.236, epochs=30, batch_size=32,
                es_patience=5, es_monitor="val_loss", restore_best_weights=True,
                learning_rate=0.001, mc_passes=100, dropout_rate=0.4)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_digest(model):
    h = hashlib.sha256()
    for w in model.get_weights():
        h.update(np.ascontiguousarray(w).tobytes())
    return h.hexdigest()


def train(model, Xtr, ytr, Xval, yval):
    es = callbacks.EarlyStopping(patience=SETTINGS["es_patience"],
                                 restore_best_weights=SETTINGS["restore_best_weights"],
                                 monitor=SETTINGS["es_monitor"])
    hist = model.fit(Xtr, ytr, validation_data=(Xval, yval),
                     epochs=SETTINGS["epochs"], batch_size=SETTINGS["batch_size"],
                     callbacks=[es], verbose=0)
    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    h["_epochs_run"] = len(h["loss"])
    h["_stopped_epoch"] = int(es.stopped_epoch)
    h["_best_epoch"] = int(getattr(es, "best_epoch", -1))
    h["_best_val_loss"] = float(es.best) if es.best is not None else None
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    runs_root = os.path.join(REV4, "runs")
    if os.path.commonpath([out, runs_root]) != runs_root:
        sys.exit(f"--out must be inside {runs_root}")
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f"{out} exists and is not empty; refusing to overwrite")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    S = SETTINGS

    # ---- Data (identical to run_authoritative.py) ----
    gen = MIMICDataGenerator(S["n_patients"], S["n_timesteps"], S["n_features"],
                             S["sepsis_rate"], random_seed=S["seed"])
    X, y, onset = gen.generate()
    pre = MIMICPreprocessor(random_seed=S["seed"])
    Xtr, Xval, Xte, ytr, yval, yte = pre.split(X, y)
    Xtr_n, Xval_n, Xte_n = pre.normalize(Xtr, Xval, Xte)
    # Split membership: replay the preprocessor's exact split on indices
    vf = pre.val_ratio / (pre.val_ratio + pre.test_ratio)
    idx = np.arange(len(y))
    i_tr, i_tmp = train_test_split(idx, train_size=pre.train_ratio,
                                   random_state=pre.random_seed, stratify=y)
    i_val, i_te = train_test_split(i_tmp, train_size=vf,
                                   random_state=pre.random_seed, stratify=y[i_tmp])
    assert np.array_equal(y[i_te], yte) and np.array_equal(X[i_te], Xte)
    onset_te = onset[i_te]
    last = Xte[:, -1, :]
    print(f"[data] train/val/test = {len(i_tr)}/{len(i_val)}/{len(i_te)}; "
          f"test positives = {int(yte.sum())}", flush=True)

    # ---- CNN-BiLSTM ----
    keras.utils.set_random_seed(S["seed"])
    full = SepsisNetModel.build_model(input_shape=(S["n_timesteps"], S["n_features"]),
                                      dropout_rate=S["dropout_rate"],
                                      learning_rate=S["learning_rate"])
    hist_full = train(full, Xtr_n, ytr, Xval_n, yval)
    print(f"[cnn-bilstm] epochs run {hist_full['_epochs_run']}, "
          f"best epoch {hist_full['_best_epoch']}", flush=True)
    full.save_weights(os.path.join(out, "cnn_bilstm.weights.h5"))
    w_before = weights_digest(full)
    bn_before = [(l.moving_mean.numpy().copy(), l.moving_variance.numpy().copy())
                 for l in batchnorm_layers(full)]

    p_det = full.predict(Xte_n, verbose=0).ravel()
    mcd = MonteCarloDropout(full, n_iterations=S["mc_passes"])
    keras.utils.set_random_seed(S["mc_seed"])
    p_corr, std_corr, var_corr, smp_corr = mcd.predict_with_uncertainty(
        Xte_n, bn_mode="inference", seed=S["mc_seed"], return_samples=True)
    keras.utils.set_random_seed(S["mc_seed"])
    p_leg, std_leg, var_leg, smp_leg = mcd.predict_with_uncertainty(
        Xte_n, bn_mode="batch", seed=S["mc_seed"], return_samples=True)
    p_det_after = full.predict(Xte_n, verbose=0).ravel()
    bn_after = [(l.moving_mean.numpy(), l.moving_variance.numpy())
                for l in batchnorm_layers(full)]
    checks = {
        "weights_unchanged_by_mc": weights_digest(full) == w_before,
        "bn_stats_unchanged_by_mc": all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                                        for a, b in zip(bn_before, bn_after)),
        "deterministic_inference_identical_after_mc": bool(np.array_equal(p_det, p_det_after)),
        "weights_digest_cnn_bilstm": w_before,
    }

    # ---- BiLSTM-only ablation ----
    keras.utils.set_random_seed(S["seed"])
    lstm = SepsisNetModel.build_model(input_shape=(S["n_timesteps"], S["n_features"]),
                                      conv_filters=(), dropout_rate=S["dropout_rate"],
                                      learning_rate=S["learning_rate"])
    hist_lstm = train(lstm, Xtr_n, ytr, Xval_n, yval)
    print(f"[bilstm-only] epochs run {hist_lstm['_epochs_run']}, "
          f"best epoch {hist_lstm['_best_epoch']}", flush=True)
    lstm.save_weights(os.path.join(out, "bilstm_only.weights.h5"))
    p_lstm = lstm.predict(Xte_n, verbose=0).ravel()
    checks["weights_digest_bilstm_only"] = weights_digest(lstm)
    checks["params_cnn_bilstm"] = int(full.count_params())
    checks["params_bilstm_only"] = int(lstm.count_params())

    # ---- SOFA / qSOFA (formula copied from run_authoritative.py) ----
    def sofa_score(L):
        s = np.zeros(len(L))
        pf = L[:, 13]; s += np.select([pf < 100, pf < 200, pf < 300, pf < 400], [4, 3, 2, 1], 0)
        pl = L[:, 12]; s += np.select([pl < 20, pl < 50, pl < 100, pl < 150], [4, 3, 2, 1], 0)
        bi = L[:, 11]; s += np.select([bi >= 12, bi >= 6, bi >= 2, bi >= 1.2], [4, 3, 2, 1], 0)
        mp = L[:, 3]; vp = L[:, 19]; s += np.select([vp > 0.2, vp > 0, mp < 70], [3, 2, 1], 0)
        gc = L[:, 7]; s += np.select([gc < 6, gc < 10, gc < 13, gc < 15], [4, 3, 2, 1], 0)
        cr = L[:, 10]; s += np.select([cr >= 5, cr >= 3.5, cr >= 2.0, cr >= 1.2], [4, 3, 2, 1], 0)
        return s

    def qsofa_score(L):
        return ((L[:, 4] >= 22).astype(int) + (L[:, 1] <= 100).astype(int)
                + (L[:, 7] < 15).astype(int))

    np.savez(os.path.join(out, "predictions.npz"),
             yte=yte, onset_te=onset_te, idx_train=i_tr, idx_val=i_val, idx_test=i_te,
             p_det=p_det,
             p_mc_corrected=p_corr, std_corrected=std_corr, var_corrected=var_corr,
             samples_corrected=smp_corr,
             p_mc_legacy=p_leg, std_legacy=std_leg, var_legacy=var_leg,
             samples_legacy=smp_leg,
             p_lstm=p_lstm, sofa=sofa_score(last), qsofa=qsofa_score(last))
    json.dump(hist_full, open(os.path.join(out, "history_cnn_bilstm.json"), "w"), indent=2)
    json.dump(hist_lstm, open(os.path.join(out, "history_bilstm_only.json"), "w"), indent=2)
    json.dump(checks, open(os.path.join(out, "run_checks.json"), "w"), indent=2)

    config = {
        "description": "Revision 4 Stage 1 deterministic seed-42 run (new training run; "
                       "not a reproduction of the Revision 3 run)",
        "settings": S,
        "seeding": "keras.utils.set_random_seed(seed) (Python random, NumPy, TF, Keras "
                   "global SeedGenerator) before each model build and before each MC "
                   "call; MC Dropout generator states reset from mc_seed inside "
                   "predict_with_uncertainty; data generator uses its own "
                   "np.random.default_rng(seed)",
        "mc_methods": {"deterministic": "model.predict (training=False)",
                       "legacy": "bn_mode='batch' (Revision 3 behaviour, BN stats restored after)",
                       "corrected": "bn_mode='inference' (primary)"},
        "environment": {k: os.environ.get(k) for k in
                        ("PYTHONHASHSEED", "TF_ENABLE_ONEDNN_OPTS", "TF_DETERMINISTIC_OPS",
                         "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "threads": {"intra_op": 1, "inter_op": 1}, "enable_op_determinism": True,
        "versions": {"python": platform.python_version(), "tensorflow": tf.__version__,
                     "keras": keras.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "platform": platform.platform()},
        "elapsed_sec": round(time.time() - t0, 1),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json.dump(config, open(os.path.join(out, "config.json"), "w"), indent=2)
    manifest = {f: sha256(os.path.join(out, f)) for f in sorted(os.listdir(out))}
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=2)
    print(f"[done] {out} in {config['elapsed_sec']} s; checks: {checks}", flush=True)


if __name__ == "__main__":
    main()
