"""10-seed deterministic multi-seed variability run.

Uses TF's op determinism and single-thread execution so results are
bit-for-bit reproducible across process runs (verified on seeds 13
and 21 via scripts/determinism_check_run.py).

Invocation:
    PYTHONHASHSEED=0 python scripts/run_multiseed_10seed_deterministic.py
"""
import os
# MUST be set BEFORE importing tensorflow / numpy
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "sepsisnet", "src"))

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

# Single-thread execution + op determinism -> bit-for-bit reproducibility
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()

from data_loader import MIMICDataGenerator, MIMICPreprocessor
from models import SepsisNetModel
from uncertainty import MonteCarloDropout
from sklearn.metrics import roc_auc_score, average_precision_score

SEEDS = [7, 13, 21, 42, 55, 67, 79, 91, 103, 115]
MAX_EPOCHS = 15
MC_PASSES  = 50

DETERMINISM_CONFIG = {
    "PYTHONHASHSEED": os.environ["PYTHONHASHSEED"],
    "TF_ENABLE_ONEDNN_OPTS": os.environ["TF_ENABLE_ONEDNN_OPTS"],
    "TF_DETERMINISTIC_OPS": os.environ["TF_DETERMINISTIC_OPS"],
    "intra_op_threads": 1,
    "inter_op_threads": 1,
    "enable_op_determinism": True,
}

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "multiseed_results_10seed_deterministic.json")
results = []
if os.path.exists(OUT_PATH):
    try:
        results = json.load(open(OUT_PATH))
        done_seeds = {r["seed"] for r in results}
        print(f"[resume] loaded {len(results)} existing seed(s): "
              f"{sorted(done_seeds)}", flush=True)
    except Exception:
        results = []
        done_seeds = set()
else:
    done_seeds = set()

for s in SEEDS:
    if s in done_seeds:
        continue
    print(f"\n===== seed {s} ===== (elapsed clock: {time.strftime('%H:%M:%S')})",
          flush=True)
    t0 = time.time()
    np.random.seed(s); tf.random.set_seed(s)

    gen = MIMICDataGenerator(500, 240, 20, 0.236, random_seed=s)
    X, y, _ = gen.generate()
    pre = MIMICPreprocessor(random_seed=s)
    Xtr, Xval, Xte, ytr, yval, yte = pre.split(X, y)
    Xtr_n, Xval_n, Xte_n = pre.normalize(Xtr, Xval, Xte)

    def es():
        return callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                       monitor="val_loss")

    tf.random.set_seed(s); np.random.seed(s)
    full = SepsisNetModel.build_model(input_shape=(240, 20))
    full.fit(Xtr_n, ytr, validation_data=(Xval_n, yval),
             epochs=MAX_EPOCHS, batch_size=32, callbacks=[es()], verbose=0)
    p_det = full.predict(Xte_n, verbose=0).ravel()
    au_det = float(roc_auc_score(yte, p_det))
    ap_det = float(average_precision_score(yte, p_det))
    mcd = MonteCarloDropout(full, n_iterations=MC_PASSES)
    p_mc, std, var = mcd.predict_with_uncertainty(Xte_n)
    au_mc = float(roc_auc_score(yte, p_mc))
    ap_mc = float(average_precision_score(yte, p_mc))

    tf.random.set_seed(s); np.random.seed(s)
    lstm = SepsisNetModel.build_model(input_shape=(240, 20), conv_filters=())
    lstm.fit(Xtr_n, ytr, validation_data=(Xval_n, yval),
             epochs=MAX_EPOCHS, batch_size=32, callbacks=[es()], verbose=0)
    p_lstm = lstm.predict(Xte_n, verbose=0).ravel()
    au_l = float(roc_auc_score(yte, p_lstm))
    ap_l = float(average_precision_score(yte, p_lstm))

    row = {
        "seed": s, "test_positives": int(yte.sum()),
        "CNN_LSTM_deterministic": {"AUROC": round(au_det, 4), "AUPRC": round(ap_det, 4)},
        "CNN_LSTM_MC":             {"AUROC": round(au_mc, 4),  "AUPRC": round(ap_mc, 4)},
        "LSTM_only":               {"AUROC": round(au_l, 4),   "AUPRC": round(ap_l, 4)},
        "elapsed_sec": round(time.time() - t0, 1),
        "max_epochs": MAX_EPOCHS, "mc_passes": MC_PASSES,
        "determinism_config": DETERMINISM_CONFIG,
        "note": None,
    }
    results.append(row)
    print(json.dumps(row), flush=True)
    json.dump(results, open(OUT_PATH, "w"), indent=2)

print(f"\n[done] {len(results)}/{len(SEEDS)} total seeds in file. "
      f"Saved to {OUT_PATH}", flush=True)
