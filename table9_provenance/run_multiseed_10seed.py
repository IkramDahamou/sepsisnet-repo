"""Extended multi-seed variability run (Reviewer 2 major #5) — 10 seeds.

Runs seeds 7, 13, 21, 42, 55, 67, 79, 91, 103, 115 under IDENTICAL
methodology/configuration: same MIMICDataGenerator / MIMICPreprocessor
data-generation and train/test-split procedure, same SepsisNetModel
architectures (CNN-BiLSTM full model + LSTM-only ablation), same
max_epochs=15 (with EarlyStopping, patience=5, restore_best_weights),
same MC dropout passes=50.

NOTE ON SEED 42: the manuscript's primary/authoritative numbers for
seed 42 come from run_authoritative.py, which uses epochs=30 and
MC n_iterations=100 — a DIFFERENT configuration than this fast
multi-seed sweep (epochs=15, MC=50). Those two configurations are
therefore not directly comparable. To keep this 10-seed variability
experiment methodologically uniform, seed 42 is rerun HERE from
scratch under the exact same fast config as the other 9 seeds. Do
not conflate these seed-42 numbers with the manuscript's primary
seed-42 numbers in revision3_results.json / metrics_seed42.json.

Saves incrementally after each seed to multiseed_results_10seed.json.
"""
import os, sys, json, time, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sepsisnet", "src"))

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from data_loader import MIMICDataGenerator, MIMICPreprocessor
from models import SepsisNetModel
from uncertainty import MonteCarloDropout
from sklearn.metrics import roc_auc_score, average_precision_score

SEEDS = [7, 13, 21, 42, 55, 67, 79, 91, 103, 115]
MAX_EPOCHS = 15  # early stopping usually kicks in ~10 anyway
MC_PASSES  = 50

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "multiseed_results_10seed.json")

results = []
for s in SEEDS:
    print(f"\n===== seed {s} ===== (elapsed clock: {time.strftime('%H:%M:%S')})", flush=True)
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
        "seed": s,
        "test_positives": int(yte.sum()),
        "CNN_LSTM_deterministic": {"AUROC": round(au_det, 4), "AUPRC": round(ap_det, 4)},
        "CNN_LSTM_MC":             {"AUROC": round(au_mc, 4),  "AUPRC": round(ap_mc, 4)},
        "LSTM_only":               {"AUROC": round(au_l, 4),   "AUPRC": round(ap_l, 4)},
        "elapsed_sec": round(time.time() - t0, 1),
        "max_epochs": MAX_EPOCHS, "mc_passes": MC_PASSES,
        "note": ("rerun under fast-sweep config (epochs=15, MC=50); "
                 "NOT the manuscript's primary seed-42 config (epochs=30, MC=100)")
                if s == 42 else None,
    }
    results.append(row)
    print(json.dumps(row), flush=True)
    json.dump(results, open(OUT_PATH, "w"), indent=2)

print(f"\n[done] {len(results)}/{len(SEEDS)} seeds completed. Saved to {OUT_PATH}", flush=True)
