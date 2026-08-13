"""Fast multi-seed variability (Reviewer 2 major #5).
Runs seeds 7, 13, 21 in addition to primary seed 42.
Uses 15 max epochs (early stopping still active) and 50 MC passes to fit
the sandbox time budget. Saves incrementally after each seed."""
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

SEEDS = [7, 13, 21]
MAX_EPOCHS = 15  # early stopping usually kicks in ~10 anyway
MC_PASSES  = 50

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
    }
    results.append(row)
    print(json.dumps(row), flush=True)
    json.dump(results, open("multiseed_results.json", "w"), indent=2)
