"""Bit-for-bit determinism check: run a single seed once and save predictions
plus metrics. Running twice with the same tag (e.g., 'a' and 'b') and comparing
the outputs verifies that op-determinism is actually producing identical
results across independent process invocations.

Invocation:
    PYTHONHASHSEED=0 python scripts/determinism_check_run.py SEED TAG
Example:
    PYTHONHASHSEED=0 python scripts/determinism_check_run.py 13 a
    PYTHONHASHSEED=0 python scripts/determinism_check_run.py 13 b
    python -c "import numpy as np; a = np.load('determinism_check_seed13_a.npz'); \
               b = np.load('determinism_check_seed13_b.npz'); \
               print('identical:', all(np.array_equal(a[k], b[k]) for k in a.files))"
"""
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "sepsisnet", "src"))

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()

from data_loader import MIMICDataGenerator, MIMICPreprocessor
from models import SepsisNetModel
from uncertainty import MonteCarloDropout
from sklearn.metrics import roc_auc_score, average_precision_score

if len(sys.argv) != 3:
    sys.exit(f"Usage: {sys.argv[0]} SEED TAG")
seed = int(sys.argv[1]); tag = sys.argv[2]

np.random.seed(seed); tf.random.set_seed(seed)
gen = MIMICDataGenerator(500, 240, 20, 0.236, random_seed=seed)
X, y, _ = gen.generate()
pre = MIMICPreprocessor(random_seed=seed)
Xtr, Xval, Xte, ytr, yval, yte = pre.split(X, y)
Xtr_n, Xval_n, Xte_n = pre.normalize(Xtr, Xval, Xte)

es = callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
tf.random.set_seed(seed); np.random.seed(seed)
full = SepsisNetModel.build_model(input_shape=(240, 20))
full.fit(Xtr_n, ytr, validation_data=(Xval_n, yval),
         epochs=15, batch_size=32, callbacks=[es], verbose=0)
p_det = full.predict(Xte_n, verbose=0).ravel()
mcd = MonteCarloDropout(full, n_iterations=50)
p_mc, std, _ = mcd.predict_with_uncertainty(Xte_n)

tf.random.set_seed(seed); np.random.seed(seed)
lstm = SepsisNetModel.build_model(input_shape=(240, 20), conv_filters=())
lstm.fit(Xtr_n, ytr, validation_data=(Xval_n, yval),
         epochs=15, batch_size=32, callbacks=[es], verbose=0)
p_lstm = lstm.predict(Xte_n, verbose=0).ravel()

result = {
    "seed": seed, "run_tag": tag,
    "CNN_LSTM_deterministic": {
        "AUROC": round(float(roc_auc_score(yte, p_det)), 6),
        "AUPRC": round(float(average_precision_score(yte, p_det)), 6)},
    "CNN_LSTM_MC": {
        "AUROC": round(float(roc_auc_score(yte, p_mc)), 6),
        "AUPRC": round(float(average_precision_score(yte, p_mc)), 6)},
    "LSTM_only": {
        "AUROC": round(float(roc_auc_score(yte, p_lstm)), 6),
        "AUPRC": round(float(average_precision_score(yte, p_lstm)), 6)},
}

out_base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", f"determinism_check_seed{seed}_{tag}")
json.dump(result, open(out_base + ".json", "w"), indent=2)
np.savez(out_base + ".npz",
         yte=yte, p_det=p_det, p_mc=p_mc, std=std, p_lstm=p_lstm)
print(json.dumps(result))
print(f"Saved to {out_base}.json and {out_base}.npz")
