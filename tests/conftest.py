"""Shared fixtures for the Revision 4 tests.

The fixture model is the real SepsisNet architecture (models.py defaults:
3 Conv1D+BN+Dropout blocks, 2 BiLSTM, 2 Dense, dropout 0.4). It is trained for
one epoch on a small synthetic cohort only so that BatchNorm moving statistics
are no longer at their initial values. Nothing produced here is a result.
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
sys.dont_write_bytecode = True

REV4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REV4, "sepsisnet", "src"))
sys.path.insert(0, os.path.join(REV4, "reference"))

import numpy as np
import pytest
import keras
import tensorflow as tf

tf.config.experimental.enable_op_determinism()

from data_loader import MIMICDataGenerator, MIMICPreprocessor
from models import SepsisNetModel


@pytest.fixture(scope="session")
def data():
    X, y, _ = MIMICDataGenerator(120, 240, 20, 0.236, random_seed=0).generate()
    pre = MIMICPreprocessor(random_seed=0)
    Xtr, Xv, Xte, ytr, yv, yte = pre.split(X, y)
    Xtr, Xv, Xte = pre.normalize(Xtr, Xv, Xte)
    return Xtr.astype("float32"), ytr, Xte.astype("float32"), yte


@pytest.fixture(scope="session")
def model(data):
    Xtr, ytr, _, _ = data
    keras.utils.set_random_seed(0)
    m = SepsisNetModel.build_model(input_shape=(240, 20))
    m.fit(Xtr, ytr, epochs=1, batch_size=32, verbose=0)
    return m


def bn_stats(model):
    from uncertainty import batchnorm_layers
    return [(l.moving_mean.numpy().copy(), l.moving_variance.numpy().copy())
            for l in batchnorm_layers(model)]


def stats_equal(a, b):
    return all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
               for x, y in zip(a, b))


def stats_maxdiff(a, b):
    return max(max(float(np.abs(x[0] - y[0]).max()),
                   float(np.abs(x[1] - y[1]).max())) for x, y in zip(a, b))
