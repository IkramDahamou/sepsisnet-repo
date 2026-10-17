"""
Monte Carlo Dropout uncertainty estimation for SepsisNet (Revision 4).

Dropout layers are kept stochastic at inference by calling the model with
training=True. In Keras 3, training=True ALSO switches BatchNormalization to
batch statistics and overwrites its moving statistics (BatchNormalization.call
branches on `training and self.trainable`). Revision 4 therefore separates
the two layer types explicitly:

  bn_mode="inference" (default, primary method)
      Every BatchNormalization layer is set to trainable=False for the duration
      of the MC passes, so it normalises with its stored moving statistics,
      never updates them, and each stay's output does not depend on the other
      stays in the batch. Dropout stays stochastic. Original flags are restored
      in a try/finally.

  bn_mode="batch" (legacy, comparison only)
      Reproduces Revision 3 behaviour (BN uses batch statistics). The BN moving
      statistics are saved before the passes and restored afterwards, so the
      model is left unchanged.

Seeding: Keras 3 Dropout draws its masks with stateless ops seeded from a
per-layer SeedGenerator counter; tf.random.set_seed() does not affect them.
When `seed` is given, every Dropout layer's generator state is reset to a
value derived from `seed` before the passes (so both bn_modes see the same
masks on the same weights) and restored afterwards.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

logger = logging.getLogger(__name__)

BN_MODES = ("inference", "batch")


def _all_layers(model: tf.keras.Model) -> List[tf.keras.layers.Layer]:
    try:
        return list(model._flatten_layers(include_self=False, recursive=True))
    except AttributeError:  # pragma: no cover - older Keras
        return list(model.layers)


def batchnorm_layers(model: tf.keras.Model) -> List[layers.BatchNormalization]:
    return [l for l in _all_layers(model)
            if isinstance(l, layers.BatchNormalization)]


def dropout_layers(model: tf.keras.Model) -> List[layers.Dropout]:
    return [l for l in _all_layers(model) if isinstance(l, layers.Dropout)]


def dropout_seed_states(seed: int, n_layers: int) -> np.ndarray:
    """Per-Dropout-layer generator states [s_i, 0] derived from one seed."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(1, 2**31 - 2, size=n_layers, dtype=np.int64)
    return np.stack([starts, np.zeros(n_layers, dtype=np.int64)], axis=1)


class MonteCarloDropout:
    """
    Uncertainty estimation via repeated stochastic forward passes.

    Parameters
    ----------
    model        : trained tf.keras.Model with Dropout layers
    n_iterations : number of MC forward passes (default 100)
    """

    def __init__(self, model: tf.keras.Model, n_iterations: int = 100) -> None:
        self.model = model
        self.n_iterations = n_iterations

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        bn_mode: str = "inference",
        seed: int | None = None,
        return_samples: bool = False,
    ) -> Tuple[np.ndarray, ...]:
        """
        Run n_iterations stochastic forward passes with Dropout active.

        Parameters
        ----------
        X              : shape (n_samples, T, F)
        bn_mode        : "inference" (primary) or "batch" (legacy comparison)
        seed           : if given, Dropout generator states are reset from it
        return_samples : also return the (n_iterations, n_samples) pass matrix

        Returns
        -------
        mean, std, variance  (each shape (n_samples,)) [, samples]
        """
        if bn_mode not in BN_MODES:
            raise ValueError(f"bn_mode must be one of {BN_MODES}, got {bn_mode!r}")

        bns = batchnorm_layers(self.model)
        drops = dropout_layers(self.model)
        saved_flags = [l.trainable for l in bns]
        saved_stats = [(l.moving_mean.numpy().copy(),
                        l.moving_variance.numpy().copy()) for l in bns]
        saved_seed_states = [l.seed_generator.state.numpy().copy()
                             for l in drops if hasattr(l, "seed_generator")]

        logger.info("Running %d MC Dropout passes (bn_mode=%s, seed=%s) …",
                    self.n_iterations, bn_mode, seed)
        X_tensor = tf.constant(X, dtype=tf.float32)
        predictions: list[np.ndarray] = []
        try:
            if seed is not None:
                states = dropout_seed_states(seed, len(drops))
                for l, st in zip(drops, states):
                    if hasattr(l, "seed_generator"):
                        cur = l.seed_generator.state.numpy()
                        l.seed_generator.state.assign(st.astype(cur.dtype))
            if bn_mode == "inference":
                for l in bns:
                    l.trainable = False
            for i in range(self.n_iterations):
                # training=True keeps Dropout active; BN follows bn_mode
                y_pred = self.model(X_tensor, training=True)
                predictions.append(y_pred.numpy().flatten())
                if (i + 1) % 25 == 0:
                    logger.info("  MC pass %d / %d", i + 1, self.n_iterations)
        finally:
            for l, flag in zip(bns, saved_flags):
                l.trainable = flag
            if bn_mode == "batch":
                for l, (mu, var) in zip(bns, saved_stats):
                    l.moving_mean.assign(mu)
                    l.moving_variance.assign(var)
            if seed is not None:
                k = 0
                for l in drops:
                    if hasattr(l, "seed_generator"):
                        l.seed_generator.state.assign(saved_seed_states[k])
                        k += 1

        preds = np.stack(predictions, axis=0)   # (n_iters, n_samples)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        variance = std ** 2

        logger.info("MC Dropout complete.")
        if return_samples:
            return mean, std, variance, preds
        return mean, std, variance

    def compute_calibration_metrics(
        self,
        variance: np.ndarray,
        y_true: np.ndarray,
    ) -> dict:
        """
        Compute variance-based calibration metrics.

        Returns dict with:
          sepsis_variance, normal_variance, variance_ratio
        """
        sepsis_var = float(variance[y_true == 1].mean())
        normal_var = float(variance[y_true == 0].mean())
        ratio = sepsis_var / normal_var if normal_var > 1e-10 else float("inf")
        return {
            "sepsis_variance": sepsis_var,
            "normal_variance": normal_var,
            "variance_ratio": ratio,
        }
