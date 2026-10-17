"""
Monte Carlo Dropout uncertainty estimation for SepsisNet.

By passing training=True during inference, Dropout layers remain active,
producing stochastic predictions whose variance encodes epistemic uncertainty.
Sepsis patients near the decision boundary exhibit higher variance than
confidently-classified normal patients — the variance ratio measures this.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run n_iterations stochastic forward passes with Dropout active.

        Parameters
        ----------
        X : shape (n_samples, T, F)

        Returns
        -------
        mean     : shape (n_samples,) — mean prediction P(sepsis)
        std      : shape (n_samples,) — predictive standard deviation
        variance : shape (n_samples,) — predictive variance
        """
        logger.info("Running %d MC Dropout passes …", self.n_iterations)
        X_tensor = tf.constant(X, dtype=tf.float32)
        predictions: list[np.ndarray] = []

        for i in range(self.n_iterations):
            # training=True keeps Dropout active during inference
            y_pred = self.model(X_tensor, training=True)
            predictions.append(y_pred.numpy().flatten())
            if (i + 1) % 25 == 0:
                logger.info("  MC pass %d / %d", i + 1, self.n_iterations)

        preds = np.stack(predictions, axis=0)   # (n_iters, n_samples)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        variance = std ** 2

        logger.info("MC Dropout complete.")
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
