"""
SepsisNet: 1D CNN-BiLSTM model for ICU sepsis detection.

Architecture overview
─────────────────────
Input (batch, T, F)
→ 3× Conv1D + BatchNorm + Dropout(0.4)       [local pattern extraction]
→ 2× Bidirectional LSTM + Dropout(0.4)        [temporal dependencies]
→ 2× Dense + Dropout(0.4)                     [classification head]
→ Dense(1, sigmoid)                            [P(sepsis)]

Dropout stays active during inference to enable Monte Carlo uncertainty
estimation (training=True forward passes).
"""
from __future__ import annotations

import logging
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Model, layers, metrics, optimizers, regularizers

logger = logging.getLogger(__name__)


class SepsisNetModel:
    """Factory for building and compiling the SepsisNet Keras model."""

    @staticmethod
    def build_model(
        input_shape: Tuple[int, int] = (240, 20),
        conv_filters: Tuple[int, ...] = (64, 128, 256),
        lstm_units: Tuple[int, ...] = (128, 64),
        dense_units: Tuple[int, ...] = (256, 128),
        dropout_rate: float = 0.4,
        learning_rate: float = 0.001,
        l2_reg: float = 1e-4,
    ) -> tf.keras.Model:
        """
        Build and compile the SepsisNet model.

        Parameters
        ----------
        input_shape : (timesteps, features)
        conv_filters : filters for each Conv1D block
        lstm_units   : units for each BiLSTM layer
        dense_units  : units for each Dense layer
        dropout_rate : applied after every sub-block
        learning_rate: Adam optimizer LR

        Returns
        -------
        Compiled tf.keras.Model
        """
        reg = regularizers.l2(l2_reg)
        inp = tf.keras.Input(shape=input_shape, name="input")
        x = inp

        # ── CNN blocks ──────────────────────────────────────────────
        for i, n_filters in enumerate(conv_filters):
            x = layers.Conv1D(
                filters=n_filters,
                kernel_size=3,
                padding="same",
                activation="relu",
                kernel_regularizer=reg,
                name=f"conv1d_{i + 1}",
            )(x)
            x = layers.BatchNormalization(name=f"bn_{i + 1}")(x)
            # training=True in call() keeps this active -> MC Dropout
            x = layers.Dropout(rate=dropout_rate, name=f"drop_conv_{i + 1}")(x)

        # ── BiLSTM blocks ────────────────────────────────────────────
        for i, n_units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            x = layers.Bidirectional(
                layers.LSTM(
                    n_units,
                    return_sequences=return_sequences,
                    dropout=0.0,
                    recurrent_dropout=0.0,
                    kernel_regularizer=reg,
                    name=f"lstm_{i + 1}",
                ),
                name=f"bilstm_{i + 1}",
            )(x)
            x = layers.Dropout(rate=dropout_rate, name=f"drop_lstm_{i + 1}")(x)

        # ── Dense blocks ─────────────────────────────────────────────
        for i, n_units in enumerate(dense_units):
            x = layers.Dense(n_units, activation="relu",
                             kernel_regularizer=reg, name=f"dense_{i + 1}")(x)
            x = layers.Dropout(rate=dropout_rate, name=f"drop_dense_{i + 1}")(x)

        # ── Output ───────────────────────────────────────────────────
        out = layers.Dense(1, activation="sigmoid", name="output")(x)

        model = Model(inputs=inp, outputs=out, name="SepsisNet")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=[
                metrics.AUC(name="auc"),
                metrics.BinaryAccuracy(name="accuracy"),
            ],
        )

        n_params = model.count_params()
        logger.info("SepsisNet built — %s parameters", f"{n_params:,}")
        return model
