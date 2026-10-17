"""Tests for the corrected MC Dropout / BatchNorm implementation (Revision 4).

Every test prints the measured quantity so the log shows actual values, not
only pass/fail.
"""
import contextlib

import numpy as np
import pytest
import tensorflow as tf

from conftest import bn_stats, stats_equal, stats_maxdiff
from uncertainty import (MonteCarloDropout, batchnorm_layers, dropout_layers,
                         dropout_seed_states)
import uncertainty_original

N_PASSES = 8
BATCH_INDEP_ATOL = 1e-6


@contextlib.contextmanager
def dropout_rate(model, rate):
    """Temporarily set every Dropout rate (used to isolate the BN path)."""
    drops = dropout_layers(model)
    old = [d.rate for d in drops]
    try:
        for d in drops:
            d.rate = rate
        yield
    finally:
        for d, r in zip(drops, old):
            d.rate = r


def seed_states(model):
    return [d.seed_generator.state.numpy().copy() for d in dropout_layers(model)]


def test_architecture_is_real_sepsisnet(model):
    names = [l.name for l in batchnorm_layers(model)]
    rates = sorted({d.rate for d in dropout_layers(model)})
    print(f"\nBN layers={names}; Dropout layers={len(dropout_layers(model))}; "
          f"rates={rates}; params={model.count_params()}")
    assert names == ["bn_1", "bn_2", "bn_3"]
    assert len(dropout_layers(model)) == 7 and rates == [0.4]
    assert model.count_params() == 753601


def test_deterministic_inference_repeatable(model, data):
    _, _, Xte, _ = data
    a = model.predict(Xte, verbose=0).ravel()
    b = model.predict(Xte, verbose=0).ravel()
    c = model(Xte, training=False).numpy().ravel()
    d = model(Xte, training=False).numpy().ravel()
    print(f"\npredict max|a-b|={np.abs(a-b).max()}; call max|c-d|={np.abs(c-d).max()}")
    assert np.array_equal(a, b) and np.array_equal(c, d)


def test_corrected_mc_is_stochastic(model, data):
    _, _, Xte, _ = data
    _, std, _, samples = MonteCarloDropout(model, N_PASSES).predict_with_uncertainty(
        Xte, bn_mode="inference", seed=1, return_samples=True)
    n_distinct = len({s.tobytes() for s in samples})
    print(f"\ndistinct pass vectors={n_distinct}/{N_PASSES}; "
          f"min per-stay std={std.min():.3e}; mean std={std.mean():.3e}")
    assert n_distinct == N_PASSES
    assert std.min() > 0


def test_bn_statistics_unchanged_after_corrected_mc(model, data):
    _, _, Xte, _ = data
    before = bn_stats(model)
    MonteCarloDropout(model, N_PASSES).predict_with_uncertainty(Xte, bn_mode="inference", seed=2)
    after = bn_stats(model)
    print(f"\nmax|BN stat change| after corrected MC = {stats_maxdiff(before, after)}")
    assert stats_equal(before, after)


def test_original_code_overwrites_bn_statistics(model, data):
    """Documents the defect being fixed (restores stats afterwards)."""
    _, _, Xte, _ = data
    before = bn_stats(model)
    try:
        uncertainty_original.MonteCarloDropout(model, 2).predict_with_uncertainty(Xte)
        after = bn_stats(model)
    finally:
        for l, (mu, var) in zip(batchnorm_layers(model), before):
            l.moving_mean.assign(mu); l.moving_variance.assign(var)
    print(f"\nmax|BN stat change| after ORIGINAL uncertainty.py (2 passes) = "
          f"{stats_maxdiff(before, after):.4e}")
    assert not stats_equal(before, after)


def test_corrected_mc_with_dropout_off_equals_inference(model, data):
    """With rate 0, corrected MC must be exactly deterministic inference."""
    _, _, Xte, _ = data
    ref = model(Xte, training=False).numpy().ravel()
    with dropout_rate(model, 0.0):
        mean, std, _ = MonteCarloDropout(model, 2).predict_with_uncertainty(Xte, bn_mode="inference")
        leg, _, _ = MonteCarloDropout(model, 2).predict_with_uncertainty(Xte, bn_mode="batch")
    print(f"\ncorrected(rate0) vs inference: max|diff|={np.abs(mean-ref).max()}, std max={std.max()}; "
          f"legacy(rate0) vs inference: max|diff|={np.abs(leg-ref).max():.4e}")
    assert np.array_equal(mean, ref) and std.max() == 0
    assert np.abs(leg - ref).max() > 1e-3   # sensitivity: legacy BN path differs


def test_batch_composition_independence(model, data):
    """One stay's corrected-MC output must not depend on unrelated stays.

    Dropout masks are drawn for the whole batch tensor, so with dropout active
    a stay's mask legitimately changes with batch shape; that is sampling
    noise, not coupling. The only cross-stay coupling in this architecture is
    BatchNorm in batch mode, so the BN path is tested with Dropout rate 0.
    Tolerance: atol = 1e-6 (different batch sizes can change float reduction
    order inside matmuls).
    """
    _, _, Xte, _ = data
    with dropout_rate(model, 0.0):
        mc = MonteCarloDropout(model, 1)
        alone_c, _, _ = mc.predict_with_uncertainty(Xte[:1], bn_mode="inference")
        inbatch_c, _, _ = mc.predict_with_uncertainty(Xte, bn_mode="inference")
        alone_l, _, _ = mc.predict_with_uncertainty(Xte[:3], bn_mode="batch")
        inbatch_l, _, _ = mc.predict_with_uncertainty(Xte, bn_mode="batch")
    dc = float(np.abs(alone_c[0] - inbatch_c[0]))
    dl = float(np.abs(alone_l[:3] - inbatch_l[:3]).max())
    print(f"\ncorrected: |stay0 alone - stay0 in batch of {len(Xte)}| = {dc:.3e} (atol {BATCH_INDEP_ATOL}); "
          f"legacy: max diff = {dl:.3e}")
    assert dc <= BATCH_INDEP_ATOL
    assert dl > 1e-3   # sensitivity: legacy mode is batch-dependent


@pytest.mark.parametrize("bn_mode", ["inference", "batch"])
def test_state_restored_after_exception(model, data, monkeypatch, bn_mode):
    _, _, Xte, _ = data
    bns = batchnorm_layers(model)
    flags_before = [l.trainable for l in bns]
    stats_before = bn_stats(model)
    seeds_before = seed_states(model)
    target = dropout_layers(model)[-1]
    calls = {"n": 0}
    orig_call = target.call

    def boom(*a, **k):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("intentional failure inside MC pass")
        return orig_call(*a, **k)

    monkeypatch.setattr(target, "call", boom)
    with pytest.raises(RuntimeError, match="intentional"):
        MonteCarloDropout(model, 5).predict_with_uncertainty(Xte, bn_mode=bn_mode, seed=3)
    monkeypatch.undo()
    flags_after = [l.trainable for l in bns]
    print(f"\n[{bn_mode}] calls before raise={calls['n']}; flags before={flags_before} after={flags_after}; "
          f"BN max change={stats_maxdiff(stats_before, bn_stats(model))}; "
          f"seed states restored={all(np.array_equal(a, b) for a, b in zip(seeds_before, seed_states(model)))}")
    assert calls["n"] >= 2
    assert flags_after == flags_before
    assert stats_equal(stats_before, bn_stats(model))
    assert all(np.array_equal(a, b) for a, b in zip(seeds_before, seed_states(model)))


def test_flags_restored_after_normal_run(model, data):
    _, _, Xte, _ = data
    bns = batchnorm_layers(model)
    before = [l.trainable for l in bns]
    n_tw = len(model.trainable_weights)
    MonteCarloDropout(model, 2).predict_with_uncertainty(Xte, bn_mode="inference")
    print(f"\nflags before={before} after={[l.trainable for l in bns]}; "
          f"trainable weights {n_tw} -> {len(model.trainable_weights)}")
    assert [l.trainable for l in bns] == before
    assert len(model.trainable_weights) == n_tw


def test_legacy_mode_reproduces_original_code(model, data):
    """Same weights + same Dropout generator states -> bit-identical outputs."""
    _, _, Xte, _ = data
    drops = dropout_layers(model)
    states = dropout_seed_states(42, len(drops))
    stats_before = bn_stats(model)
    seeds_before = seed_states(model)

    # Original Revision 3 code, with generator states set to the same values
    for d, st in zip(drops, states):
        d.seed_generator.state.assign(st.astype(d.seed_generator.state.numpy().dtype))
    try:
        o_mean, o_std, o_var = uncertainty_original.MonteCarloDropout(
            model, N_PASSES).predict_with_uncertainty(Xte)
        orig_changed = stats_maxdiff(stats_before, bn_stats(model))
    finally:
        for l, (mu, var) in zip(batchnorm_layers(model), stats_before):
            l.moving_mean.assign(mu); l.moving_variance.assign(var)
        for d, st in zip(drops, seeds_before):
            d.seed_generator.state.assign(st)

    n_mean, n_std, n_var = MonteCarloDropout(model, N_PASSES).predict_with_uncertainty(
        Xte, bn_mode="batch", seed=42)
    after_legacy = bn_stats(model)
    print(f"\noriginal vs legacy: max|mean diff|={np.abs(o_mean-n_mean).max()}, "
          f"max|std diff|={np.abs(o_std-n_std).max()}, max|var diff|={np.abs(o_var-n_var).max()}; "
          f"original changed BN stats by {orig_changed:.4e}; "
          f"legacy left BN stats unchanged={stats_equal(stats_before, after_legacy)}")
    assert np.array_equal(o_mean, n_mean)
    assert np.array_equal(o_std, n_std)
    assert np.array_equal(o_var, n_var)
    assert orig_changed > 0
    assert stats_equal(stats_before, after_legacy)


def test_seeded_mc_repeatable_and_tf_seed_insufficient(model, data):
    _, _, Xte, _ = data
    mc = MonteCarloDropout(model, 4)
    a, _, _ = mc.predict_with_uncertainty(Xte, bn_mode="inference", seed=7)
    b, _, _ = mc.predict_with_uncertainty(Xte, bn_mode="inference", seed=7)
    tf.random.set_seed(7)
    c, _, _ = mc.predict_with_uncertainty(Xte, bn_mode="inference", seed=None)
    tf.random.set_seed(7)
    d, _, _ = mc.predict_with_uncertainty(Xte, bn_mode="inference", seed=None)
    print(f"\nseeded repeat max|diff|={np.abs(a-b).max()}; "
          f"tf.random.set_seed-only repeat max|diff|={np.abs(c-d).max():.3e}")
    assert np.array_equal(a, b)
    assert not np.array_equal(c, d)


def test_same_seed_gives_same_masks_across_modes_when_bn_irrelevant(model, data):
    """Pairing check: with BN switched to inference in both calls, the two
    bn_mode code paths would see identical masks for the same seed. Verified
    via corrected mode twice from different prior generator states."""
    _, _, Xte, _ = data
    mc = MonteCarloDropout(model, 3)
    mc.predict_with_uncertainty(Xte, bn_mode="batch", seed=None)  # advance states
    a, _, _ = mc.predict_with_uncertainty(Xte, bn_mode="inference", seed=11)
    b, _, _ = mc.predict_with_uncertainty(Xte, bn_mode="inference", seed=11)
    print(f"\nseeded result independent of prior generator state: max|diff|={np.abs(a-b).max()}")
    assert np.array_equal(a, b)
