# SepsisNet

Code release accompanying the manuscript *"An Uncertainty-Aware Deep
Learning Baseline for Synthetic ICU Sepsis-Status Classification with
Structured Alert Escalation"* (Dahamou & Daoui).

This repository contains the complete generator, model, uncertainty,
escalation-policy, and Petri Net code, together with the evaluation and
figure-generation scripts referenced in the manuscript's Data Availability
statement. **The dataset used throughout the paper is entirely synthetic**
and is produced by the generator in this repository (`sepsisnet/src/data_loader.py`);
no PhysioNet-restricted or other real patient data is used or distributed
here.

## Repository layout

```
sepsisnet/
├── sepsisnet/                  # released package
│   ├── src/
│   │   ├── data_loader.py      # synthetic ICU data generator + preprocessor
│   │   ├── models.py           # 1D CNN-BiLSTM model factory
│   │   ├── uncertainty.py      # Monte Carlo Dropout uncertainty estimation
│   │   ├── mdp_escalation.py   # MDP alert-escalation policy (GREEN→RED)
│   │   └── petri_net.py        # Petri Net formal verification of the alert workflow
│   ├── config/config.yaml      # documented hyperparameters (see note below)
│   └── requirements.txt
└── scripts/
    ├── run_authoritative.py                 # single-seed (42) run reproducing every reported number
    ├── regenerate_figures.py                # regenerates the data-driven figures from saved predictions
    ├── run_multiseed_fast.py                # 3-seed variability check (seeds 7, 13, 21)
    ├── run_multiseed_10seed_deterministic.py# 10-seed deterministic sweep (seeds 7-115)
    └── determinism_check_run.py             # bit-for-bit determinism verifier for a single seed
```

## Installation

```bash
git clone https://github.com/ikramdahamou/sepsisnet
cd sepsisnet
pip install -r sepsisnet/requirements.txt
```

Requires Python 3.10+ (the reported numbers in the manuscript were produced
under Python 3.13, TensorFlow 2.21.0, NumPy 2.4, scikit-learn 1.8, CPU-only).
Because TensorFlow's numeric behaviour under fixed seeds is not perfectly
portable across major versions or between CPU and GPU, exact bit-for-bit
reproduction on a different environment is not guaranteed — the manuscript
quantifies this via bootstrap 95% CIs (Section 4.1) and the multi-seed
apparatus (Section 4.10).

The scripts add `sepsisnet/src` to `sys.path` themselves at import time, so
no `PYTHONPATH` configuration is needed. **Run every command below from the
repository root** — outputs are written relative to the current working
directory.

## Reproducing the reported results

```bash
# 1. Run the authoritative pipeline (seed 42, ~20-30 min on CPU).
#    Trains the full CNN-BiLSTM and the LSTM-only ablation, runs 100-pass
#    MC Dropout, bootstraps 95% CIs, computes calibration, operating points,
#    SOFA/qSOFA baselines, uncertainty error-detection, alpha-sensitivity,
#    and the escalation-policy simulation.
python scripts/run_authoritative.py
# Writes ./revision3_results.json and ./real_pred_full.npz

# 2. Regenerate the data-driven figures from the saved predictions.
mkdir -p figures
python scripts/regenerate_figures.py
# Writes figures/fig2_cnn_lstm.{png,pdf}, fig4a_roc_curve.{png,pdf},
#        fig4b_pr_curve.{png,pdf}, fig5_uncertainty.{png,pdf},
#        fig6_error_detection.{png,pdf}, fig7_alpha_sensitivity.{png,pdf},
#        fig8_calibration_ci.{png,pdf}
# Note: the Petri Net schematic (fig3 in the manuscript) is a static diagram
# of the model in src/petri_net.py, not a data-driven plot, and is not
# reproduced by this script.

# 3. (Optional) Multi-seed variability.
python scripts/run_multiseed_fast.py                       # fast, 3 extra seeds
PYTHONHASHSEED=0 python scripts/run_multiseed_10seed_deterministic.py  # full 10-seed deterministic sweep

# 4. (Optional) Verify bit-for-bit determinism on a single seed.
PYTHONHASHSEED=0 python scripts/determinism_check_run.py 13 a
PYTHONHASHSEED=0 python scripts/determinism_check_run.py 13 b
python -c "import numpy as np; a=np.load('determinism_check_seed13_a.npz'); b=np.load('determinism_check_seed13_b.npz'); print('bit-for-bit identical:', all(np.array_equal(a[k], b[k]) for k in a.files))"
```

On Windows PowerShell, set the environment variable separately instead of
inline, e.g. `$env:PYTHONHASHSEED = "0"; python scripts/determinism_check_run.py 13 a`.

## Notes on the released configuration

- `sepsisnet/config/config.yaml` documents the intended architecture and
  training hyperparameters, but the scripts under `scripts/` do not load it
  — they set hyperparameters directly in code (this matches how every
  number in the manuscript was actually produced). One value in the YAML is
  stale: it lists `dropout_rate: 0.5`, while the model actually trained
  (`sepsisnet/src/models.py`, and every run script) uses `dropout_rate=0.4`.
- The synthetic generator (`sepsisnet/src/data_loader.py`) is seeded
  end-to-end via `numpy.random.default_rng(seed)`; the default pipeline
  seed is 42.

## Citation

If you use this code, please cite the manuscript (full citation to be
added once the DOI is assigned).

## Contact

Ikram Dahamou (corresponding author) — `dahamou.ikram@usms.ac.ma`
Faculty of Science and Technology, Sultan Moulay Slimane University, Beni
Mellal, Morocco.
