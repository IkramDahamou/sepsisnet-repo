# SepsisNet — code release for Revision 4

Code, tests and scripts for *"An Uncertainty-Aware Deep Learning Baseline for
Synthetic ICU Sepsis-Status Classification with Structured Alert Escalation"*
(Dahamou & Daoui). **No real patient data are used.** This is a methodological
baseline on synthetic data, not a clinically validated system.

## Contents

| Path | Purpose |
|---|---|
| `sepsisnet/src/` | Generator, model, MC Dropout (BatchNorm in inference mode), escalation rule, Petri Net |
| `tests/` | pytest suite (MC Dropout / BatchNorm behaviour, escalation rule) |
| `scripts/` | Runners and analysis scripts (below) |
| `reference/` | Previous (Revision 3) `uncertainty.py` and `mdp_escalation.py`, used only by equivalence tests |
| `reference_data/` | Revision 3 predictions and multi-seed results, used by tests and the Table 7 check |
| `table9_provenance/` | Scripts and result files behind the previous version's deterministic vs non-deterministic comparison (manuscript Table 9), with SHA-256 hashes (`PROVENANCE.json`) |
| `RELEASE_ARTIFACTS.md` | Trained weights, predictions and results to download (Zenodo) |

## Environment

```
pip install -r requirements.txt     # TF 2.21.0, Keras 3.14.1, NumPy 2.2.6, scikit-learn 1.8.0, ...
```
All results were produced on CPU (Windows 11, Python 3.13.1). Every runner must be
started with these variables set **before** Python starts:
```
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
```
Runners call `keras.utils.set_random_seed(seed)` (Python, NumPy and TensorFlow),
enable TensorFlow op determinism, disable oneDNN and use one thread. Each runner
takes an explicit output folder and refuses to overwrite an existing one.

## Reproducibility statement

On this platform, the QR decomposition used by the Keras orthogonal initialiser
of the LSTM recurrent kernels can return one of two numerically different results
in different processes, and the difference can grow during training. Running each
of the 10 seeds twice in separate processes, 5 seeds were bit-identical and 5
differed: by at most 0.114 in a predicted probability, at most 0.013 in a per-seed
AUROC or AUPRC, at most 0.052 in a per-seed variance ratio, and at most 0.0054 in
any 10-seed mean. Results are therefore reproducible to these tolerances, not
bit-for-bit. The trained weights and predictions of every reported run
(`RELEASE_ARTIFACTS.md`) allow every reported number to be recomputed exactly.

## Commands: every table and figure

```
# 0. tests
python -m pytest tests -v

# 1. 10-seed sweep (Tables 2-5, Figure 3): run a (reported) and run b (reproducibility)
python scripts/run_sweep.py --scheme keras_set_random_seed --tag a
python scripts/run_sweep.py --scheme keras_set_random_seed --tag b
python scripts/summarize_sweep.py --out results/sweep_summary.json --md results/sweep_summary.md

# 2. seed-42 illustrative run (Tables 6-8, Figures 4-9)
python scripts/run_seed42.py --out runs/seed42_pinned_a
python scripts/evaluate_methods.py runs/seed42_pinned_a/predictions.npz --out results/seed42_pinned_metrics.json --md results/seed42_pinned_metrics.md
python scripts/seed42_detailed.py --run runs/seed42_pinned_a --out results/seed42_pinned_detailed.json
python scripts/escalation_analysis.py runs/seed42_pinned_a/predictions.npz --p-key p_mc_corrected --std-key std_corrected --out results/seed42_pinned_escalation_corrected.json

# 3. reproducibility check between two runs
python scripts/compare_runs.py runs/seed42_pinned_a runs/<other_run> --out results/<file>.json

# 4. figures (writes figures/)
python scripts/regenerate_figures_rev4.py

# 5. Table 9 (previous version): provenance only; the non-deterministic sweep is
#    not re-runnable to identical values by construction. See table9_provenance/.
```
With the downloaded artifacts (`RELEASE_ARTIFACTS.md`) placed under `runs/`,
steps 1–2 can be skipped and all tables/figures recomputed from the saved
predictions (`summarize_sweep.py`, `evaluate_methods.py`, `seed42_detailed.py`,
`escalation_analysis.py`, `regenerate_figures_rev4.py`).

## Changes from the previous release (commit c041e0e)

* `uncertainty.py`: MC Dropout no longer switches BatchNorm to batch statistics
  (`bn_mode="inference"` default); the previous behaviour is `bn_mode="batch"`.
* `mdp_escalation.py`: per-stay counter (`stay_ids`), `alpha` parameter; default unchanged.
* All runners seed Python's `random` via `keras.utils.set_random_seed` (the previous
  multi-seed and determinism-check scripts seeded only NumPy and TensorFlow, which is
  not sufficient in Keras 3.14.1) and take explicit output folders.
* `run_authoritative.py`, `run_multiseed_10seed_deterministic.py`,
  `determinism_check_run.py`, `run_multiseed_fast.py` and the previous figure/verification
  scripts are superseded by `run_seed42.py`, `run_sweep.py` (`--tag a/b` replaces the
  determinism check), `regenerate_figures_rev4.py` and the summary scripts.
* README claims corrected: no "Petri Net verification", "deadlock-free", fixed
  "expected results" or "verified reproduction" claims; no `main.py`.
