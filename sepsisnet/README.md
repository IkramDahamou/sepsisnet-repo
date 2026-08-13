# sepsisnet (package)

ICU sepsis detection using Deep Learning + Uncertainty Quantification + Formal Verification.

## Architecture

```
Input (batch, 240, 20)
  ↓ Conv1D(64) + BN + Dropout(0.4)
  ↓ Conv1D(128) + BN + Dropout(0.4)
  ↓ Conv1D(256) + BN + Dropout(0.4)
  ↓ BiLSTM(128, return_seq=True) + Dropout(0.4)
  ↓ BiLSTM(64, return_seq=False) + Dropout(0.4)
  ↓ Dense(256, relu) + Dropout(0.4)
  ↓ Dense(128, relu) + Dropout(0.4)
  ↓ Dense(1, sigmoid)  →  P(sepsis)
```

## Modules

| File | Description |
|------|-------------|
| `src/data_loader.py` | Synthetic MIMIC-III-like generator + preprocessor |
| `src/models.py` | 1D CNN-BiLSTM model factory |
| `src/uncertainty.py` | Monte Carlo Dropout uncertainty estimation |
| `src/mdp_escalation.py` | MDP alert escalation (GREEN→RED) |
| `src/petri_net.py` | Petri Net formal verification |

## Usage

This package is not run standalone — see the top-level
[`README.md`](../README.md) for the actual reproduction pipeline
(`../scripts/run_authoritative.py`), which imports these modules directly.

## Expected results (seed 42, full pipeline)

| Metric | Value |
|--------|-------|
| AUROC (CNN-BiLSTM + MC Dropout) | ≈ 0.877 |
| AUPRC (CNN-BiLSTM + MC Dropout) | ≈ 0.716 |
| Uncertainty variance ratio (sepsis/normal) | ≈ 2.81× |
| Deadlock-free | YES |
| Bounded | YES |

See the manuscript for the full results table and confidence intervals.
