# SepsisNet — source modules

Synthetic ICU sepsis-status classification baseline: 1D CNN-BiLSTM classifier,
Monte Carlo Dropout uncertainty estimation, a simulated rule-based escalation
rule, and a conceptual Petri Net model of the escalation workflow.
**No real patient data are used; this is not a clinical system.**

## Architecture (`src/models.py`)

```
Input (batch, 240, 20)
  ↓ Conv1D(64)  + BatchNorm + Dropout(0.4)
  ↓ Conv1D(128) + BatchNorm + Dropout(0.4)
  ↓ Conv1D(256) + BatchNorm + Dropout(0.4)
  ↓ BiLSTM(128, return_sequences=True) + Dropout(0.4)
  ↓ BiLSTM(64) + Dropout(0.4)
  ↓ Dense(256, relu) + Dropout(0.4)
  ↓ Dense(128, relu) + Dropout(0.4)
  ↓ Dense(1, sigmoid)  →  P(sepsis)
```
753,601 trainable parameters; the BiLSTM-only ablation (`conv_filters=()`) has 382,977.

## Modules

| File | Description |
|------|-------------|
| `src/data_loader.py` | Synthetic MIMIC-III-like generator (not derived from real data) and preprocessor |
| `src/models.py` | 1D CNN-BiLSTM model factory |
| `src/uncertainty.py` | MC Dropout. `bn_mode="inference"` (default): Dropout active, BatchNorm in inference mode. `bn_mode="batch"`: previous behaviour, for comparison only. `seed=` resets the Dropout seed generators. |
| `src/mdp_escalation.py` | Rule-based escalation (`AlertEscalationMDP`; the name is historical — no MDP is solved). `stay_ids=` keeps the hysteresis counter per stay; `alpha=` sets the uncertainty coefficient (default 0.10). |
| `src/petri_net.py` | Conceptual Petri Net of the escalation workflow with single-trace simulation. It performs **no** formal verification (no reachability analysis); its `verify_*` methods are simple single-trace / attribute checks. |

There is no `main.py`; the evaluation is run with the scripts in `../scripts/` (see `../README.md`).
Results are summarised in the manuscript; no target values are claimed here.
