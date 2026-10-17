# Files to release with the code (not in git; archive on Zenodo)

**Note:** the trained weights, predictions and result files listed below are archived on **Zenodo**, not committed to the GitHub repository (the repository contains code, tests, reference data and Table 9 provenance only). The manuscript's Data availability statement carries the placeholder "[commit hash and Zenodo DOI to be inserted before submission]".

These files let every number, table and figure be recomputed without retraining. Each run folder contains a `manifest.json` with SHA-256 hashes of its files.

| Folder | Purpose | Files | Weights (MB) | Other (MB) | Total (MB) |
|---|---|---|---|---|---|
| `runs/sweep_a/` | 10-seed sweep, run a (REPORTED results: Tables 2-5, Figure 3) | 92 | 137.9 | 0.6 | 138.5 |
| `runs/sweep_b/` | 10-seed sweep, run b (reproducibility tolerance only, Section 3.7) | 92 | 137.9 | 0.6 | 138.5 |
| `runs/seed42_pinned_a/` | Official seed-42 illustrative run (Tables 6-8, Figures 4-9) | 8 | 13.8 | 0.1 | 13.9 |
| `runs/seed42_pinned_b/` | Replicate of the official seed-42 run | 8 | 13.8 | 0.1 | 13.9 |
| `runs/seed42_a/` | Seed-42 run in the other QR branch (BN diagnostic; branch-gap statement) | 8 | 13.8 | 0.1 | 13.9 |
| `diagnostics/bn_pinned/` | Batch Normalization diagnostic on the official seed-42 weights (stored vs population statistics) | 4 | 0.0 | 0.0 | 0.0 |
| `diagnostics/bn/` | Batch Normalization diagnostic on the earlier seed-42 model (incl. recalibrated predictions) | 7 | 0.0 | 0.1 | 0.1 |
| `results/` | Summary JSON/MD files from which every manuscript number is generated | 26 | 0.0 | 0.4 | 0.4 |
| **Total** | | | **317.2** | **1.9** | **319.2** |

Without weight files (predictions, configs and results only) the archive is 1.9 MB; all metrics can be recomputed from predictions alone, while the weight files are needed to re-run inference (e.g. the MC Dropout comparison) and to verify the weight-level reproducibility figures.

Not to be released: `runs/seed42_unpinned_check_*` (internal check, identical to `seed42_pinned_a`), `diagnostics/seeding/` (A1 seeding diagnostics; may be added as supplementary material if desired).
