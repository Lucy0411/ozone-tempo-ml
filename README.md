# Surface Ozone Modeling with TEMPO Column O3

Code accompanying our study on predicting hourly surface ozone (O3) concentrations
using gap-filled TEMPO satellite column O3 as a predictor, alongside meteorological,
land-use, and emission covariates. Models are trained with XGBoost and evaluated
with random, temporal, and spatial cross-validation.

## Repository structure

```
ozone-tempo-ml/
├── model/          # Train the final XGBoost surface ozone model (TEMPO / noTEMPO)
├── cv/             # Random / temporal / spatial 10-fold cross-validation
└── requirements.txt
```

### `model/`

`train_ozone_model.py` trains an XGBoost regressor on station ozone
observations. Use `--model TEMPO` to include the gap-filled TEMPO column O3
predictor, or `--model noTEMPO` to exclude it (baseline comparison).
Hyperparameters are tuned with Optuna, then a final model is refit on the
combined train+validation set and evaluated on a held-out test set.

```bash
python model/train_ozone_model.py \
  --model TEMPO \
  --input /path/to/final_dataset.csv \
  --output-dir /path/to/output/model_tempo \
  --n-trials 100
```

Outputs: trained model (`xgb_model_*.json`), scaler, metrics, permutation
feature importance, and observed-vs-predicted scatter plots.

### `cv/`

`cv_ozone_xgb.py` runs 10-fold cross-validation under three protocols to
assess robustness:

- **random** — standard shuffled K-fold
- **temporal** — contiguous date blocks (tests forecast-style generalization)
- **spatial** — grouped by station location (tests generalization to unseen sites)

```bash
python cv/cv_ozone_xgb.py \
  --model TEMPO \
  --input /path/to/final_dataset.csv \
  --output-dir /path/to/output/cv_tempo \
  --n-splits 10
```

## Input data

Scripts expect an input CSV with hourly station ozone observations (`ozone`,
in ppb) joined with meteorological (ERA5-derived), GEOS-CF, TEMPO column O3,
land-use, population, and emission-inventory covariates. Station metadata
columns (`Site Num`, `Latitude`, `Longitude`, `Date GMT`, `Time GMT`) are
required for the temporal/spatial CV splits. Due to data-sharing
restrictions, the dataset itself is not included; contact the corresponding
author for access.

## Installation

```bash
pip install -r requirements.txt
```

GPU training is used automatically when available (via PyTorch's CUDA
detection) and can be disabled with `--no-gpu`.

## Citation

If you use this code, please cite the associated publication:

Hang, Y.; Mei, A.; Cui, Z.; North, K.; Highland, H.; Aguilera, J. A.; Liu, Y.;
Liu, X.; Xiao, Q. Geostationary Satellite Observation Informed Modeling of
Ozone Dynamics along the U.S.-Mexico Border. *ACS ES&T Air* **2026**.
https://doi.org/10.1021/acsestair.6c00108

## Contact

Aodong Mei — Aodong.Mei@uth.tmc.edu

## License

MIT License (see `LICENSE`).
