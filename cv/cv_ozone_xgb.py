#!/usr/bin/env python3
"""Random / temporal / spatial 10-fold cross-validation for the surface ozone
XGBoost models (TEMPO and noTEMPO feature sets).

CV protocols
------------
1. random:   standard shuffled KFold.
2. temporal: dates sorted and split into N contiguous date blocks.
3. spatial:  GroupKFold grouped by station location (Site + Latitude + Longitude).

Example
-------
python cv_ozone_xgb.py --model TEMPO --input final_dataset.csv --output-dir cv_tempo --n-splits 10
python cv_ozone_xgb.py --model noTEMPO --input final_dataset.csv --output-dir cv_notempo --n-splits 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor


TARGET = "ozone"
RANDOM_STATE = 42
N_SPLITS = 10
HIGH_WEIGHT_QUANTILE = 0.97
HIGH_WEIGHT = 3.0

COMMON_FEATURES = [
    "t2m", "sp", "u10", "v10", "humidity", "tcc", "hcc", "lcc", "mcc", "vis",
    "veg", "mstav", "sr", "sh2", "lhtfl", "shtfl", "dlwrf", "O3",
    "population", "road_dis_km", "NTL_Radiance", "BC", "NH3", "agriculture",
    "forest", "grassland", "wetland", "urban", "bare", "water",
]
FEATURES_BY_MODEL = {
    "TEMPO": [
        "t2m", "sp", "u10", "v10", "humidity", "tcc", "hcc", "lcc", "mcc", "vis",
        "veg", "mstav", "sr", "sh2", "lhtfl", "shtfl", "dlwrf", "column_amount_o3",
        "O3", "population", "road_dis_km", "NTL_Radiance", "BC", "NH3", "agriculture",
        "forest", "grassland", "wetland", "urban", "bare", "water",
    ],
    "noTEMPO": COMMON_FEATURES,
}

# Best hyperparameters found via Optuna tuning (see the model/ training script).
BEST_PARAMS = {
    "TEMPO": {
        "learning_rate": 0.03717737029030374,
        "max_depth": 15,
        "min_child_weight": 52,
        "subsample": 0.9512931329577369,
        "colsample_bytree": 0.7763683358720174,
        "reg_alpha": 0.6151794129810663,
        "reg_lambda": 0.23229476721807496,
    },
    "noTEMPO": {
        "learning_rate": 0.05055914716388915,
        "max_depth": 14,
        "min_child_weight": 21,
        "subsample": 0.9550837928984293,
        "colsample_bytree": 0.8862077221731585,
        "reg_alpha": 0.001961782350122121,
        "reg_lambda": 0.02094436231205362,
    },
}


def normalize_model_name(name: str) -> str:
    lowered = name.strip().lower()
    if lowered == "tempo":
        return "TEMPO"
    if lowered in {"notempo", "no_tempo", "no-tempo"}:
        return "noTEMPO"
    raise ValueError(f"Unknown model name {name!r}. Use TEMPO or noTEMPO.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run random/temporal/spatial 10-fold CV.")
    parser.add_argument("--model", default="TEMPO", help="TEMPO or noTEMPO.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional random sample for quick tests.")
    parser.add_argument("--gpu-id", default=None, help="Set CUDA_VISIBLE_DEVICES before training.")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=3000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--protocols", nargs="+", default=["random", "temporal", "spatial"], choices=["random", "temporal", "spatial"])
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "cv_training_pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Log file: %s", log_file)


def xgb_device(no_gpu: bool) -> str:
    if no_gpu:
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def make_timestamp(df: pd.DataFrame) -> pd.Series:
    text = df["Date GMT"].astype(str).str.strip() + " " + df["Time GMT"].astype(str).str.strip()
    parsed = pd.to_datetime(text, errors="coerce", infer_datetime_format=True)
    if parsed.isna().all():
        parsed = pd.to_datetime(text, errors="coerce")
    return parsed


def make_spatial_group(df: pd.DataFrame) -> pd.Series:
    site_col = "Site Num" if "Site Num" in df.columns else "Site"
    return (
        df[site_col].astype(str).str.strip()
        + "|"
        + pd.to_numeric(df["Latitude"], errors="coerce").round(6).astype(str)
        + "|"
        + pd.to_numeric(df["Longitude"], errors="coerce").round(6).astype(str)
    )


def load_data(input_path: Path, features: list[str], sample_rows: int | None) -> pd.DataFrame:
    needed = list(dict.fromkeys(features + [TARGET, "Site Num", "Latitude", "Longitude", "Date GMT", "Time GMT"]))
    logging.info("Loading data: %s", input_path)
    df = pd.read_csv(input_path, usecols=lambda c: c in needed, low_memory=False)
    logging.info("Loaded rows=%s columns=%s", len(df), len(df.columns))

    missing = [c for c in features + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if sample_rows is not None and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=RANDOM_STATE).copy()
        logging.info("Using sampled rows=%s", len(df))

    df["_timestamp"] = make_timestamp(df)
    logging.info("Timestamp parse success: %s/%s rows", int(df["_timestamp"].notna().sum()), len(df))
    df["_date"] = df["_timestamp"].dt.date.astype(str)
    df["_spatial_group"] = make_spatial_group(df)

    for col in features + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    required_subset = features + [TARGET, "_timestamp", "_spatial_group"]
    missing_fraction = df[required_subset].isna().mean().sort_values(ascending=False)
    logging.info("Top missing fractions before dropna:\n%s", missing_fraction.head(20).to_string())
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required_subset).copy()
    logging.info("Dropped rows with missing required values: %s", before - len(df))

    before = len(df)
    df = df[df[TARGET] >= 0].copy()
    logging.info("Dropped rows with ozone < 0: %s", before - len(df))
    logging.info("Clean rows=%s", len(df))
    if len(df) < 10:
        raise ValueError(f"Only {len(df)} clean rows remain after filtering. Check input columns and timestamp parsing.")
    return df


def model_params(model_name: str, device: str, n_estimators: int, early_stopping_rounds: int) -> dict:
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "booster": "gbtree",
        "tree_method": "hist",
        "device": device,
        "verbosity": 0,
        "n_estimators": n_estimators,
        "early_stopping_rounds": early_stopping_rounds,
    }
    params.update(BEST_PARAMS[model_name])
    return params


def make_high_weights(y: pd.Series) -> np.ndarray:
    high = y.quantile(HIGH_WEIGHT_QUANTILE)
    weights = np.ones(len(y), dtype=np.float32)
    weights[y.to_numpy() > high] = HIGH_WEIGHT
    logging.info("High-ozone weights: high>%.3f ppb weight=%.1f weighted_rows=%s", high, HIGH_WEIGHT, int((weights > 1).sum()))
    return weights


def metrics_row(protocol: str, fold: int, y_true: pd.Series, y_pred: np.ndarray, **extra) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    rse = float(np.sqrt(np.sum((y_true.to_numpy() - y_pred) ** 2) / np.sum((y_true.to_numpy() - y_true.mean()) ** 2)))
    row = {"protocol": protocol, "fold": fold, "n": int(len(y_true)), "r2": r2, "rmse": rmse, "mae": mae, "rse": rse}
    row.update(extra)
    return row


def fit_predict_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    features: list[str],
    params: dict,
) -> np.ndarray:
    scaler = MinMaxScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=features, index=X_val.index)
    weights = make_high_weights(y_train)
    model = XGBRegressor(**params)
    model.fit(X_train_s, y_train, sample_weight=weights, eval_set=[(X_val_s, y_val)], verbose=False)
    return model.predict(X_val_s)


def plot_scatter_density(y_true: pd.Series, y_pred: pd.Series, output_path: Path, title: str, max_points: int = 120_000) -> None:
    obs = y_true.to_numpy()
    pred = y_pred.to_numpy()
    if len(obs) > max_points:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(obs), size=max_points, replace=False)
        obs = obs[idx]
        pred = pred[idx]
    lr = LinearRegression().fit(obs.reshape(-1, 1), pred.reshape(-1, 1))
    slope = float(lr.coef_[0][0])
    intercept = float(lr.intercept_[0])
    try:
        xy = np.vstack([obs, pred])
        z = gaussian_kde(xy)(xy)
        order = z.argsort()
        obs, pred, z = obs[order], pred[order], z[order]
    except Exception:
        z = None

    r2 = r2_score(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred))
    mae = mean_absolute_error(obs, pred)
    hi = max(float(np.nanpercentile(np.concatenate([obs, pred]), 99.8) * 1.05), 1.0)

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=300)
    if z is None:
        ax.scatter(obs, pred, s=4, alpha=0.45, edgecolors="none")
    else:
        sc = ax.scatter(obs, pred, c=z, cmap="viridis", s=4, alpha=0.55, edgecolors="none", rasterized=True)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label("Density")
    ax.plot([0, hi], [0, hi], "k--", lw=0.8)
    x_line = np.array([0, hi])
    ax.plot(x_line, intercept + slope * x_line, "r-", lw=0.8)
    ax.text(0.97, 0.04, f"R2={r2:.3f}\nRMSE={rmse:.2f}\nMAE={mae:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.85, "ec": "none"})
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Observed ozone (ppb)")
    ax.set_ylabel("Predicted ozone (ppb)")
    ax.set_title(title)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_protocol_outputs(protocol: str, output_dir: Path, y: pd.Series, oof: pd.Series, fold_metrics: list[dict]) -> dict:
    valid = oof.notna()
    overall = metrics_row(protocol, 0, y.loc[valid], oof.loc[valid].to_numpy(), fold_label="overall")
    logging.info("Overall %s metrics: %s", protocol, overall)

    pd.DataFrame(fold_metrics).to_csv(output_dir / f"cv_fold_metrics_{protocol}.csv", index=False)
    pd.DataFrame({"row_index": oof.loc[valid].index, "observed_ozone_ppb": y.loc[valid].to_numpy(), "predicted_ozone_ppb": oof.loc[valid].to_numpy()}).to_csv(output_dir / f"oof_predictions_{protocol}.csv", index=False)
    plot_scatter_density(y.loc[valid], oof.loc[valid], output_dir / f"scatter_density_{protocol}.png", protocol)
    return overall


def random_splits(df: pd.DataFrame, n_splits: int):
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    yield from splitter.split(df)


def spatial_splits(df: pd.DataFrame, n_splits: int):
    splitter = GroupKFold(n_splits=n_splits)
    yield from splitter.split(df, groups=df["_spatial_group"])


def temporal_splits(df: pd.DataFrame, n_splits: int):
    dates = np.array(sorted(df["_date"].unique()))
    date_chunks = np.array_split(dates, n_splits)
    for chunk in date_chunks:
        val_mask = df["_date"].isin(chunk).to_numpy()
        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        yield train_idx, val_idx


def run_protocol(protocol: str, df: pd.DataFrame, features: list[str], params: dict, n_splits: int, output_dir: Path) -> dict:
    splitters = {"random": random_splits, "temporal": temporal_splits, "spatial": spatial_splits}
    logging.info("=" * 60)
    logging.info("Starting %s 10-fold CV", protocol)
    X = df[features]
    y = df[TARGET]
    oof = pd.Series(index=df.index, dtype=float)
    fold_metrics = []

    for fold, (train_pos, val_pos) in enumerate(splitters[protocol](df, n_splits), start=1):
        train_idx = df.index[train_pos]
        val_idx = df.index[val_pos]
        extra = {}
        if protocol == "temporal":
            val_dates = df.loc[val_idx, "_date"]
            extra["date_from"] = val_dates.min()
            extra["date_to"] = val_dates.max()
        if protocol == "spatial":
            extra["n_validation_site_locations"] = int(df.loc[val_idx, "_spatial_group"].nunique())
        logging.info("%s fold %s/%s train=%s val=%s extra=%s", protocol, fold, n_splits, len(train_idx), len(val_idx), extra)

        pred = fit_predict_fold(X.loc[train_idx], y.loc[train_idx], X.loc[val_idx], y.loc[val_idx], features, params)
        oof.loc[val_idx] = pred
        fold_metrics.append(metrics_row(protocol, fold, y.loc[val_idx], pred, **extra))
        logging.info("%s fold %s metrics: %s", protocol, fold, fold_metrics[-1])

    return save_protocol_outputs(protocol, output_dir, y, oof, fold_metrics)


def main() -> None:
    args = parse_args()
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    model_name = normalize_model_name(args.model)
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)
    logging.info("Model: %s", model_name)
    logging.info("CUDA_VISIBLE_DEVICES: %s", os.environ.get("CUDA_VISIBLE_DEVICES"))

    features = FEATURES_BY_MODEL[model_name]
    params = model_params(model_name, xgb_device(args.no_gpu), args.n_estimators, args.early_stopping_rounds)
    logging.info("Features (%s): %s", len(features), ", ".join(features))
    logging.info("XGB params: %s", params)

    df = load_data(Path(args.input), features, args.sample_rows)
    summary = {
        "model": model_name,
        "input": str(args.input),
        "output_dir": str(output_dir),
        "n_rows": int(len(df)),
        "n_features": int(len(features)),
        "features": features,
        "params": params,
        "protocols": args.protocols,
    }
    overall_rows = []
    for protocol in args.protocols:
        overall_rows.append(run_protocol(protocol, df, features, params, args.n_splits, output_dir))

    pd.DataFrame(overall_rows).to_csv(output_dir / "cv_overall_metrics.csv", index=False)
    (output_dir / "cv_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("CV finished. Outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
