#!/usr/bin/env python3
"""Train an XGBoost surface ozone model, with or without TEMPO column O3 as a
predictor.

Two variants share the same pipeline and only differ in feature set:
  --model TEMPO    includes column_amount_o3 (gap-filled TEMPO column O3)
  --model noTEMPO  excludes column_amount_o3

Input CSV must contain an `ozone` target column (ppb) and all feature
columns listed in FEATURES_BY_MODEL below.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency
    optuna = None


TARGET_COLUMN = "ozone"
RANDOM_STATE = 42

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

FEATURE_NAME_MAPPING = {
    "t2m": "Temperature at 2m",
    "sp": "Surface Pressure",
    "u10": "U Wind at 10m",
    "v10": "V Wind at 10m",
    "humidity": "Relative Humidity",
    "tcc": "Total Cloud Cover",
    "hcc": "High Cloud Cover",
    "lcc": "Low Cloud Cover",
    "mcc": "Middle Cloud Cover",
    "vis": "Visibility",
    "veg": "Vegetation Fraction",
    "mstav": "Moisture Availability",
    "sr": "Surface Roughness",
    "sh2": "Specific Humidity",
    "lhtfl": "Latent Heat Flux",
    "shtfl": "Sensible Heat Flux",
    "dlwrf": "Downward Longwave Radiation",
    "column_amount_o3": "TEMPO Column O3",
    "O3": "GEOS-CF O3",
    "population": "Population Density",
    "road_dis_km": "Distance to Road",
    "NTL_Radiance": "Nighttime Light",
    "BC": "Black Carbon",
    "NH3": "Ammonia",
    "agriculture": "Land: Agriculture",
    "forest": "Land: Forest",
    "grassland": "Land: Grassland",
    "wetland": "Land: Wetland",
    "urban": "Land: Urban",
    "bare": "Land: Bare",
    "water": "Land: Water",
}


def normalize_model_name(name: str) -> str:
    lowered = name.strip().lower()
    if lowered == "tempo":
        return "TEMPO"
    if lowered in {"notempo", "no_tempo", "no-tempo"}:
        return "noTEMPO"
    raise ValueError(f"Unknown model name {name!r}. Use TEMPO or noTEMPO.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an XGBoost surface ozone model.")
    parser.add_argument("--model", default="TEMPO", help="TEMPO or noTEMPO.")
    parser.add_argument("--input", required=True, help="Path to the training CSV.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--n-trials", type=int, default=100, help="Optuna trials. Use 0 to skip tuning.")
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--val-size", type=float, default=0.10, help="Fraction of all rows used as validation.")
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional random sample for quick tests.")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU XGBoost.")
    parser.add_argument("--keep-negative-ozone", action="store_true", help="Do not drop ozone < 0 rows.")
    parser.add_argument(
        "--apply-stagnant-filter",
        action="store_true",
        help="Drop stalled-sensor blocks of >=8 identical consecutive hourly ozone readings.",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "training_pipeline.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def xgb_device(no_gpu: bool) -> str:
    if no_gpu:
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_and_clean_data(
    input_path: Path,
    features: list[str],
    sample_rows: int | None,
    keep_negative_ozone: bool,
    apply_stagnant_filter: bool,
) -> pd.DataFrame:
    logging.info("Loading data: %s", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    logging.info("Loaded rows=%s columns=%s", len(df), len(df.columns))

    if sample_rows is not None and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=RANDOM_STATE).copy()
        logging.info("Using random sample rows=%s", len(df))

    required = features + [TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    logging.info("Dropped rows with missing feature/target values: %s", before - len(df))

    if keep_negative_ozone:
        logging.info("Keeping ozone < 0 rows because --keep-negative-ozone was set.")
    else:
        before = len(df)
        df = df[df[TARGET_COLUMN] >= 0].copy()
        logging.info("Dropped rows with ozone < 0: %s", before - len(df))

    if apply_stagnant_filter:
        df = remove_stagnant_ozone_blocks(df)

    logging.info("Clean data rows=%s", len(df))
    return df


def remove_stagnant_ozone_blocks(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"Latitude", "Longitude", "Date GMT", "Time GMT", TARGET_COLUMN}
    if not needed.issubset(df.columns):
        logging.warning("Skipping stagnant filter because date/location columns are not all present.")
        return df

    before = len(df)
    work = df.copy()
    work["_datetime_utc"] = pd.to_datetime(
        work["Date GMT"].astype(str).str.strip() + " " + work["Time GMT"].astype(str).str.strip(),
        format="mixed",
        errors="coerce",
    )
    work = work.dropna(subset=["_datetime_utc"]).sort_values(["Latitude", "Longitude", "_datetime_utc"])
    group_cols = ["Latitude", "Longitude"]
    prev_ozone = work.groupby(group_cols)[TARGET_COLUMN].shift(1)
    prev_time = work.groupby(group_cols)["_datetime_utc"].shift(1)
    is_new_block = (
        (work[TARGET_COLUMN] != prev_ozone)
        | ((work["_datetime_utc"] - prev_time) != pd.Timedelta(hours=1))
    )
    work["_block_id"] = is_new_block.groupby([work["Latitude"], work["Longitude"]]).cumsum()
    work["_block_size"] = work.groupby(["Latitude", "Longitude", "_block_id"])[TARGET_COLUMN].transform("size")
    work = work[work["_block_size"] < 8].drop(columns=["_datetime_utc", "_block_id", "_block_size"])
    logging.info("Dropped stagnant same-ozone blocks >=8 hours: %s", before - len(work))
    return work


def train_val_test_split(df: pd.DataFrame, features: list[str], test_size: float, val_size: float):
    X = df[features].copy()
    y = df[TARGET_COLUMN].copy()

    train_val_size = 1.0 - test_size
    val_fraction_of_train_val = val_size / train_val_size
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_fraction_of_train_val,
        random_state=RANDOM_STATE,
    )
    logging.info("Split rows: train=%s val=%s test=%s", len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_extreme_weights(y: pd.Series, high_q: float = 0.97, weight: float = 3.0) -> np.ndarray:
    high = y.quantile(high_q)
    weights = np.ones(len(y), dtype=np.float32)
    weights[y.to_numpy() > high] = weight
    logging.info(
        "High-ozone weights: high>%.3f ppb weight=%.1f weighted_rows=%s",
        high, weight, int((weights > 1).sum()),
    )
    return weights


def scale_features(output_dir: Path, features: list[str], X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    scaler = MinMaxScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=features, index=X_val.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)
    with (output_dir / "scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)
    return scaler, X_train_s, X_val_s, X_test_s


def default_params(device: str) -> dict:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "booster": "gbtree",
        "tree_method": "hist",
        "device": device,
        "verbosity": 0,
        "n_estimators": 1200,
        "learning_rate": 0.05,
        "max_depth": 12,
        "min_child_weight": 10,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
    }


def tune_params(
    n_trials: int,
    device: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weight: np.ndarray,
) -> tuple[dict, int, pd.DataFrame]:
    if n_trials <= 0 or optuna is None:
        if optuna is None and n_trials > 0:
            logging.warning("Optuna is not installed. Using default XGBoost parameters.")
        params = default_params(device)
        return params, int(params["n_estimators"]), pd.DataFrame()

    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "booster": "gbtree",
            "tree_method": "hist",
            "device": device,
            "verbosity": 0,
            "n_estimators": 3000,
            "early_stopping_rounds": 50,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 18),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 80),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sample_weight, eval_set=[(X_val, y_val)], verbose=False)
        trial.set_user_attr("best_iteration", int(model.best_iteration or params["n_estimators"]))
        pred = model.predict(X_val)
        return r2_score(y_val, pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial
    logging.info("Best trial=%s validation_r2=%.5f", best.number, best.value)
    logging.info("Best params=%s", best.params)
    trials_df = study.trials_dataframe()
    params = default_params(device)
    params.update(best.params)
    params["n_estimators"] = int(best.user_attrs.get("best_iteration", 1200))
    return params, int(params["n_estimators"]), trials_df


def metrics_dict(y_true: pd.Series, pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    mae = float(mean_absolute_error(y_true, pred))
    r2 = float(r2_score(y_true, pred))
    rse = float(np.sqrt(np.sum((y_true - pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)))
    return {"r2": r2, "rmse": rmse, "mae": mae, "rse": rse, "n": int(len(y_true))}


def plot_scatter_density(y_true: pd.Series, pred: np.ndarray, output_path: Path, metrics: dict, max_points: int = 120_000) -> None:
    obs = y_true.to_numpy()
    prd = np.asarray(pred)
    if len(obs) > max_points:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(obs), size=max_points, replace=False)
        obs = obs[idx]
        prd = prd[idx]

    lr = LinearRegression().fit(obs.reshape(-1, 1), prd.reshape(-1, 1))
    slope = float(lr.coef_[0][0])
    intercept = float(lr.intercept_[0])

    try:
        xy = np.vstack([obs, prd])
        z = gaussian_kde(xy)(xy)
        order = z.argsort()
        obs, prd, z = obs[order], prd[order], z[order]
    except Exception:
        z = None

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=300)
    if z is None:
        ax.scatter(obs, prd, s=4, alpha=0.45, edgecolors="none")
    else:
        scatter = ax.scatter(obs, prd, c=z, cmap="viridis", s=4, alpha=0.55, edgecolors="none", rasterized=True)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04).set_label("Density")

    hi = float(np.nanpercentile(np.concatenate([obs, prd]), 99.8) * 1.05)
    hi = max(hi, 1.0)
    ax.plot([0, hi], [0, hi], "k--", lw=0.8, label="1:1")
    x_line = np.array([0, hi])
    ax.plot(x_line, intercept + slope * x_line, "r-", lw=0.8, label=f"Fit: y={intercept:.2f}+{slope:.2f}x")
    ax.text(
        0.97, 0.04,
        f"R2={metrics['r2']:.3f}\nRMSE={metrics['rmse']:.2f} ppb\nMAE={metrics['mae']:.2f} ppb",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.85, "ec": "none"},
    )
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Observed ozone (ppb)")
    ax.set_ylabel("Predicted ozone (ppb)")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_feature_importance(model: XGBRegressor, features: list[str], X_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> pd.DataFrame:
    baseline = mean_squared_error(y_test, model.predict(X_test))
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for feature in features:
        X_perm = X_test.copy()
        X_perm[feature] = rng.permutation(X_perm[feature].to_numpy())
        perm_mse = mean_squared_error(y_test, model.predict(X_perm))
        rows.append({"feature": feature, "display_name": FEATURE_NAME_MAPPING.get(feature, feature), "inc_mse": float(perm_mse - baseline)})
    importance = pd.DataFrame(rows).sort_values("inc_mse", ascending=False)
    importance.to_csv(output_dir / "permutation_importance_inc_mse.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    plot_df = importance.head(30).iloc[::-1]
    ax.barh(plot_df["display_name"], plot_df["inc_mse"], color="#2f6f9f")
    ax.set_xlabel("Permutation importance (increase in MSE)")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance_incmse.png", dpi=300)
    plt.close(fig)
    return importance


def main() -> None:
    args = parse_args()
    model_name = normalize_model_name(args.model)
    features = FEATURES_BY_MODEL[model_name]
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)

    device = xgb_device(args.no_gpu)
    logging.info("Model variant: %s", model_name)
    logging.info("XGBoost device: %s", device)
    logging.info("Features (%s): %s", len(features), ", ".join(features))

    df = load_and_clean_data(input_path, features, args.sample_rows, args.keep_negative_ozone, args.apply_stagnant_filter)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df, features, args.test_size, args.val_size)
    train_weights = make_extreme_weights(y_train)

    scaler, X_train_s, X_val_s, X_test_s = scale_features(output_dir, features, X_train, X_val, X_test)

    best_params, best_n_estimators, trials_df = tune_params(
        args.n_trials, device, X_train_s, y_train, X_val_s, y_val, train_weights
    )
    if not trials_df.empty:
        trials_df.to_csv(output_dir / "optuna_trials.csv", index=False)

    X_train_val_s = pd.concat([X_train_s, X_val_s])
    y_train_val = pd.concat([y_train, y_val])
    train_val_weights = make_extreme_weights(y_train_val)

    final_params = best_params.copy()
    final_params.pop("early_stopping_rounds", None)
    final_params["n_estimators"] = best_n_estimators
    final_model = XGBRegressor(**final_params)
    final_model.fit(X_train_val_s, y_train_val, sample_weight=train_val_weights)

    model_path = output_dir / f"xgb_model_{model_name.lower()}.json"
    final_model.save_model(model_path)
    logging.info("Saved model: %s", model_path)

    test_pred = final_model.predict(X_test_s)
    metrics = metrics_dict(y_test, test_pred)
    logging.info("Test metrics: %s", metrics)

    (output_dir / "selected_features.txt").write_text("\n".join(features) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "model_params.json").write_text(json.dumps(final_params, indent=2), encoding="utf-8")

    pd.DataFrame(
        {
            "row_index": X_test.index,
            "observed_ozone_ppb": y_test.to_numpy(),
            "predicted_ozone_ppb": test_pred,
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    plot_scatter_density(y_test, test_pred, output_dir / "scatter_density_test_set.png", metrics)
    plot_feature_importance(final_model, features, X_test_s, y_test, output_dir)

    logging.info("All outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
