import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning


@dataclass(frozen=True)
class CellRef:
    model: str
    dataset: str


def load_score_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "model" not in df.columns:
        raise ValueError("CSV must contain a 'model' column.")
    return df.set_index("model")


def build_feature_vector(
    score_table: pd.DataFrame,
    target_model: str,
    target_dataset: str,
    feature_mode: str = "row_col",
) -> np.ndarray:
    dataset_names: List[str] = score_table.columns.tolist()
    model_names: List[str] = score_table.index.tolist()

    if feature_mode == "row_col":
        row_other = [float(score_table.loc[target_model, ds]) for ds in dataset_names if ds != target_dataset]
        col_other = [float(score_table.loc[m, target_dataset]) for m in model_names if m != target_model]

        row_mean = float(np.mean(row_other)) if row_other else 0.0
        row_std = float(np.std(row_other)) if row_other else 0.0
        col_mean = float(np.mean(col_other)) if col_other else 0.0
        col_std = float(np.std(col_other)) if col_other else 0.0

        features = np.array(row_other + col_other + [row_mean, row_std, col_mean, col_std], dtype=float)
        return features

    if feature_mode == "full_matrix":
        matrix = score_table.to_numpy(dtype=float).copy()
        row_idx = model_names.index(target_model)
        col_idx = dataset_names.index(target_dataset)
        matrix[row_idx, col_idx] = 0.0

        model_one_hot = np.zeros(len(model_names), dtype=float)
        model_one_hot[row_idx] = 1.0
        dataset_one_hot = np.zeros(len(dataset_names), dtype=float)
        dataset_one_hot[col_idx] = 1.0

        row_without_target = np.delete(matrix[row_idx, :], col_idx)
        col_without_target = np.delete(matrix[:, col_idx], row_idx)
        extras = np.array(
            [
                float(np.mean(row_without_target)) if row_without_target.size else 0.0,
                float(np.std(row_without_target)) if row_without_target.size else 0.0,
                float(np.mean(col_without_target)) if col_without_target.size else 0.0,
                float(np.std(col_without_target)) if col_without_target.size else 0.0,
            ],
            dtype=float,
        )

        features = np.concatenate(
            [
                matrix.reshape(-1),
                model_one_hot,
                dataset_one_hot,
                extras,
            ]
        )
        return features

    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def build_training_matrix(
    score_table: pd.DataFrame,
    feature_mode: str = "row_col",
    holdout: CellRef | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[CellRef]]:
    feature_rows = []
    labels = []
    refs: List[CellRef] = []

    for model in score_table.index:
        for dataset in score_table.columns:
            ref = CellRef(model=model, dataset=dataset)
            if holdout is not None and ref == holdout:
                continue
            feature_rows.append(build_feature_vector(score_table, model, dataset, feature_mode=feature_mode))
            labels.append(float(score_table.loc[model, dataset]))
            refs.append(ref)

    return np.vstack(feature_rows), np.array(labels, dtype=float), refs


def make_regressor(regressor_name: str, alpha: float):
    if regressor_name == "ridge":
        base = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        return TransformedTargetRegressor(regressor=base)

    if regressor_name == "huber":
        base = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("huber", HuberRegressor(alpha=alpha, max_iter=1000)),
            ]
        )
        return TransformedTargetRegressor(regressor=base)

    if regressor_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    if regressor_name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )

    raise ValueError(f"Unsupported regressor: {regressor_name}")


def leave_one_cell_out_predictions(
    score_table: pd.DataFrame,
    alpha: float,
    feature_mode: str,
    regressor_name: str,
) -> pd.DataFrame:
    rows = []
    for model in score_table.index:
        for dataset in score_table.columns:
            holdout = CellRef(model=model, dataset=dataset)
            x_train, y_train, _ = build_training_matrix(score_table, feature_mode=feature_mode, holdout=holdout)
            reg = make_regressor(regressor_name=regressor_name, alpha=alpha)
            reg.fit(x_train, y_train)

            x_test = build_feature_vector(score_table, model, dataset, feature_mode=feature_mode).reshape(1, -1)
            pred = float(reg.predict(x_test)[0])
            actual = float(score_table.loc[model, dataset])

            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "actual": actual,
                    "predicted": pred,
                    "abs_error": abs(actual - pred),
                }
            )

    return pd.DataFrame(rows)


def predict_single(
    score_table: pd.DataFrame,
    target_model: str,
    target_dataset: str,
    alpha: float,
    feature_mode: str,
    regressor_name: str,
) -> float:
    if target_model not in score_table.index:
        raise ValueError(f"Unknown model: {target_model}")
    if target_dataset not in score_table.columns:
        raise ValueError(f"Unknown dataset: {target_dataset}")

    holdout = CellRef(model=target_model, dataset=target_dataset)
    x_train, y_train, _ = build_training_matrix(score_table, feature_mode=feature_mode, holdout=holdout)
    reg = make_regressor(regressor_name=regressor_name, alpha=alpha)
    reg.fit(x_train, y_train)
    x_test = build_feature_vector(score_table, target_model, target_dataset, feature_mode=feature_mode).reshape(1, -1)
    return float(reg.predict(x_test)[0])


def summarize_predictions(predictions: pd.DataFrame) -> dict:
    return {
        "mae": float(mean_absolute_error(predictions["actual"], predictions["predicted"])),
        "rmse": float(np.sqrt(mean_squared_error(predictions["actual"], predictions["predicted"]))),
        "r2": float(r2_score(predictions["actual"], predictions["predicted"])),
    }


def print_summary(predictions: pd.DataFrame, feature_mode: str, regressor_name: str) -> None:
    metrics = summarize_predictions(predictions)
    print(f"Leave-one-cell-out regression summary ({feature_mode}, {regressor_name})")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R^2:  {metrics['r2']:.4f}")
    print()

    worst = predictions.sort_values("abs_error", ascending=False).head(10)
    print("Worst 10 held-out predictions")
    print(worst.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict model x dataset scores from other rows and columns.")
    parser.add_argument(
        "--csv",
        default="common_models_table.csv",
        help="Path to the common-model score table CSV.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["row_col", "full_matrix"],
        default="full_matrix",
        help="Feature set to use for prediction.",
    )
    parser.add_argument(
        "--regressor",
        choices=["ridge", "random_forest", "gradient_boosting", "huber"],
        default="ridge",
        help="Regressor to use.",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Run all supported regressors and print a comparison table.",
    )
    parser.add_argument(
        "--target-model",
        help="Optional target model for a single prediction.",
    )
    parser.add_argument(
        "--target-dataset",
        help="Optional target dataset for a single prediction.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    score_table = load_score_table(csv_path)

    if args.compare_all:
        rows = []
        for regressor_name in ["ridge", "huber", "random_forest", "gradient_boosting"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                predictions = leave_one_cell_out_predictions(
                    score_table,
                    alpha=args.alpha,
                    feature_mode=args.feature_mode,
                    regressor_name=regressor_name,
                )
            metrics = summarize_predictions(predictions)
            rows.append(
                {
                    "regressor": regressor_name,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                }
            )
        comparison = pd.DataFrame(rows).sort_values(["mae", "rmse"], ascending=True)
        print(f"Model comparison ({args.feature_mode})")
        print(comparison.to_string(index=False))
        return

    predictions = leave_one_cell_out_predictions(
        score_table,
        alpha=args.alpha,
        feature_mode=args.feature_mode,
        regressor_name=args.regressor,
    )
    print_summary(predictions, args.feature_mode, args.regressor)

    if args.target_model and args.target_dataset:
        predicted = predict_single(
            score_table,
            args.target_model,
            args.target_dataset,
            alpha=args.alpha,
            feature_mode=args.feature_mode,
            regressor_name=args.regressor,
        )
        actual = float(score_table.loc[args.target_model, args.target_dataset])
        print()
        print(f"Single held-out estimate for {args.target_model} on {args.target_dataset}")
        print(f"Predicted: {predicted:.4f}")
        print(f"Actual:    {actual:.4f}")


if __name__ == "__main__":
    main()
