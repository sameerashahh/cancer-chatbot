import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------------------------
# DeepSeek is excluded.
# IBM Granite and Falcon only have the first 5 dataset scores in the screenshots,
# so they are left as NaN for the last 5. The script will automatically drop any
# incomplete rows before training.
# -----------------------------------------------------------------------------

DATASETS = [
    "GSM8K",
    "ARC",
    "BoolQ",
    "WinoGrande",
    "HellaSwag",
    "MMLU",
    "PIQA",
    "OpenBookQA",
    "StrategyQA",
    "TruthfulQA",
]

RAW_SCORES = {
    "Qwen/Qwen3-4B-Instruct-2507":      [0.664, 0.903, 0.848, 0.574, 0.812, 0.706, 0.840, 0.852, 0.638, 0.612],
    "Qwen/Qwen2.5-3B-Instruct":         [0.492, 0.839, 0.764, 0.534, 0.784, 0.616, 0.800, 0.526, 0.566, 0.582],
    "mistralai/Mistral-Nemo-Instruct-2407": [0.576, 0.849, 0.842, 0.528, 0.750, 0.620, 0.802, 0.810, 0.528, 0.632],
    "Qwen/Qwen2.5-1.5B-Instruct":       [0.404, 0.789, 0.756, 0.504, 0.604, 0.528, 0.718, 0.626, 0.574, 0.462],
    "google/gemma-3-4b-it":             [0.590, 0.773, 0.830, 0.572, 0.632, 0.524, 0.766, 0.760, 0.634, 0.564],
    "meta-llama/Llama-3.2-3B-Instruct": [0.532, 0.749, 0.728, 0.512, 0.590, 0.530, 0.780, 0.746, 0.644, 0.516],
    "meta-llama/Llama-3.2-1B-Instruct": [0.288, 0.528, 0.588, 0.490, 0.260, 0.402, 0.574, 0.360, 0.540, 0.184],
    "google/gemma-3-1b-it":             [0.218, 0.435, 0.678, 0.498, 0.400, 0.334, 0.632, 0.378, 0.524, 0.080],
    "allenai/Olmo-3-7B-Instruct":       [0.218, 0.753, 0.846, 0.572, 0.680, 0.560, 0.772, 0.740, 0.652, 0.678],
    "ibm-granite/granite-3.1-2b-instruct": [0.488, 0.719, 0.780, 0.564, 0.644, np.nan, np.nan, np.nan, np.nan, np.nan],
    "tiiuae/Falcon3-3B-Instruct":       [0.530, 0.742, 0.266, 0.552, 0.590, np.nan, np.nan, np.nan, np.nan, np.nan],
}


def build_complete_score_table() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(RAW_SCORES, orient="index", columns=DATASETS)
    incomplete_models = df.index[df.isna().any(axis=1)].tolist()
    if incomplete_models:
        print("Dropping incomplete models because they do not have all 10 dataset scores:")
        for m in incomplete_models:
            print(f"  - {m}")
    df = df.dropna(axis=0).copy()
    return df


def build_feature(scores: np.ndarray, model_idx: int, dataset_idx: int) -> np.ndarray:
    """
    Feature layout exactly as requested:
      [all OTHER models on target dataset] + [current model on all OTHER datasets]
    """
    other_models_same_dataset = np.delete(scores[:, dataset_idx], model_idx)
    same_model_other_datasets = np.delete(scores[model_idx, :], dataset_idx)
    return np.concatenate([other_models_same_dataset, same_model_other_datasets])



def build_training_examples(df: pd.DataFrame):
    scores = df.to_numpy(dtype=float)
    X, y, meta = [], [], []
    for m_idx, model_name in enumerate(df.index):
        for d_idx, dataset_name in enumerate(df.columns):
            X.append(build_feature(scores, m_idx, d_idx))
            y.append(scores[m_idx, d_idx])
            meta.append((model_name, dataset_name, m_idx, d_idx))
    return np.asarray(X), np.asarray(y), meta



def make_regressor() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-4, 4, 81))),
    ])



def leave_one_cell_out_eval(df: pd.DataFrame):
    X, y, meta = build_training_examples(df)
    preds = []
    alphas = []

    for i in range(len(y)):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[i] = False
        model = make_regressor()
        model.fit(X[train_mask], y[train_mask])
        pred = float(model.predict(X[i:i + 1])[0])
        preds.append(pred)
        alphas.append(float(model.named_steps["ridge"].alpha_))

    mae = mean_absolute_error(y, preds)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    r2 = r2_score(y, preds)

    rows = []
    for (model_name, dataset_name, _, _), actual, pred, alpha in zip(meta, y, preds, alphas):
        rows.append({
            "model": model_name,
            "dataset": dataset_name,
            "actual": float(actual),
            "predicted": float(pred),
            "abs_error": float(abs(actual - pred)),
            "alpha": alpha,
        })

    pred_df = pd.DataFrame(rows).sort_values(["dataset", "model"])
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2), "predictions": pred_df}



def fit_final_model(df: pd.DataFrame) -> Pipeline:
    X, y, _ = build_training_examples(df)
    model = make_regressor()
    model.fit(X, y)
    return model



def predict_single_cell(df: pd.DataFrame, fitted_model: Pipeline, model_name: str, dataset_name: str) -> float:
    m_idx = df.index.get_loc(model_name)
    d_idx = df.columns.get_loc(dataset_name)
    x = build_feature(df.to_numpy(dtype=float), m_idx, d_idx).reshape(1, -1)
    return float(fitted_model.predict(x)[0])


if __name__ == "__main__":
    df = build_complete_score_table()

    print("\nComplete score table used for regression:")
    print(df)
    print(f"\nUsing {df.shape[0]} models x {df.shape[1]} datasets.")

    results = leave_one_cell_out_eval(df)
    print("\nLeave-one-cell-out evaluation")
    print(f"MAE  = {results['mae']:.4f}")
    print(f"RMSE = {results['rmse']:.4f}")
    print(f"R^2  = {results['r2']:.4f}")

    results["predictions"].to_csv("cell_predictions.csv", index=False)
    print("\nSaved prediction table to cell_predictions.csv")

    final_model = fit_final_model(df)
    example_model = "Qwen/Qwen3-4B-Instruct-2507"
    example_dataset = "TruthfulQA"
    example_pred = predict_single_cell(df, final_model, example_model, example_dataset)
    print(
        f"\nExample fitted prediction for ({example_model}, {example_dataset}): "
        f"{example_pred:.4f}"
    )
