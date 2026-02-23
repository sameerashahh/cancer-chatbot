import os, glob, json, argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_results(results_dir):
    paths = sorted(glob.glob(os.path.join(results_dir, "*_results.json")))
    if not paths:
        raise FileNotFoundError(f"No *_results.json files found in {results_dir}")

    model_to_scores = {}
    for p in paths:
        base = os.path.basename(p)
        model_name = base.replace("_results.json", "")
        with open(p, "r") as f:
            d = json.load(f)
        # force float
        model_to_scores[model_name] = {k: float(v) for k, v in d.items()}

    # union of datasets
    all_datasets = sorted({ds for m in model_to_scores for ds in model_to_scores[m].keys()})
    all_models = sorted(model_to_scores.keys())

    # build matrix: rows=datasets, cols=models
    mat = pd.DataFrame(index=all_datasets, columns=all_models, dtype=float)
    for m in all_models:
        for ds, val in model_to_scores[m].items():
            mat.loc[ds, m] = val

    return mat

def fit_per_model_anchor_regressors(score_mat, anchors, alpha=1.0):
    # score_mat: rows=datasets, cols=models
    for a in anchors:
        if a not in score_mat.columns:
            raise ValueError(f"Anchor model '{a}' not found. Available models: {list(score_mat.columns)}")

    regressors = {}
    for target_model in score_mat.columns:
        if target_model in anchors:
            continue  # you can predict anchors too, but usually they are inputs not targets

        # training rows where target + all anchors exist
        rows_ok = score_mat[anchors + [target_model]].dropna(axis=0)
        if len(rows_ok) < max(3, len(anchors) + 1):
            # not enough data to fit a stable regression
            continue

        X = rows_ok[anchors].values
        y = rows_ok[target_model].values

        reg = Ridge(alpha=alpha, random_state=0)
        reg.fit(X, y)
        regressors[target_model] = reg

    return regressors

def predict_for_dataset(score_mat, anchors, regressors, dataset_name):
    if dataset_name not in score_mat.index:
        raise ValueError(f"Dataset '{dataset_name}' not found in your JSON files.")

    anchor_vals = score_mat.loc[dataset_name, anchors]
    if anchor_vals.isna().any():
        missing = anchor_vals[anchor_vals.isna()].index.tolist()
        raise ValueError(
            f"Missing anchor scores for dataset '{dataset_name}'. Missing anchors: {missing}\n"
            f"You must have anchor results on the dataset to predict with the anchor method."
        )

    x = anchor_vals.values.reshape(1, -1)
    preds = {}
    for model, reg in regressors.items():
        preds[model] = float(reg.predict(x)[0])

    return preds

def loo_eval(score_mat, anchors, alpha=1.0):
    # Leave-One-Dataset-Out evaluation per target model
    results = []

    for target_model in score_mat.columns:
        if target_model in anchors:
            continue

        rows_ok = score_mat[anchors + [target_model]].dropna(axis=0)
        if len(rows_ok) < max(5, len(anchors) + 2):
            continue

        X = rows_ok[anchors].values
        y = rows_ok[target_model].values
        ds_names = rows_ok.index.tolist()

        loo = LeaveOneOut()
        y_true, y_pred = [], []
        for train_idx, test_idx in loo.split(X):
            reg = Ridge(alpha=alpha, random_state=0)
            reg.fit(X[train_idx], y[train_idx])
            pred = reg.predict(X[test_idx])[0]
            y_true.append(y[test_idx][0])
            y_pred.append(pred)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)

        results.append({
            "target_model": target_model,
            "n_datasets": len(ds_names),
            "MAE": mae,
            "RMSE": rmse
        })

    return pd.DataFrame(results).sort_values(["MAE", "RMSE"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=".", help="Folder with *_results.json files")
    ap.add_argument("--anchors", required=True, help="Comma-separated anchor model names (must match filenames without _results.json)")
    ap.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength")
    ap.add_argument("--predict_dataset", default=None, help="Dataset name to predict for (must exist in your JSONs)")
    ap.add_argument("--out_csv", default="anchor_predictions.csv", help="Where to write predictions (if predict_dataset is used)")
    ap.add_argument("--eval", action="store_true", help="Run leave-one-dataset-out evaluation")
    args = ap.parse_args()

    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]
    score_mat = load_results(args.results_dir)

    print("\nLoaded:")
    print("  models:", len(score_mat.columns))
    print("  datasets:", len(score_mat.index))
    print("Anchors:", anchors)

    regressors = fit_per_model_anchor_regressors(score_mat, anchors, alpha=args.alpha)
    print(f"Trained regressors for {len(regressors)} non-anchor models.\n")

    if args.eval:
        df_eval = loo_eval(score_mat, anchors, alpha=args.alpha)
        print(df_eval.to_string(index=False))
        df_eval.to_csv("anchor_method_loo_eval.csv", index=False)
        print("\nWrote: anchor_method_loo_eval.csv")

    if args.predict_dataset:
        preds = predict_for_dataset(score_mat, anchors, regressors, args.predict_dataset)

        # also include actuals if available
        out = []
        for m, p in sorted(preds.items()):
            actual = score_mat.loc[args.predict_dataset, m]
            out.append({
                "model": m,
                "pred": p,
                "actual": None if pd.isna(actual) else float(actual)
            })

        df = pd.DataFrame(out)
        df.to_csv(args.out_csv, index=False)
        print(f"\nPredicted scores for dataset '{args.predict_dataset}'. Wrote: {args.out_csv}")
        print(df.head(15).to_string(index=False))

if __name__ == "__main__":
    main()
