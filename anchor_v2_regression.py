import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


class DataBundle(object):
    # simple container (no dataclasses; works on old python)
    def __init__(self, models, datasets, scores):
        self.models = list(models)
        self.datasets = list(datasets)
        self.scores = scores  # pandas DataFrame (rows=models, cols=datasets)


def load_results(results_dir: str) -> DataBundle:
    paths = sorted(glob.glob(os.path.join(results_dir, "*_results.json")))
    if not paths:
        raise FileNotFoundError("No *_results.json files found in: %s" % results_dir)

    model_to_scores: Dict[str, Dict[str, float]] = {}

    for p in paths:
        base = os.path.basename(p)
        model_name = base.replace("_results.json", "")
        with open(p, "r") as f:
            d = json.load(f)
        model_to_scores[model_name] = {k: float(v) for k, v in d.items()}

    all_datasets = sorted({ds for m in model_to_scores for ds in model_to_scores[m].keys()})
    all_models = sorted(model_to_scores.keys())

    mat = []
    for m in all_models:
        row = []
        for ds in all_datasets:
            row.append(model_to_scores[m].get(ds, np.nan))
        mat.append(row)

    scores = pd.DataFrame(mat, index=all_models, columns=all_datasets)

    if scores.isna().any().any():
        missing = int(scores.isna().sum().sum())
        raise ValueError(
            "Found %d missing (model,dataset) scores. Fill them or restrict to intersection first."
            % missing
        )

    return DataBundle(models=all_models, datasets=all_datasets, scores=scores)


def build_training_rows(
    bundle: DataBundle,
    anchors: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Build supervised dataset of examples: (model m, dataset d)

    Features:
      - model profile vector: scores of model m on all datasets, but with target dataset d hidden (set to 0)
      - mask vector: one-hot indicating which dataset was hidden (length D)
      - anchor difficulty vector: [anchor_1 score on d, anchor_2 score on d, ...] (length A)

    Target:
      - true score of model m on dataset d
    """
    D = len(bundle.datasets)

    X_rows = []
    y = []
    keys = []

    for m in bundle.models:
        m_vec_full = bundle.scores.loc[m].values.astype(float)  # length D

        for d_idx, d in enumerate(bundle.datasets):
            # model vector with target hidden
            m_vec = m_vec_full.copy()
            target = float(m_vec[d_idx])
            m_vec[d_idx] = 0.0

            # mask tells which dataset is hidden
            mask = np.zeros(D, dtype=float)
            mask[d_idx] = 1.0

            # anchor scores on this dataset d
            # (assumes anchors exist in bundle.models; checked in main)
            a_vec = np.array([float(bundle.scores.loc[a, d]) for a in anchors], dtype=float)

            feat = np.concatenate([m_vec, mask, a_vec], axis=0)
            X_rows.append(feat)
            y.append(target)
            keys.append((m, d))

    X = np.vstack(X_rows)
    y = np.array(y, dtype=float)
    return X, y, keys


def leave_one_model_out_eval(bundle: DataBundle, anchors: List[str], alpha: float) -> pd.DataFrame:
    """
    For each held-out model m*, train on all other models' (model,dataset) pairs,
    then predict all datasets for m* (one dataset at a time, i.e., missing-cell prediction).
    """
    results = []

    for held in bundle.models:
        train_models = [m for m in bundle.models if m != held]

        sub = DataBundle(
            models=train_models,
            datasets=bundle.datasets,
            scores=bundle.scores.loc[train_models, :].copy(),
        )
        X_train, y_train, _ = build_training_rows(sub, anchors)

        reg = Ridge(alpha=alpha, random_state=0)
        reg.fit(X_train, y_train)

        held_bundle = DataBundle(
            models=[held],
            datasets=bundle.datasets,
            scores=bundle.scores.loc[[held], :].copy(),
        )
        X_test, y_test, _ = build_training_rows(held_bundle, anchors)
        preds = reg.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        results.append({"heldout_model": held, "MAE": float(mae), "RMSE": float(rmse)})

    return pd.DataFrame(results).sort_values("MAE")


def predict_for_dataset(bundle: DataBundle, anchors: List[str], alpha: float, target_dataset: str) -> pd.DataFrame:
    """
    Train on ALL models (examples are still per-(model,dataset) with target dataset hidden per row),
    then report predictions specifically for target_dataset.
    """
    if target_dataset not in bundle.datasets:
        raise ValueError("Unknown dataset '%s'. Available: %s" % (target_dataset, bundle.datasets))

    X, y, keys = build_training_rows(bundle, anchors)

    reg = Ridge(alpha=alpha, random_state=0)
    reg.fit(X, y)

    rows = []
    for i, (m, d) in enumerate(keys):
        if d == target_dataset:
            pred = float(reg.predict(X[i:i + 1])[0])
            rows.append((m, pred, float(y[i])))

    out = pd.DataFrame(rows, columns=["model", "pred", "actual"]).sort_values("pred", ascending=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True, help="Folder with *_results.json files")
    ap.add_argument("--anchors", type=str, required=True, help="Comma-separated anchor model names (must match filenames)")
    ap.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength")
    ap.add_argument("--eval_lomo", action="store_true", help="Leave-one-model-out evaluation")
    ap.add_argument("--predict_dataset", type=str, default=None, help="Dataset name to print predictions for")
    ap.add_argument("--out_csv", type=str, default=None, help="Write prediction table to CSV")
    args = ap.parse_args()

    bundle = load_results(args.results_dir)
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]

    if len(anchors) < 1:
        raise ValueError("Provide at least 1 anchor model.")

    missing_anchors = [a for a in anchors if a not in bundle.models]
    if missing_anchors:
        raise ValueError(
            "Anchors not found in loaded models: %s\nLoaded models: %s"
            % (missing_anchors, bundle.models)
        )

    print("Loaded:")
    print("  models: %d" % len(bundle.models))
    print("  datasets: %d" % len(bundle.datasets))
    print("Anchors (%d): %s" % (len(anchors), anchors))
    print("Ridge alpha: %s" % args.alpha)

    if args.eval_lomo:
        df = leave_one_model_out_eval(bundle, anchors, alpha=args.alpha)
        print("\nLeave-one-model-out eval (lower is better):")
        print(df.to_string(index=False))
        out_eval = "anchor_v2_lomo_eval.csv"
        df.to_csv(out_eval, index=False)
        print("\nWrote: %s" % out_eval)

    if args.predict_dataset:
        pred_df = predict_for_dataset(bundle, anchors, alpha=args.alpha, target_dataset=args.predict_dataset)
        print("\nPredictions for dataset '%s':" % args.predict_dataset)
        print(pred_df.to_string(index=False))
        out_path = args.out_csv or ("preds_%s.csv" % args.predict_dataset)
        pred_df.to_csv(out_path, index=False)
        print("\nWrote: %s" % out_path)


if __name__ == "__main__":
    main()
