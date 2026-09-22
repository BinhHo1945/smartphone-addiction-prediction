"""
Model training, K-Fold cross-validation, and ensembling for the final
Smartphone Addiction Prediction submission.

Final model: LightGBM (tuned, 3-seed averaged) blended with XGBoost
(default hyperparameters), weighted 0.60 / 0.40 respectively — the
weight was selected via grid search on out-of-fold (OOF) predictions.
CatBoost was also tried as a third ensemble member but contributed
negligible improvement (+0.00002 OOF) and is excluded here for simplicity.
"""

from typing import Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from preprocess import prepare_data, CATEGORICAL_COLS, TARGET, ID_COL

RANDOM_STATE = 42
N_FOLDS = 5
SEEDS = [42, 2024, 7]

# Best LightGBM hyperparameters found via Optuna tuning (see tune.py)
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 5000,
    "n_jobs": -1,
    "verbosity": -1,
    "learning_rate": 0.02019711450378175,
    "num_leaves": 60,
    "max_depth": 10,
    "min_child_samples": 168,
    "subsample": 0.6770576098049709,
    "colsample_bytree": 0.6519415109251234,
    "reg_alpha": 5.491719596180856,
    "reg_lambda": 0.01198316134356364,
}

# Reasonable default hyperparameters for XGBoost (not deeply tuned —
# Optuna tuning with a limited trial budget underperformed these defaults)
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 5000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "enable_categorical": True,
    "tree_method": "hist",
    "n_jobs": -1,
    "verbosity": 0,
}


def train_lightgbm_kfold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    random_state: int,
    n_folds: int = N_FOLDS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train LightGBM with Stratified K-Fold CV. Returns (oof_preds, test_preds)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    X, y = train_df[feature_cols], train_df[TARGET]
    X_test = test_df[feature_cols]

    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    for train_idx, val_idx in skf.split(X, y):
        model = lgb.LGBMClassifier(**{**LGBM_PARAMS, "random_state": random_state})
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            eval_metric="auc",
            categorical_feature=CATEGORICAL_COLS,
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        oof_preds[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / n_folds

    return oof_preds, test_preds


def train_lightgbm_seed_averaged(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    seeds: list[int] = SEEDS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Average LightGBM K-Fold OOF/test predictions across multiple seeds
    to reduce prediction variance."""
    oof_per_seed = np.zeros((len(seeds), len(train_df)))
    test_per_seed = np.zeros((len(seeds), len(test_df)))

    for i, seed in enumerate(seeds):
        oof_per_seed[i], test_per_seed[i] = train_lightgbm_kfold(
            train_df, test_df, feature_cols, random_state=seed
        )

    return oof_per_seed.mean(axis=0), test_per_seed.mean(axis=0)


def train_xgboost_kfold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    random_state: int = RANDOM_STATE,
    n_folds: int = N_FOLDS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train XGBoost with Stratified K-Fold CV. Returns (oof_preds, test_preds)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    X, y = train_df[feature_cols], train_df[TARGET]
    X_test = test_df[feature_cols]

    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    for train_idx, val_idx in skf.split(X, y):
        model = xgb.XGBClassifier(**XGB_PARAMS, random_state=random_state, early_stopping_rounds=100)
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            verbose=False,
        )
        oof_preds[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / n_folds

    return oof_preds, test_preds


def find_optimal_blend_weight(
    oof_a: np.ndarray, oof_b: np.ndarray, y_true: np.ndarray, step: float = 0.05
) -> float:
    """Grid-search the blend weight for model A (1 - w for model B) that
    maximizes AUC on OOF predictions."""
    best_weight, best_auc = 0.5, 0.0
    for w in np.arange(0.0, 1.0 + step, step):
        auc = roc_auc_score(y_true, w * oof_a + (1 - w) * oof_b)
        if auc > best_auc:
            best_auc, best_weight = auc, w
    return best_weight


def create_submission(test_ids: pd.Series, pred_proba: np.ndarray, output_path: str) -> None:
    """Write a Kaggle-format submission file."""
    submission_df = pd.DataFrame({ID_COL: test_ids, TARGET: pred_proba})
    submission_df.to_csv(output_path, index=False)


def run_pipeline(train_path: str, test_path: str, output_path: str) -> None:
    """End-to-end training + ensembling pipeline producing the final submission."""
    train_df, test_df, feature_cols = prepare_data(train_path, test_path)
    y_true = train_df[TARGET].values

    lgbm_oof, lgbm_test = train_lightgbm_seed_averaged(train_df, test_df, feature_cols)
    xgb_oof, xgb_test = train_xgboost_kfold(train_df, test_df, feature_cols)

    weight = find_optimal_blend_weight(lgbm_oof, xgb_oof, y_true)
    final_test_preds = weight * lgbm_test + (1 - weight) * xgb_test

    blended_auc = roc_auc_score(y_true, weight * lgbm_oof + (1 - weight) * xgb_oof)
    print(f"Blend weight (LightGBM): {weight:.2f} | OOF AUC: {blended_auc:.5f}")

    create_submission(test_df[ID_COL], final_test_preds, output_path)


if __name__ == "__main__":
    run_pipeline(
        train_path="data/train.csv",
        test_path="data/test.csv",
        output_path="submission.csv",
    )