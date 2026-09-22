"""
Hyperparameter tuning for LightGBM using Optuna (Bayesian optimization).
Tuned on a single hold-out split for speed; results should be confirmed
via K-Fold CV before use (see train.py).

Note: the same approach was also tried for XGBoost and CatBoost, but with
a limited trial budget (~20-40 trials) neither beat their default
hyperparameters — so no equivalent tuning script is included for them here.
"""

import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from preprocess import prepare_data, FEATURE_COLS, CATEGORICAL_COLS, TARGET

RANDOM_STATE = 42
N_TRIALS = 50


def objective(trial: optuna.Trial, X_train, y_train, X_val, y_val) -> float:
    """Train LightGBM on hold-out split, return validation AUC."""
    params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 5000,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=CATEGORICAL_COLS,
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, val_pred)


def run_tuning(train_path: str, test_path: str, n_trials: int = N_TRIALS) -> dict:
    """Run Optuna study for LightGBM and return the best hyperparameters found."""
    train_df, _, feature_cols = prepare_data(train_path, test_path)

    X = train_df[feature_cols]
    y = train_df[TARGET]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
    )

    return study.best_params


if __name__ == "__main__":
    best_params = run_tuning(train_path="data/train.csv", test_path="data/test.csv")
    print("Best params:", best_params)