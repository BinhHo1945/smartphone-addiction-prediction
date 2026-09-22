"""
Data loading, feature engineering, and dtype preparation for the
Smartphone Addiction Prediction pipeline.
"""

from typing import Tuple

import numpy as np
import pandas as pd

TARGET = "addicted_label"
ID_COL = "id"

NUMERIC_COLS = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day",
    "app_opens_per_day", "weekend_screen_time",
]
CATEGORICAL_COLS = ["gender", "stress_level", "academic_work_impact"]

# Features engineered from EDA-identified correlation structure among
# screen-time related columns. Round 2 (product/interaction features) was
# tested and discarded — it degraded OOF AUC due to redundancy with these.
ENGINEERED_FEATURE_COLS = [
    "ratio_social_to_daily", "ratio_gaming_to_daily", "ratio_work_to_daily",
    "sum_sub_components", "diff_daily_vs_sum_components",
    "diff_weekend_vs_daily", "ratio_weekend_vs_daily",
    "notifications_per_app_open", "diff_sleep_vs_daily_screen",
    "n_missing_numeric",
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS + ENGINEERED_FEATURE_COLS


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train/test CSV files."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def engineer_features(df: pd.DataFrame, numeric_cols: list[str], eps: float = 1e-6) -> pd.DataFrame:
    """
    Create technical features based on structure observed in EDA (high
    correlation among screen-time related columns). No domain-specific
    assumption about "addiction behavior" is encoded here — usefulness
    was validated purely via OOF AUC improvement.
    """
    result_df = df.copy()

    result_df["ratio_social_to_daily"] = result_df["social_media_hours"] / (result_df["daily_screen_time_hours"] + eps)
    result_df["ratio_gaming_to_daily"] = result_df["gaming_hours"] / (result_df["daily_screen_time_hours"] + eps)
    result_df["ratio_work_to_daily"] = result_df["work_study_hours"] / (result_df["daily_screen_time_hours"] + eps)

    result_df["sum_sub_components"] = (
        result_df["social_media_hours"] + result_df["gaming_hours"] + result_df["work_study_hours"]
    )
    result_df["diff_daily_vs_sum_components"] = result_df["daily_screen_time_hours"] - result_df["sum_sub_components"]

    result_df["diff_weekend_vs_daily"] = result_df["weekend_screen_time"] - result_df["daily_screen_time_hours"]
    result_df["ratio_weekend_vs_daily"] = result_df["weekend_screen_time"] / (result_df["daily_screen_time_hours"] + eps)

    result_df["notifications_per_app_open"] = result_df["notifications_per_day"] / (result_df["app_opens_per_day"] + eps)
    result_df["diff_sleep_vs_daily_screen"] = result_df["sleep_hours"] - result_df["daily_screen_time_hours"]

    result_df["n_missing_numeric"] = df[numeric_cols].isna().sum(axis=1)

    return result_df


def convert_categorical_dtype(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Convert categorical columns to 'category' dtype (required by LightGBM/XGBoost
    for native categorical handling; NaN is preserved as a valid category)."""
    result_df = df.copy()
    for col in cat_cols:
        result_df[col] = result_df[col].astype("category")
    return result_df


def prepare_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Full preprocessing pipeline: load -> feature engineer -> dtype conversion.
    Missing values are intentionally NOT imputed — LightGBM/XGBoost handle
    NaN natively, and EDA confirmed missingness is unrelated to the target
    (near-MCAR/MAR, not MNAR), so imputation offers no benefit here.
    """
    train_df, test_df = load_data(train_path, test_path)

    train_df = engineer_features(train_df, NUMERIC_COLS)
    test_df = engineer_features(test_df, NUMERIC_COLS)

    train_df = convert_categorical_dtype(train_df, CATEGORICAL_COLS)
    test_df = convert_categorical_dtype(test_df, CATEGORICAL_COLS)

    return train_df, test_df, FEATURE_COLS