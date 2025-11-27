# main.py

import os
import pandas as pd

from cleaning import enhanced_clean
from preprocessing import run_preprocessing_pipeline
from train_xgboost import train_xgboost_log_target
from Lin_reg import run_tuned_linear_models


# ============================================================
# 1. PATHS
# ============================================================
RAW_PATH = "data/raw/raw.csv"
CLEANED_PATH = "data/processed/cleaned_v2.csv"
FEATURE_ENGINEERED_PATH = "data/processed/feature_engineered.csv"
MODEL_PATH = "models/xgboost_log_model.pkl"


# ============================================================
# 2. OPTIONAL FEATURE ENGINEERING
# ============================================================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for feature engineering logic.
    Extend this function with additional transformations.
    """
    # Example features (add your own):
    # df["price_per_m2"] = df["price"] / df["living_area"]
    # df["age"] = 2025 - df["build_year"]
    return df


# ============================================================
# 3. MAIN WORKFLOW
# ============================================================
def main():

    print("\n========== STEP 1: Cleaning ==========")
    df_clean = enhanced_clean(
        path=RAW_PATH,
        save_path=CLEANED_PATH
    )

    print("\n========== STEP 2: Feature Engineering ==========")
    df_fe = feature_engineering(df_clean)
    os.makedirs(os.path.dirname(FEATURE_ENGINEERED_PATH), exist_ok=True)
    df_fe.to_csv(FEATURE_ENGINEERED_PATH, index=False)
    print(f"Feature engineered dataset saved to: {FEATURE_ENGINEERED_PATH}")

    print("\n========== STEP 3: Preprocessing ==========")
    X_train_clean, X_test, y_train_clean, y_test, preprocessor = \
        run_preprocessing_pipeline(
            path=FEATURE_ENGINEERED_PATH,
            target="price"
        )

    print("\n========== STEP 4: Model Training (Log-XGBoost) ==========")
    df_train_for_xgb = X_train_clean.copy()
    df_train_for_xgb["price"] = y_train_clean

    xgb_model, xgb_results = train_xgboost_log_target(
        df=df_train_for_xgb,
        target="price",
        save_path=MODEL_PATH
    )

    print("\nXGBoost Log-Target Metrics:")
    print(xgb_results)

    print("\n========== STEP 5: Linear Model Benchmarks ==========")
    linear_results = run_tuned_linear_models(
        X_train_clean,
        X_test,
        y_train_clean,
        y_test,
        preprocessor
    )

    print("\nLinear Model Results:")
    print(linear_results)

    print("\n========== PIPELINE COMPLETE ==========")
    print("XGBoost model saved at:", MODEL_PATH)
    print("All results computed successfully.")


# ============================================================
# 4. RUN MAIN
# ============================================================
if __name__ == "__main__":
    main()
