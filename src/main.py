# main.py

import os

from src.cleaning import enhanced_clean
from src.feature_engineering import feature_engineering
from src.preprocessing import run_preprocessing_pipeline
from src.train_xgboost import train_xgboost_log_target
from src.Lin_reg import run_tuned_linear_models


RAW_PATH = "data/raw/raw.csv"
CLEANED_PATH = "data/processed/cleaned_v2.csv"
FEATURE_ENGINEERED_PATH = "data/processed/feature_engineered.csv"
POSTAL_REF_CSV = "data/reference/zipcodes_num_nl_2025.csv"

XGB_MODEL_PATH = "models/xgboost_log_model.pkl"


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("\n========== STEP 1: Cleaning ==========")
    enhanced_clean(
        path=RAW_PATH,
        save_path=CLEANED_PATH,
        postal_ref_csv=POSTAL_REF_CSV,
    )

    print("\n========== STEP 2: Feature Engineering ==========")
    feature_engineering(
        path=CLEANED_PATH,
        save_path=FEATURE_ENGINEERED_PATH,
    )

    print("\n========== STEP 3A: Preprocessing (XGBoost) ==========")
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb, _ = run_preprocessing_pipeline(
        path=FEATURE_ENGINEERED_PATH,
        target="price",
        drop_postal_code=False,   # keep postal_code for tree models
        remove_outliers=False,
        test_size=0.20,
        random_state=42,
    )

    print("\n========== STEP 4: Model Training (Log-XGBoost) ==========")
    df_train_for_xgb = X_train_xgb.copy()
    df_train_for_xgb["price"] = y_train_xgb

    xgb_model, xgb_results = train_xgboost_log_target(
        df=df_train_for_xgb,
        target="price",
        save_path=XGB_MODEL_PATH,
    )

    print("\nXGBoost Log-Target Metrics:")
    print(xgb_results)

    print("\n========== STEP 3B: Preprocessing (Linear Models) ==========")
    X_train_lin, X_test_lin, y_train_lin, y_test_lin, preprocessor_lin = run_preprocessing_pipeline(
        path=FEATURE_ENGINEERED_PATH,
        target="price",
        drop_postal_code=True,    # drop postal_code for linear models
        remove_outliers=False,
        test_size=0.20,
        random_state=42,
    )

    print("\n========== STEP 5: Linear Model Benchmarks ==========")
    linear_results = run_tuned_linear_models(
        X_train_lin,
        X_test_lin,
        y_train_lin,
        y_test_lin,
        preprocessor_lin,
    )

    print("\nLinear Model Results:")
    print(linear_results)

    print("\n========== PIPELINE COMPLETE ==========")
    print("XGBoost model saved at:", XGB_MODEL_PATH)


if __name__ == "__main__":
    main()
