# main.py

import os
import pandas as pd

from cleaning import enhanced_clean
from preprocessing import run_preprocessing_pipeline
from train_xgboost import train_xgboost_log_target


# ============================================================
# 1. PATHS (adjust if needed)
# ============================================================
RAW_PATH = "data/raw/raw.csv"
CLEANED_PATH = "data/processed/cleaned_v2.csv"

# Optional: feature engineering output
FEATURE_ENGINEERED_PATH = "data/processed/feature_engineered.csv"

MODEL_PATH = "models/xgboost_log_model.pkl"


# ============================================================
# 2. OPTIONAL FEATURE ENGINEERING
# (Placeholder – fill in if you have a script)
# ============================================================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stub for feature engineering.
    Add your transformations here.
    """
    # Example: df["price_per_m2"] = df["price"] / df["living_area"]
    # Example: df["is_old"] = (df["build_year"] < 1950).astype(int)
    return df


# ============================================================
# 3. MAIN WORKFLOW
# ============================================================
def main():

    print("\n========== STEP 1: Cleaning ==========")
    df_clean = enhanced_clean(path=RAW_PATH, save_path=CLEANED_PATH)

    print("\n========== STEP 2: Feature Engineering ==========")
    df_fe = feature_engineering(df_clean)
    os.makedirs(os.path.dirname(FEATURE_ENGINEERED_PATH), exist_ok=True)
    df_fe.to_csv(FEATURE_ENGINEERED_PATH, index=False)
    print(f"Feature engineered dataset saved to: {FEATURE_ENGINEERED_PATH}")

    print("\n========== STEP 3: Preprocessing ==========")
    X_train, X_test, y_train, y_test, preprocessor = run_preprocessing_pipeline(
        path=FEATURE_ENGINEERED_PATH,
        target="price"
    )

    print("\n========== STEP 4: Model Training (XGBoost Log Target) ==========")
    # Combine X_train_clean + y_train_clean into a single df for your function
    df_train_full = X_train.copy()
    df_train_full["price"] = y_train

    model, results = train_xgboost_log_target(
        df=df_train_full,
        target="price",
        save_path=MODEL_PATH
    )

    print("\n========== PIPELINE COMPLETE ==========")
    print("Model saved at:", MODEL_PATH)
    print("Training metrics:", results)


# ============================================================
# 4. Run Main
# ============================================================
if __name__ == "__main__":
    main()
