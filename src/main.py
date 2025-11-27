from enhanced_cleaning import enhanced_cleaning
from feature_engineering import feature_engineering
from train_xgboost import train_xgboost


def main():

    print("Step 1: Enhanced Cleaning...")
    df_clean = enhanced_cleaning(
        path="data/raw/raw.csv",
        save_path="data/processed/cleaned_v2.csv"
    )

    print("Step 2: Feature Engineering...")
    df_fe = feature_engineering(
        path="data/processed/cleaned_v2.csv",
        save_path="data/processed/feature_engineered.csv"
    )

    print("Step 3: Train XGBoost...")
    train_xgboost(df_fe,
                  target="price",
                  save_path="models/xgboost_geo_tuned.pkl")


if __name__ == "__main__":
    main()
