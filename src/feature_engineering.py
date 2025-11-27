import pandas as pd
import numpy as np
import os


def feature_engineering(path="data/processed/cleaned_v2.csv",
                        save_path="data/processed/feature_engineered.csv"):
    """
    Minimal, high-signal feature engineering for ImmoEliza dataset.
    """

    df = pd.read_csv(path, dtype={"postal_code": "string"}).copy()

    # Ensure postal_code is string
    df["postal_code"] = df["postal_code"].fillna("unknown").astype("string")

    # ------------------------------
    # Build-year features
    # ------------------------------
    current_year = 2024
    df["house_age"] = current_year - df["build_year"]
    df.loc[df["house_age"] < 0, "house_age"] = np.nan

    df["is_new_build"] = (df["house_age"] <= 5).astype("Int64")
    df["is_recent"] = (df["house_age"] <= 20).astype("Int64")
    df["is_old"] = (df["house_age"] >= 50).astype("Int64")

    df["build_decade"] = (df["build_year"] // 10 * 10).astype("Int64")

    # ------------------------------
    # Boolean flags
    # ------------------------------
    bool_map = {"yes": 1, "no": 0}
    for col in ["garden", "terrace", "swimming_pool"]:
        if col in df.columns:
            df[col + "_flag"] = df[col].map(bool_map).astype("Int64")

    # ------------------------------
    # Location encoding
    # ------------------------------
    df["postal_prefix"] = df["postal_code"].str[:2].astype("string")

    # ------------------------------
    # Save
    # ------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"[OK] Feature engineering complete. Saved to {save_path}")
    return df
