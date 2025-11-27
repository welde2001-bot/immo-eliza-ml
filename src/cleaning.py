# cleaning.py

import pandas as pd
import numpy as np
import os


def enhanced_clean(
    path: str = "data/raw/raw.csv",
    save_path: str = "data/processed/cleaned_v2.csv"
):
    """
    Unified cleaning pipeline for ImmoEliza.

    Operations:
    - Remove unused columns
    - Normalize boolean-like fields
    - Convert numeric-like columns
    - Apply numeric sanity constraints
    - Ensure price integrity
    - Drop locality_name entirely
    """

    # 1. Load data
    df = pd.read_csv(path, dtype={"postal_code": "string"}).copy()

    # 2. Drop useless columns
    drop_cols = [
        "property_id",
        "property_url",
        "property_type_name",
        "state_mapped",
        "locality_name"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # 3. Normalize boolean-like columns
    bool_cols = ["garden", "terrace", "swimming_pool"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .str.strip()
                .replace({
                    "1": "yes",
                    "true": "yes",
                    "yes": "yes",
                    "0": "no",
                    "false": "no",
                    "no": "no"
                })
            )

    # 4. Convert numeric-like columns (comma → dot)
    numeric_cols = ["build_year", "number_rooms", "facades", "living_area"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Price
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Postal code stays categorical
    df["postal_code"] = df["postal_code"].astype("string")

    # 5. Numeric sanity constraints
    df.loc[df["build_year"] < 1800, "build_year"] = np.nan
    df.loc[df["build_year"] > 2025, "build_year"] = np.nan

    df.loc[df["number_rooms"] <= 0, "number_rooms"] = np.nan
    df.loc[df["number_rooms"] > 12, "number_rooms"] = np.nan

    df.loc[df["living_area"] < 10, "living_area"] = np.nan
    df.loc[df["living_area"] > 500, "living_area"] = np.nan

    df = df[df["price"] >= 10000]
    df.loc[df["price"] > 7_500_000, "price"] = np.nan
    df = df[df["price"].notna()]

    df["province"] = (
        df["province"]
        .astype(str)
        .str.strip()
        .replace("nan", np.nan)
    )

    # 6. Locality fully removed — nothing to do

    # 7. Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"Enhanced Cleaned dataset saved to: {save_path}")
    print("Final shape:", df.shape)
    return df


if __name__ == "__main__":
    enhanced_clean()
