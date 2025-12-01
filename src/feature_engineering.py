import pandas as pd
import numpy as np
import os
from datetime import datetime


def feature_engineering(
    path="data/processed/cleaned_v2.csv",
    save_path="data/processed/feature_engineered.csv",
):
    """
    Feature Engineering for ImmoEliza.

    Adds:
    - build_year features (house_age, build_decade, age flags)
    - region mapping from province (uppercase NL)
    - boolean flags (garden/terrace/swimming_pool)
    - postal_code normalization (4-digit string or "unknown")
    """

    df = pd.read_csv(path, dtype={"postal_code": "string"}).copy()

    # postal_code: clean 4-digit string or "unknown"
    df["postal_code"] = (
        df["postal_code"]
        .astype("string")
        .str.strip()
        .str.extract(r"(\d{4})", expand=False)
        .fillna("unknown")
        .astype("string")
    )

    # build_year features
    if "build_year" in df.columns:
        build_year_num = pd.to_numeric(df["build_year"], errors="coerce")
        current_year = datetime.now().year

        df["house_age"] = current_year - build_year_num
        df.loc[df["house_age"] < 0, "house_age"] = np.nan

        df["is_new_build"] = (df["house_age"] <= 5).astype("Int64")
        df["is_recent"] = (df["house_age"] <= 20).astype("Int64")
        df["is_old"] = (df["house_age"] >= 50).astype("Int64")

        df["build_decade"] = (build_year_num // 10 * 10).astype("Int64")
    else:
        df["house_age"] = np.nan
        df["is_new_build"] = np.nan
        df["is_recent"] = np.nan
        df["is_old"] = np.nan
        df["build_decade"] = np.nan

    # region from province (province expected uppercase NL from cleaning.py)
    region_map_nl_upper = {
        "ANTWERPEN": "Flanders",
        "OOST-VLAANDEREN": "Flanders",
        "WEST-VLAANDEREN": "Flanders",
        "LIMBURG": "Flanders",
        "VLAAMS-BRABANT": "Flanders",
        "WAALS-BRABANT": "Wallonia",
        "HENEGOUWEN": "Wallonia",
        "LUIK": "Wallonia",
        "LUXEMBURG": "Wallonia",
        "NAMEN": "Wallonia",
        "BRUSSEL": "Brussels",
    }

    if "province" in df.columns:
        df["region"] = (
            df["province"]
            .astype("string")
            .str.strip()
            .str.upper()
            .map(region_map_nl_upper)
            .fillna("unknown")
            .astype("string")
        )
    else:
        df["region"] = "unknown"

    # boolean flags
    bool_map = {"yes": 1, "no": 0}
    for col in ["garden", "terrace", "swimming_pool"]:
        if col in df.columns:
            df[f"{col}_flag"] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.lower()
                .map(bool_map)
                .astype("Int64")
            )
        else:
            df[f"{col}_flag"] = np.nan

    # save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"Feature-engineered dataset saved to: {save_path}")
    print("Final shape:", df.shape)
    print("Region counts:\n", df["region"].value_counts(dropna=False))
    print("\nColumns:", df.columns.tolist())

    return df
