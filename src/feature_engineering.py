import pandas as pd
import numpy as np
import os

def feature_engineering(
        path="data/processed/cleaned_v2.csv",
        save_path="data/processed/feature_engineered.csv"
    ):
    """
    FINAL Feature Engineering for ImmoEliza.
    Removes: locality_name
    Adds:
    - numeric postal_prefix
    - postal_area
    - postal_region_from_prefix
    - postal_density_group
    - build_year features
    - region mapping
    - boolean flags
    """

    df = pd.read_csv(path, dtype={"postal_code": "string"}).copy()

    # ------------------------------------------------------------
    # Base: ensure postal_code string
    # ------------------------------------------------------------
    df["postal_code"] = df["postal_code"].fillna("unknown").astype("string")

    # ------------------------------------------------------------
    # Build-year engineering
    # ------------------------------------------------------------
    if "build_year" in df.columns:
        current_year = 2024

        df["house_age"] = current_year - df["build_year"]
        df.loc[df["house_age"] < 0, "house_age"] = np.nan

        df["is_new_build"] = (df["house_age"] <= 5).astype("Int64")
        df["is_recent"]     = (df["house_age"] <= 20).astype("Int64")
        df["is_old"]        = (df["house_age"] >= 50).astype("Int64")

        df["build_decade"] = (df["build_year"] // 10 * 10).astype("Int64")
    else:
        df["house_age"] = np.nan
        df["is_new_build"] = np.nan
        df["is_recent"] = np.nan
        df["is_old"] = np.nan
        df["build_decade"] = np.nan

    # ------------------------------------------------------------
    # Province → Region mapping
    # ------------------------------------------------------------
    region_map = {
        "Antwerp": "Flanders",
        "East Flanders": "Flanders",
        "West Flanders": "Flanders",
        "Limburg": "Flanders",
        "Flemish Brabant": "Flanders",

        "Walloon Brabant": "Wallonia",
        "Hainaut": "Wallonia",
        "Liège": "Wallonia",
        "Luxembourg": "Wallonia",
        "Namur": "Wallonia",

        "Brussels": "Brussels"
    }

    df["region"] = df["province"].map(region_map).fillna("unknown").astype("string")

    # ------------------------------------------------------------
    # Boolean flags
    # ------------------------------------------------------------
    bool_map = {"yes": 1, "no": 0}

    for col in ["garden", "terrace", "swimming_pool"]:
        if col in df.columns:
            df[col + "_flag"] = df[col].map(bool_map).astype("Int64")
        else:
            df[col + "_flag"] = np.nan

    # ------------------------------------------------------------
    # POSTAL PREFIX (numeric)
    # ------------------------------------------------------------
    df["postal_prefix"] = (
        df["postal_code"]
        .str[:2]
        .apply(lambda x: int(x) if x.isdigit() else -1)
        .astype("Int64")
    )

    # ------------------------------------------------------------
    # Postal Area
    # ------------------------------------------------------------
    def map_area(prefix):
        if prefix in range(10, 20): return "Brussels-Capital"
        if prefix in range(20, 40): return "Flanders-Central"
        if prefix in range(40, 60): return "Flanders-West"
        if prefix in range(60, 70): return "Limburg-East"
        if prefix in range(70, 80): return "Hainaut-South"
        if prefix in range(80, 90): return "Liège-East"
        if prefix in range(90, 100): return "Luxembourg-South"
        return "unknown"

    df["postal_area"] = df["postal_prefix"].apply(map_area).astype("string")

    # ------------------------------------------------------------
    # Region from postal prefix
    # ------------------------------------------------------------
    def prefix_to_region(prefix):
        if prefix in range(10, 20): return "Brussels"
        if prefix in range(20, 60): return "Flanders"
        if prefix in range(60, 100): return "Wallonia"
        return "unknown"

    df["postal_region_from_prefix"] = df["postal_prefix"].apply(prefix_to_region).astype("string")

    # ------------------------------------------------------------
    # Density class from postal prefix
    # ------------------------------------------------------------
    def prefix_density(prefix):
        if prefix in {10, 11, 12, 13, 20, 21, 30}: return "urban"
        if prefix in range(40, 60): return "suburban"
        if prefix in range(60, 100): return "rural"
        return "unknown"

    df["postal_density_group"] = df["postal_prefix"].apply(prefix_density).astype("string")

    # ------------------------------------------------------------
    # Save engineered dataset
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"Feature-engineered dataset saved to: {save_path}")
    print("Final shape:", df.shape)
    print("\nColumns:", df.columns.tolist())

    return df
