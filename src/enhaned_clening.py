import pandas as pd
import numpy as np
import os


def enhanced_cleaning(path="data/raw/raw.csv",
                      save_path="data/processed/cleaned_v2.csv"):
    """
    Full production-grade cleaning pipeline for ImmoEliza dataset.
    """

    df = pd.read_csv(path, dtype={"postal_code": "string"}).copy()

    # ------------------------------
    # Drop irrelevant columns
    # ------------------------------
    drop_cols = [
        "property_id",
        "property_url",
        "property_type_name",
        "state_mapped"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ------------------------------
    # Normalize booleans
    # ------------------------------
    bool_cols = ["garden", "terrace", "swimming_pool"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.lower().str.strip()
                .replace({
                    "1": "yes",
                    "true": "yes",
                    "yes": "yes",
                    "0": "no",
                    "false": "no",
                    "no": "no",
                    "nan": np.nan
                })
            )

    # ------------------------------
    # Convert numeric-like columns
    # ------------------------------
    for col in ["build_year", "number_rooms", "facades", "living_area"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            )

    # Keep as string but also convert to numeric for sanity filtering
    df["postal_code"] = df["postal_code"].astype("string")
    df["postal_code_num"] = pd.to_numeric(df["postal_code"], errors="coerce")

    # ------------------------------
    # Numeric sanity filters
    # ------------------------------
    df.loc[df["build_year"].between(1800, 2025) == False, "build_year"] = np.nan
    df.loc[df["number_rooms"].between(1, 12) == False, "number_rooms"] = np.nan
    df.loc[df["living_area"].between(10, 500) == False, "living_area"] = np.nan

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"] >= 10_000]
    df.loc[df["price"] > 7_500_000, "price"] = np.nan
    df = df[df["price"].notna()]

    # ------------------------------
    # Locality reduction
    # ------------------------------
    if "locality_name" in df.columns:
        top_k = df["locality_name"].value_counts().head(50).index
        df["locality_name"] = df["locality_name"].where(
            df["locality_name"].isin(top_k),
            "Other"
        )

    # ------------------------------
    # Region extraction
    # ------------------------------
    province_to_region = {
        "Antwerp": "Flanders",
        "Limburg": "Flanders",
        "East Flanders": "Flanders",
        "Flemish Brabant": "Flanders",
        "West Flanders": "Flanders",

        "Walloon Brabant": "Wallonia",
        "Hainaut": "Wallonia",
        "Liège": "Wallonia",
        "Namur": "Wallonia",
        "Luxembourg": "Wallonia",

        "Brussels": "Brussels"
    }

    if "province" in df.columns:
        df["region"] = df["province"].map(province_to_region).fillna("Other")

    # ------------------------------
    # Final save
    # ------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"[OK] Enhanced cleaned dataset saved to {save_path}")
    return df
