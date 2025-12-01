# cleaning.py

import os
import unicodedata

import numpy as np
import pandas as pd


def load_bpost_csv(path: str) -> pd.DataFrame:
    """
    Robust loader for the bpost postcode reference CSV.
    Tries common separators and encodings used in Belgian exports.
    """
    last_err = None
    for sep in [";", ",", "\t", "|"]:
        for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
            try:
                ref = pd.read_csv(path, sep=sep, encoding=enc, dtype=str, engine="python")
                if ref.shape[1] >= 2 and ref.shape[0] >= 1:
                    return ref
            except Exception as e:
                last_err = e
    raise ValueError(f"Could not read {path}. Last error: {last_err}")


def _norm_text(x):
    """
    Normalize messy text values into clean strings or NA.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return pd.NA
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return pd.NA
    s = unicodedata.normalize("NFKC", s)
    s = " ".join(s.split())
    return s


def enhanced_clean(
    path: str = "data/raw/raw.csv",
    save_path: str = "data/processed/cleaned_v2.csv",
    postal_ref_csv: str = "data/reference/zipcodes_num_nl_2025.csv",
):
    """
    Cleaning pipeline (ImmoEliza)

    - Drop unused columns
    - Normalize booleans to "yes"/"no"
    - Parse numeric columns (handles commas/units)
    - Extract 4-digit postal codes
    - Fill/standardize province using bpost lookup (postal_code -> Provincie)
    - Apply sanity constraints and remove invalid target values
    - Save cleaned CSV
    """
    df = pd.read_csv(path, dtype={"postal_code": "string"}).copy()

    drop_cols = ["property_id", "property_url", "property_type_name", "state_mapped", "locality_name"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    bool_cols = ["garden", "terrace", "swimming_pool"]
    bool_map = {
        "1": "yes", "true": "yes", "yes": "yes", "y": "yes",
        "0": "no",  "false": "no",  "no": "no",   "n": "no",
    }
    for col in bool_cols:
        if col in df.columns:
            s = df[col].astype("string").str.strip().str.lower().replace(bool_map)
            df[col] = s.where(s.isin(["yes", "no"]), other=pd.NA).astype("string")

    def to_numeric_series(series: pd.Series) -> pd.Series:
        s = series.astype("string").str.strip()
        s = s.str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
        s = s.str.replace(r"[^0-9\.\-]+", "", regex=True)
        return pd.to_numeric(s, errors="coerce")

    for col in ["build_year", "number_rooms", "facades", "living_area"]:
        if col in df.columns:
            df[col] = to_numeric_series(df[col])

    if "price" not in df.columns:
        raise KeyError("Column 'price' not found in dataset.")
    df["price"] = to_numeric_series(df["price"])

    if "postal_code" in df.columns:
        df["postal_code"] = (
            df["postal_code"]
            .astype("string")
            .str.strip()
            .str.extract(r"(\d{4})", expand=False)
            .astype("string")
        )
    else:
        df["postal_code"] = pd.NA

    if "province" in df.columns:
        df["province"] = df["province"].apply(_norm_text).astype("string")
    else:
        df["province"] = pd.NA

    ref = load_bpost_csv(postal_ref_csv).copy()
    ref.columns = [c.strip() for c in ref.columns]
    colmap = {c.lower(): c for c in ref.columns}

    postcode_col = colmap.get("postcode")
    provincie_col = colmap.get("provincie")
    if postcode_col is None or provincie_col is None:
        raise KeyError(
            f"Expected columns like 'Postcode' and 'Provincie' in {postal_ref_csv}. "
            f"Found: {list(ref.columns)}"
        )

    ref = ref.rename(columns={postcode_col: "postal_code", provincie_col: "province_bpost"})
    ref["postal_code"] = ref["postal_code"].astype(str).str.strip().str.extract(r"(\d{4})", expand=False)
    ref = ref.dropna(subset=["postal_code"])
    ref["postal_code"] = ref["postal_code"].astype(str).str.zfill(4)
    ref["province_bpost"] = ref["province_bpost"].apply(_norm_text).astype("string")
    ref = ref[["postal_code", "province_bpost"]].drop_duplicates("postal_code")

    df = df.merge(ref, on="postal_code", how="left")
    df["province"] = df["province_bpost"].fillna(df["province"])
    df = df.drop(columns=["province_bpost"])
    df["province"] = df["province"].astype("string").str.strip().str.upper()

    if "build_year" in df.columns:
        df.loc[(df["build_year"] < 1800) | (df["build_year"] > 2025), "build_year"] = np.nan

    if "number_rooms" in df.columns:
        df.loc[(df["number_rooms"] <= 0) | (df["number_rooms"] > 12), "number_rooms"] = np.nan

    if "living_area" in df.columns:
        df.loc[(df["living_area"] < 10) | (df["living_area"] > 500), "living_area"] = np.nan

    df = df[df["price"].notna()]
    df = df[df["price"] >= 10_000]
    df.loc[df["price"] > 7_500_000, "price"] = np.nan
    df = df[df["price"].notna()]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"Enhanced Cleaned dataset saved to: {save_path}")
    print("Final shape:", df.shape)
    print("Share missing postal_code:", df["postal_code"].isna().mean())
    print("Share missing province:", df["province"].isna().mean())

    return df


if __name__ == "__main__":
    enhanced_clean()
