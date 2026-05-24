"""
Data Cleaning Script for Insurance Risk Analytics

Reads the raw MachineLearningRating_v3.txt dataset and produces a cleaned
version at data/cleaned_insurance_data.csv.

Cleaning steps:
1. Strip whitespace from string columns
2. Convert TransactionMonth to datetime
3. Convert numeric columns stored as strings to proper numeric types
4. Handle missing values:
   - Fill missing CrossBorder and NumberOfVehiclesInFleet with 0
   - Fill missing categorical fields (NewVehicle, WrittenOff, Rebuilt, Converted) with "Unknown"
5. Standardize text values (e.g., "Not specified" → NaN for Gender/MaritalStatus)
6. Drop columns that are completely empty
7. Remove duplicate rows

Usage:
    python src/data_cleaning.py
"""

import os
import pandas as pd
import numpy as np


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the raw pipe-delimited dataset."""
    print(f"Loading raw data from {filepath}...")
    df = pd.read_csv(filepath, sep="|", low_memory=False)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning transformations."""
    print("Cleaning data...")
    initial_rows = len(df)

    # 1. Strip whitespace from string columns
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
        # Replace empty strings and whitespace-only with NaN
        df[col] = df[col].replace(["", "nan", "None"], np.nan)

    # 2. Convert TransactionMonth to datetime
    if "TransactionMonth" in df.columns:
        df["TransactionMonth"] = pd.to_datetime(
            df["TransactionMonth"], errors="coerce"
        )

    # 3. Convert numeric columns that may have been read as strings
    numeric_candidates = [
        "TotalPremium", "TotalClaims", "SumInsured",
        "CalculatedPremiumPerTerm", "CustomValueEstimate",
        "CrossBorder", "NumberOfVehiclesInFleet",
        "CapitalOutstanding", "cubiccapacity", "kilowatts",
        "Cylinders", "NumberOfDoors", "PostalCode",
        "RegistrationYear"
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Handle missing values for specific columns
    fill_zero_cols = ["CrossBorder", "NumberOfVehiclesInFleet"]
    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    fill_unknown_cols = ["NewVehicle", "WrittenOff", "Rebuilt", "Converted"]
    for col in fill_unknown_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # 5. Standardize "Not specified" values in categorical columns
    categorical_cols = ["Gender", "MaritalStatus", "Citizenship", "Title"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].replace("Not specified", np.nan)

    # 6. Drop columns that are entirely NaN
    empty_cols = df.columns[df.isnull().all()]
    if len(empty_cols) > 0:
        print(f"  Dropping {len(empty_cols)} entirely empty columns: {list(empty_cols)}")
        df = df.drop(columns=empty_cols)

    # 7. Remove duplicate rows
    before_dedup = len(df)
    df = df.drop_duplicates()
    dupes_removed = before_dedup - len(df)
    if dupes_removed > 0:
        print(f"  Removed {dupes_removed:,} duplicate rows")

    final_rows = len(df)
    print(f"  Cleaning complete: {initial_rows:,} → {final_rows:,} rows "
          f"({initial_rows - final_rows:,} removed)")
    print(f"  Final shape: {df.shape}")

    return df


def save_cleaned_data(df: pd.DataFrame, filepath: str) -> None:
    """Save cleaned DataFrame to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  Saved cleaned data to {filepath} ({size_mb:.1f} MB)")


def main():
    raw_path = os.path.join("data", "MachineLearningRating_v3.txt")
    cleaned_path = os.path.join("data", "cleaned_insurance_data.csv")

    # Load
    df = load_raw_data(raw_path)

    # Clean
    df = clean_data(df)

    # Save
    save_cleaned_data(df, cleaned_path)

    # Print summary statistics
    print("\n--- Cleaned Data Summary ---")
    print(f"Missing values per column (top 10):")
    missing = df.isnull().sum().sort_values(ascending=False).head(10)
    for col, count in missing.items():
        pct = count / len(df) * 100
        print(f"  {col}: {count:,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
