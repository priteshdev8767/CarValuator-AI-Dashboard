"""
Data Preprocessing Module
==========================
Handles data loading, missing value imputation,
outlier removal, and data quality checks.
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def load_data(filepath):
    """Load car dataset from CSV file."""
    df = pd.read_csv(filepath)
    print(f"  ✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def check_data_quality(df):
    """Print a data quality report."""
    print("\n  📋 Data Quality Report:")
    print(f"     Total rows       : {df.shape[0]}")
    print(f"     Total columns    : {df.shape[1]}")
    print(f"     Duplicate rows   : {df.duplicated().sum()}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"     Missing values   :")
        for col, count in missing.items():
            pct = count / len(df) * 100
            print(f"       - {col}: {count} ({pct:.1f}%)")
    else:
        print(f"     Missing values   : None")
    return df


def handle_missing_values(df):
    """Impute missing values using median (numerical) and mode (categorical)."""
    df = df.copy()

    # Numerical columns → fill with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"     ➜ {col}: filled {n_missing} nulls with median ({median_val})")

    # Categorical columns → fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"     ➜ {col}: filled {n_missing} nulls with mode ({mode_val})")

    print("  ✅ Missing values handled")
    return df


def remove_outliers(df, target_col='Selling_Price', z_threshold=3.5):
    """Remove outliers from the target column using Z-score method."""
    df = df.copy()
    initial_rows = len(df)

    z_scores = np.abs(stats.zscore(df[target_col].values))
    df = df[z_scores < z_threshold].reset_index(drop=True)

    removed = initial_rows - len(df)
    print(f"  ✅ Outliers removed: {removed} rows (Z-score > {z_threshold})")
    return df


def remove_duplicates(df):
    """Remove duplicate rows."""
    initial = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = initial - len(df)
    print(f"  ✅ Duplicates removed: {removed} rows")
    return df


def preprocess_data(filepath):
    """
    Complete preprocessing pipeline.
    
    Parameters
    ----------
    filepath : str
        Path to the raw CSV file.
    
    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed DataFrame.
    """
    print("\n" + "=" * 60)
    print("🔧  DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    df = load_data(filepath)
    df = check_data_quality(df)

    print("\n  📌 Step 1: Handling missing values...")
    df = handle_missing_values(df)

    print("\n  📌 Step 2: Removing outliers...")
    df = remove_outliers(df)

    print("\n  📌 Step 3: Removing duplicates...")
    df = remove_duplicates(df)

    print(f"\n  ✅ Preprocessing complete: {df.shape[0]} rows × {df.shape[1]} columns")
    print("=" * 60)
    return df
