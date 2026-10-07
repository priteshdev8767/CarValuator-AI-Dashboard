"""
Advanced Feature Engineering Module
=====================================
Creates derived features that capture depreciation patterns,
usage intensity, power efficiency, and market segmentation.
None of these features use the target variable (no data leakage).
"""
import pandas as pd
import numpy as np

CURRENT_YEAR = 2024

# ── Feature Column Lists (used across the project) ────────────────────
NUMERICAL_FEATURES = [
    'Present_Price', 'Kms_Driven', 'Engine_CC', 'Max_Power_bhp',
    'Mileage_kmpl', 'Seats', 'Car_Age', 'Km_Per_Year',
    'Power_Per_CC', 'Log_Kms_Driven', 'Owner_Num', 'Is_Luxury',
]

CATEGORICAL_FEATURES = ['Brand', 'Fuel_Type', 'Seller_Type', 'Transmission']

TARGET = 'Selling_Price'

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def create_features(df):
    """
    Create advanced derived features from raw car data.
    
    New features created:
      - Car_Age          : Years since manufacture
      - Km_Per_Year      : Average annual usage
      - Power_Per_CC     : Engine efficiency ratio
      - Log_Kms_Driven   : Log-transformed kms (reduces skew)
      - Owner_Num        : Numerical owner encoding
      - Is_Luxury        : Binary luxury brand flag
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw or preprocessed car data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with new features added.
    """
    df = df.copy()

    print("\n" + "=" * 60)
    print("🧬  FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    # 1. Car Age
    df['Car_Age'] = CURRENT_YEAR - df['Year']
    print("  ✅ Car_Age          — years since manufacture")

    # 2. Kilometers per Year (usage intensity)
    df['Km_Per_Year'] = df['Kms_Driven'] / df['Car_Age'].clip(lower=1)
    print("  ✅ Km_Per_Year      — average annual usage")

    # 3. Power per CC (engine efficiency)
    df['Power_Per_CC'] = (
        df['Max_Power_bhp'] / df['Engine_CC'].clip(lower=1) * 1000
    )
    print("  ✅ Power_Per_CC     — engine power efficiency")

    # 4. Log of Kms Driven (reduce right skew)
    df['Log_Kms_Driven'] = np.log1p(df['Kms_Driven'])
    print("  ✅ Log_Kms_Driven   — log-transformed distance")

    # 5. Owner as numerical
    owner_map = {'First': 1, 'Second': 2, 'Third': 3, 'Fourth & Above': 4}
    df['Owner_Num'] = df['Owner'].map(owner_map).fillna(1).astype(int)
    print("  ✅ Owner_Num        — numerical owner count")

    # 6. Is Luxury brand flag
    luxury_brands = ['BMW', 'Audi', 'Mercedes']
    df['Is_Luxury'] = df['Brand'].isin(luxury_brands).astype(int)
    print("  ✅ Is_Luxury        — luxury brand indicator")

    print(f"\n  ✅ Feature engineering complete: {df.shape[1]} total columns")
    print("=" * 60)
    return df
