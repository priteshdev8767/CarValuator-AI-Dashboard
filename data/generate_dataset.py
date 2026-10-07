"""
Realistic Synthetic Car Dataset Generator
==========================================
Generates ~5000 records of Indian used car market data with
realistic correlations between features like depreciation,
mileage, brand tiers, and ownership patterns.
"""
import pandas as pd
import numpy as np
import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def generate_car_dataset(n_samples=5000, random_state=42):
    """
    Generate a realistic synthetic car dataset for the Indian used car market.
    
    Parameters
    ----------
    n_samples : int
        Number of records to generate.
    random_state : int
        Random seed for reproducibility.
    
    Returns
    -------
    pd.DataFrame
        Generated car dataset.
    """
    np.random.seed(random_state)
    current_year = 2024

    # ── Car Catalog with Realistic Specifications ──────────────────────
    catalog = {
        'Maruti': {
            'tier': 'Budget',
            'models': {
                'Swift':         {'price': 6.5,  'engine': 1197, 'power': 82,  'mileage': 22.0, 'seats': 5},
                'Baleno':        {'price': 7.0,  'engine': 1197, 'power': 83,  'mileage': 22.4, 'seats': 5},
                'Alto':          {'price': 3.5,  'engine': 796,  'power': 47,  'mileage': 25.0, 'seats': 5},
                'Dzire':         {'price': 7.5,  'engine': 1197, 'power': 82,  'mileage': 23.3, 'seats': 5},
                'Vitara Brezza': {'price': 9.5,  'engine': 1462, 'power': 103, 'mileage': 18.8, 'seats': 5},
                'Ciaz':          {'price': 10.0, 'engine': 1462, 'power': 103, 'mileage': 20.3, 'seats': 5},
                'Ertiga':        {'price': 9.0,  'engine': 1462, 'power': 103, 'mileage': 19.3, 'seats': 7},
                'WagonR':        {'price': 5.5,  'engine': 998,  'power': 67,  'mileage': 24.4, 'seats': 5},
            }
        },
        'Hyundai': {
            'tier': 'Budget',
            'models': {
                'i20':     {'price': 8.0,  'engine': 1197, 'power': 83,  'mileage': 20.4, 'seats': 5},
                'Creta':   {'price': 12.0, 'engine': 1497, 'power': 113, 'mileage': 17.0, 'seats': 5},
                'Venue':   {'price': 8.5,  'engine': 1197, 'power': 83,  'mileage': 18.2, 'seats': 5},
                'Verna':   {'price': 10.5, 'engine': 1497, 'power': 113, 'mileage': 18.6, 'seats': 5},
                'i10':     {'price': 5.0,  'engine': 1086, 'power': 69,  'mileage': 20.4, 'seats': 5},
                'Tucson':  {'price': 23.0, 'engine': 1995, 'power': 185, 'mileage': 14.0, 'seats': 5},
                'Santro':  {'price': 5.5,  'engine': 1086, 'power': 69,  'mileage': 20.3, 'seats': 5},
            }
        },
        'Honda': {
            'tier': 'Mid',
            'models': {
                'City':  {'price': 12.0, 'engine': 1497, 'power': 119, 'mileage': 17.8, 'seats': 5},
                'Amaze': {'price': 7.5,  'engine': 1199, 'power': 90,  'mileage': 18.6, 'seats': 5},
                'WRV':   {'price': 9.5,  'engine': 1199, 'power': 90,  'mileage': 17.0, 'seats': 5},
                'Jazz':  {'price': 8.5,  'engine': 1199, 'power': 90,  'mileage': 18.0, 'seats': 5},
                'Civic': {'price': 20.0, 'engine': 1799, 'power': 141, 'mileage': 16.5, 'seats': 5},
            }
        },
        'Toyota': {
            'tier': 'Mid',
            'models': {
                'Innova Crysta': {'price': 18.0, 'engine': 2393, 'power': 148, 'mileage': 12.0, 'seats': 7},
                'Fortuner':      {'price': 35.0, 'engine': 2755, 'power': 177, 'mileage': 10.0, 'seats': 7},
                'Glanza':        {'price': 7.5,  'engine': 1197, 'power': 90,  'mileage': 22.4, 'seats': 5},
                'Urban Cruiser': {'price': 9.0,  'engine': 1462, 'power': 103, 'mileage': 18.8, 'seats': 5},
                'Camry':         {'price': 42.0, 'engine': 2487, 'power': 178, 'mileage': 19.1, 'seats': 5},
            }
        },
        'Tata': {
            'tier': 'Budget',
            'models': {
                'Nexon':  {'price': 8.0,  'engine': 1199, 'power': 120, 'mileage': 17.5, 'seats': 5},
                'Harrier': {'price': 16.0, 'engine': 1956, 'power': 170, 'mileage': 14.6, 'seats': 5},
                'Altroz': {'price': 7.0,  'engine': 1199, 'power': 86,  'mileage': 20.5, 'seats': 5},
                'Tiago':  {'price': 5.5,  'engine': 1199, 'power': 86,  'mileage': 20.1, 'seats': 5},
                'Safari': {'price': 17.0, 'engine': 1956, 'power': 170, 'mileage': 14.1, 'seats': 7},
                'Punch':  {'price': 6.0,  'engine': 1199, 'power': 86,  'mileage': 18.2, 'seats': 5},
            }
        },
        'Mahindra': {
            'tier': 'Mid',
            'models': {
                'XUV700': {'price': 14.0, 'engine': 1997, 'power': 200, 'mileage': 16.0, 'seats': 7},
                'Scorpio': {'price': 13.0, 'engine': 2179, 'power': 140, 'mileage': 15.4, 'seats': 7},
                'Thar':   {'price': 13.0, 'engine': 1997, 'power': 150, 'mileage': 15.2, 'seats': 4},
                'XUV300': {'price': 9.0,  'engine': 1497, 'power': 115, 'mileage': 17.0, 'seats': 5},
                'Bolero': {'price': 9.5,  'engine': 1493, 'power': 76,  'mileage': 16.5, 'seats': 7},
            }
        },
        'Kia': {
            'tier': 'Mid',
            'models': {
                'Seltos':   {'price': 11.0, 'engine': 1497, 'power': 115, 'mileage': 17.0, 'seats': 5},
                'Sonet':    {'price': 8.0,  'engine': 1197, 'power': 83,  'mileage': 18.2, 'seats': 5},
                'Carnival': {'price': 30.0, 'engine': 2199, 'power': 200, 'mileage': 13.9, 'seats': 7},
            }
        },
        'BMW': {
            'tier': 'Luxury',
            'models': {
                '3 Series': {'price': 45.0, 'engine': 1998, 'power': 258, 'mileage': 15.4, 'seats': 5},
                '5 Series': {'price': 65.0, 'engine': 1998, 'power': 252, 'mileage': 14.5, 'seats': 5},
                'X1':       {'price': 40.0, 'engine': 1998, 'power': 192, 'mileage': 16.8, 'seats': 5},
                'X3':       {'price': 55.0, 'engine': 1998, 'power': 252, 'mileage': 14.7, 'seats': 5},
                'X5':       {'price': 80.0, 'engine': 2993, 'power': 265, 'mileage': 12.0, 'seats': 5},
            }
        },
        'Audi': {
            'tier': 'Luxury',
            'models': {
                'A4': {'price': 43.0, 'engine': 1984, 'power': 190, 'mileage': 17.7, 'seats': 5},
                'A6': {'price': 60.0, 'engine': 1984, 'power': 245, 'mileage': 14.1, 'seats': 5},
                'Q3': {'price': 38.0, 'engine': 1984, 'power': 190, 'mileage': 17.0, 'seats': 5},
                'Q5': {'price': 55.0, 'engine': 1984, 'power': 252, 'mileage': 14.2, 'seats': 5},
                'Q7': {'price': 80.0, 'engine': 2995, 'power': 249, 'mileage': 11.7, 'seats': 7},
            }
        },
        'Mercedes': {
            'tier': 'Luxury',
            'models': {
                'C-Class': {'price': 50.0, 'engine': 1991, 'power': 204, 'mileage': 14.1, 'seats': 5},
                'E-Class': {'price': 65.0, 'engine': 1991, 'power': 197, 'mileage': 14.1, 'seats': 5},
                'GLA':     {'price': 40.0, 'engine': 1332, 'power': 163, 'mileage': 17.0, 'seats': 5},
                'GLC':     {'price': 60.0, 'engine': 1991, 'power': 197, 'mileage': 14.0, 'seats': 5},
                'GLE':     {'price': 85.0, 'engine': 2925, 'power': 272, 'mileage': 12.0, 'seats': 5},
            }
        },
    }

    # ── Brand Market Share Weights ─────────────────────────────────────
    brand_names = list(catalog.keys())
    brand_weights = [0.25, 0.18, 0.06, 0.07, 0.14, 0.10, 0.08, 0.04, 0.04, 0.04]

    records = []

    for _ in range(n_samples):
        # Pick brand by market share weight
        brand = np.random.choice(brand_names, p=brand_weights)
        brand_info = catalog[brand]
        tier = brand_info['tier']

        # Pick random model within brand
        model_name = np.random.choice(list(brand_info['models'].keys()))
        specs = brand_info['models'][model_name]

        # Year of manufacture (2005–2023)
        year = int(np.random.randint(2005, 2024))
        car_age = current_year - year

        # Present (showroom) price with ±10% variance
        present_price = round(specs['price'] * np.random.uniform(0.90, 1.10), 2)

        # Specs with small variance
        engine = int(specs['engine'] * np.random.uniform(0.98, 1.02))
        power = round(specs['power'] * np.random.uniform(0.95, 1.05), 1)
        mileage = round(specs['mileage'] * np.random.uniform(0.90, 1.10), 1)
        seats = specs['seats']

        # Fuel type (realistic distribution per tier)
        if tier == 'Luxury':
            fuel_type = np.random.choice(['Petrol', 'Diesel'], p=[0.50, 0.50])
        elif engine > 1500:
            fuel_type = np.random.choice(['Petrol', 'Diesel', 'CNG'], p=[0.40, 0.50, 0.10])
        else:
            fuel_type = np.random.choice(['Petrol', 'Diesel', 'CNG'], p=[0.60, 0.25, 0.15])

        # Transmission
        if tier == 'Luxury':
            transmission = np.random.choice(['Manual', 'Automatic'], p=[0.10, 0.90])
        else:
            transmission = np.random.choice(['Manual', 'Automatic'], p=[0.65, 0.35])

        # Seller type
        seller_type = np.random.choice(
            ['Dealer', 'Individual', 'Trustmark Dealer'], p=[0.45, 0.40, 0.15]
        )

        # Owner history (correlated with car age)
        if car_age <= 3:
            owner = np.random.choice(['First', 'Second'], p=[0.85, 0.15])
        elif car_age <= 7:
            owner = np.random.choice(['First', 'Second', 'Third'], p=[0.40, 0.45, 0.15])
        else:
            owner = np.random.choice(
                ['First', 'Second', 'Third', 'Fourth & Above'],
                p=[0.15, 0.35, 0.35, 0.15]
            )

        # Kms driven (correlated with age)
        avg_km_per_year = np.random.uniform(8000, 20000)
        kms_driven = int(car_age * avg_km_per_year * np.random.uniform(0.70, 1.30))
        kms_driven = max(500, kms_driven)

        # ── Selling Price Calculation (Realistic Depreciation) ─────────
        depreciation_rates = {'Budget': 0.12, 'Mid': 0.14, 'Luxury': 0.18}
        annual_dep = depreciation_rates[tier]
        depreciation_factor = (1 - annual_dep) ** car_age

        fuel_adj = {'Petrol': 1.00, 'Diesel': 1.08, 'CNG': 0.92}[fuel_type]
        trans_adj = 1.05 if transmission == 'Automatic' else 1.00
        owner_adj = {'First': 1.00, 'Second': 0.92, 'Third': 0.84, 'Fourth & Above': 0.75}[owner]
        km_factor = max(0.70, 1.0 - (kms_driven / 500000) * 0.15)
        seller_adj = {'Dealer': 1.05, 'Trustmark Dealer': 1.08, 'Individual': 0.95}[seller_type]

        selling_price = (
            present_price * depreciation_factor * fuel_adj
            * trans_adj * owner_adj * km_factor * seller_adj
        )
        # Add ±10% random noise
        selling_price *= np.random.uniform(0.90, 1.10)
        selling_price = round(max(0.30, selling_price), 2)

        records.append({
            'Brand': brand,
            'Model': model_name,
            'Year': year,
            'Present_Price': present_price,
            'Kms_Driven': kms_driven,
            'Fuel_Type': fuel_type,
            'Seller_Type': seller_type,
            'Transmission': transmission,
            'Owner': owner,
            'Engine_CC': engine,
            'Max_Power_bhp': power,
            'Mileage_kmpl': mileage,
            'Seats': seats,
            'Selling_Price': selling_price,
        })

    df = pd.DataFrame(records)

    # Introduce ~3% missing values in some columns (realistic)
    for col in ['Engine_CC', 'Max_Power_bhp', 'Mileage_kmpl', 'Seats']:
        mask = np.random.random(len(df)) < 0.03
        df.loc[mask, col] = np.nan

    return df


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(script_dir, exist_ok=True)

    df = generate_car_dataset(n_samples=5000)
    output_path = os.path.join(script_dir, 'car_data.csv')
    df.to_csv(output_path, index=False)

    print(f"✅ Dataset generated successfully!")
    print(f"   📁 Path : {output_path}")
    print(f"   📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n{'='*50}")
    print("First 5 rows:")
    print(df.head().to_string())
    print(f"\n{'='*50}")
    print("Column Info:")
    print(df.info())
    print(f"\n{'='*50}")
    print("Statistics:")
    print(df.describe().to_string())
