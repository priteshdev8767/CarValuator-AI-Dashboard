"""
eda.py — Exploratory Data Analysis
====================================
Generates 10+ publication-quality visualizations
to understand the car price dataset.
"""
import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import preprocess_data
from src.feature_engineering import create_features


def run_eda(data_path='data/car_data.csv', save_dir='plots/eda'):
    """Generate all EDA plots and save to disk."""
    os.makedirs(save_dir, exist_ok=True)

    # Load & preprocess
    df = preprocess_data(data_path)
    df = create_features(df)

    print("\n" + "=" * 60)
    print("📊  EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # ── 1. Selling Price Distribution ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Selling Price Distribution', fontsize=15, fontweight='bold')

    axes[0].hist(df['Selling_Price'], bins=50, color='#7b2ff7', edgecolor='white', alpha=0.85)
    axes[0].set_xlabel('Selling Price (Lakhs)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Histogram')

    axes[1].hist(np.log1p(df['Selling_Price']), bins=50, color='#00d2ff', edgecolor='white', alpha=0.85)
    axes[1].set_xlabel('Log(Selling Price)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Log-Transformed')

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, '01_price_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✅ 01_price_distribution.png")

    # ── 2. Price by Brand (Box Plot) ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    brand_order = df.groupby('Brand')['Selling_Price'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='Brand', y='Selling_Price', order=brand_order, palette='viridis', ax=ax)
    ax.set_title('Selling Price by Brand', fontsize=14, fontweight='bold')
    ax.set_xlabel('Brand')
    ax.set_ylabel('Selling Price (Lakhs)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, '02_price_by_brand.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✅ 02_price_by_brand.png")

    # ── 3. Price vs Car Age ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df['Car_Age'], df['Selling_Price'],
        c=df['Present_Price'], cmap='plasma', alpha=0.5, s=15, edgecolors='none'
    )
    plt.colorbar(scatter, label='Present Price (Lakhs)')
    ax.set_xlabel('Car Age (Years)')
    ax.set_ylabel('Selling Price (Lakhs)')
    ax.set_title('Selling Price vs Car Age (colored by Present Price)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, '03_price_vs_age.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✅ 03_price_vs_age.png")

    # ── 4. Price vs Kms Driven ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['Kms_Driven'], df['Selling_Price'], alpha=0.4, s=15, color='#00d2ff', edgecolors='none')
    ax.set_xlabel('Kilometers Driven')
    ax.set_ylabel('Selling Price (Lakhs)')
    ax.set_title('Selling Price vs Kilometers Driven', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, '04_price_vs_kms.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✅ 04_price_vs_kms.png")

    # ── 5. Correlation Heatmap ────────────────────────────────────────
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numerical_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, '05_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✅ 05_correlation_heatmap.png")

    # ── 6. Fuel Type & Transmission Distribution ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df['Fuel_Type'].value_counts().plot.pie(
        autopct='%1.1f%%', colors=sns.color_palette('viridis', 3),
        ax=axes[0], startangle=90, textprops={'fontsize': 10}
    )
    axes[0].set_title('Fuel Type Distribution', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('')

    df['Transmission'].value_counts().plot.pie(
        autopct='%1.1f%%', colors=['#7b2ff7', '#00d2ff'],
        ax=axes[1], startangle=90, textprops={'fontsize': 10}
    )
    axes[1].set_title('Transmission Distribution', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('')

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, '06_fuel_transmission_dist.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✅ 06_fuel_transmission_dist.png")

    # ── 7. Price by Fuel Type (Violin) ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x='Fuel_Type', y='Selling_Price', palette='viridis', ax=ax)
    ax.set_title('Selling Price by Fuel Type', fontsize=13, fontweight='bold')
    ax.set_xlabel('Fuel Type')
    ax.set_ylabel('Selling Price (Lakhs)')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, '07_price_by_fuel.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✅ 07_price_by_fuel.png")

    # ── 8. Brand Market Share ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    brand_counts = df['Brand'].value_counts()
    bars = ax.bar(brand_counts.index, brand_counts.values, color=sns.color_palette('viridis', len(brand_counts)))
    ax.set_title('Brand Distribution (Market Share)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Brand')
    ax.set_ylabel('Count')
    for bar, val in zip(bars, brand_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(val), ha='center', fontsize=9)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, '08_brand_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✅ 08_brand_distribution.png")

    # ── 9. Depreciation Curve by Brand Tier ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for tier, color in [('Budget', '#00d2ff'), ('Mid', '#7b2ff7'), ('Luxury', '#ff6b6b')]:
        tier_brands = {
            'Budget': ['Maruti', 'Hyundai', 'Tata'],
            'Mid': ['Honda', 'Toyota', 'Mahindra', 'Kia'],
            'Luxury': ['BMW', 'Audi', 'Mercedes'],
        }[tier]
        subset = df[df['Brand'].isin(tier_brands)]
        grouped = subset.groupby('Car_Age').agg(
            avg_retention=('Selling_Price', 'mean'),
            avg_present=('Present_Price', 'mean'),
        )
        grouped['retention_pct'] = (grouped['avg_retention'] / grouped['avg_present']) * 100
        ax.plot(grouped.index, grouped['retention_pct'], 'o-', color=color, label=tier, markersize=5)

    ax.set_xlabel('Car Age (Years)')
    ax.set_ylabel('Value Retention (%)')
    ax.set_title('Depreciation Curve by Brand Tier', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, '09_depreciation_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✅ 09_depreciation_curve.png")

    # ── 10. Owner vs Price ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    owner_order = ['First', 'Second', 'Third', 'Fourth & Above']
    sns.boxplot(data=df, x='Owner', y='Selling_Price', order=owner_order, palette='viridis', ax=ax)
    ax.set_title('Selling Price by Ownership History', fontsize=13, fontweight='bold')
    ax.set_xlabel('Owner')
    ax.set_ylabel('Selling Price (Lakhs)')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, '10_price_by_owner.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✅ 10_price_by_owner.png")

    print(f"\n  🎉 All EDA plots saved to: {save_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    run_eda()
