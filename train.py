"""
train.py — Main Training Pipeline
===================================
Orchestrates the full ML workflow:
  1. Generate / load dataset
  2. Preprocess data
  3. Engineer features
  4. Train 7 models + Stacking Ensemble
  5. Evaluate & save results, plots, and models
"""
import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.generate_dataset import generate_car_dataset
from src.preprocessing import preprocess_data
from src.feature_engineering import (
    create_features,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
    TARGET,
)
from src.models import train_all_models, train_stacking_ensemble
from src.evaluation import (
    create_comparison_table,
    plot_model_comparison,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance,
)


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, 'data')
    model_dir = os.path.join(project_root, 'models')
    plot_dir = os.path.join(project_root, 'plots')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # ── Step 1: Generate Dataset ──────────────────────────────────────
    data_path = os.path.join(data_dir, 'car_data.csv')
    if not os.path.exists(data_path):
        print("\n🚗  Generating synthetic car dataset ...")
        df_raw = generate_car_dataset(n_samples=5000)
        df_raw.to_csv(data_path, index=False)
        print(f"  ✅ Saved to {data_path}")
    else:
        print(f"\n🚗  Dataset already exists: {data_path}")

    # ── Step 2: Preprocess ────────────────────────────────────────────
    df = preprocess_data(data_path)

    # ── Step 3: Feature Engineering ───────────────────────────────────
    df = create_features(df)

    # ── Step 4: Prepare X and y ───────────────────────────────────────
    X = df[ALL_FEATURES]
    y = df[TARGET]

    print(f"\n📐  Feature matrix : {X.shape}")
    print(f"📐  Target vector  : {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"📐  Train set      : {X_train.shape[0]} samples")
    print(f"📐  Test set       : {X_test.shape[0]} samples")

    # ── Step 5: Train Individual Models ───────────────────────────────
    results, pipelines = train_all_models(X_train, y_train, X_test, y_test)

    # ── Step 6: Train Stacking Ensemble ───────────────────────────────
    stack_result, stacking_model = train_stacking_ensemble(
        X_train, y_train, X_test, y_test
    )
    results['Stacking Ensemble'] = stack_result

    # ── Step 7: Comparison Table ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("🏆  MODEL COMPARISON (sorted by Test R²)")
    print("=" * 60)
    comparison = create_comparison_table(results)
    print(comparison.to_string(index=False))
    print("=" * 60)

    # Best model
    best_name = comparison.iloc[0]['Model']
    best_r2 = comparison.iloc[0]['Test R²']
    print(f"\n🥇  Best Model: {best_name} (Test R² = {best_r2:.4f})")

    # ── Step 8: Generate Plots ────────────────────────────────────────
    print("\n📊  Generating evaluation plots ...")
    plot_model_comparison(results, save_dir=plot_dir)

    # Plots for the best pipeline model (skip if best is Stacking)
    if best_name in pipelines:
        best_pipeline = pipelines[best_name]
    else:
        # If stacking is best, use the top individual model for plots
        individual_best = comparison[comparison['Model'] != 'Stacking Ensemble'].iloc[0]['Model']
        best_pipeline = pipelines[individual_best]
        best_name_for_plots = individual_best
        print(f"  ℹ️  Using {individual_best} for detailed plots (Stacking has no single pipeline)")

    plot_best_name = best_name if best_name in pipelines else individual_best

    y_pred_test = (
        pipelines[plot_best_name].predict(X_test)
        if plot_best_name in pipelines
        else stacking_model.predict(X_test)
    )

    plot_actual_vs_predicted(y_test.values, y_pred_test, plot_best_name, save_dir=plot_dir)
    plot_residuals(y_test.values, y_pred_test, plot_best_name, save_dir=plot_dir)

    # Feature importance for tree-based models
    for name in ['Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting']:
        if name in pipelines:
            plot_feature_importance(pipelines[name], name, save_dir=plot_dir)

    # ── Step 9: Save Models & Config ──────────────────────────────────
    print("\n💾  Saving models ...")

    joblib.dump(pipelines, os.path.join(model_dir, 'all_pipelines.joblib'))
    print(f"  ✅ Saved all pipelines → models/all_pipelines.joblib")

    joblib.dump(stacking_model, os.path.join(model_dir, 'stacking_ensemble.joblib'))
    print(f"  ✅ Saved stacking model → models/stacking_ensemble.joblib")

    # Save a sample of training data (for SHAP in the app)
    sample_size = min(200, len(X_train))
    train_sample = X_train.sample(n=sample_size, random_state=42)
    joblib.dump(train_sample, os.path.join(model_dir, 'train_sample.joblib'))
    print(f"  ✅ Saved training sample → models/train_sample.joblib")

    # Save config
    config = {
        'results': results,
        'best_model': best_name,
        'best_r2': best_r2,
        'numerical_features': NUMERICAL_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'all_features': ALL_FEATURES,
        'target': TARGET,
        'train_size': len(X_train),
        'test_size': len(X_test),
    }
    joblib.dump(config, os.path.join(model_dir, 'config.joblib'))
    print(f"  ✅ Saved config → models/config.joblib")

    # Save comparison table
    comparison.to_csv(os.path.join(model_dir, 'model_comparison.csv'), index=False)
    print(f"  ✅ Saved comparison → models/model_comparison.csv")

    print("\n" + "=" * 60)
    print("🎉  TRAINING PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Review plots in: {plot_dir}/")
    print(f"  2. Launch dashboard: streamlit run app.py")
    print()


if __name__ == '__main__':
    main()
