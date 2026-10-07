"""
Model Evaluation & Visualization Module
=========================================
Creates comparison tables and publication-quality plots
for model performance analysis.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Use a clean style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')


def create_comparison_table(results):
    """
    Build a DataFrame comparing all models on test metrics.

    Parameters
    ----------
    results : dict
        {model_name: {'train': metrics, 'test': metrics}}

    Returns
    -------
    pd.DataFrame sorted by Test R² descending.
    """
    rows = []
    for name, res in results.items():
        rows.append({
            'Model': name,
            'Train R²': res['train']['R2 Score'],
            'Test R²': res['test']['R2 Score'],
            'Train MAE': res['train']['MAE'],
            'Test MAE': res['test']['MAE'],
            'Train RMSE': res['train']['RMSE'],
            'Test RMSE': res['test']['RMSE'],
            'Test MAPE (%)': res['test']['MAPE'],
        })
    df = pd.DataFrame(rows).sort_values('Test R²', ascending=False)
    return df.reset_index(drop=True)


def plot_model_comparison(results, save_dir='plots'):
    """Bar chart comparing R², MAE and RMSE across all models."""
    os.makedirs(save_dir, exist_ok=True)
    df = create_comparison_table(results)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)

    colors = sns.color_palette('viridis', n_colors=len(df))

    # R² Score
    axes[0].barh(df['Model'], df['Test R²'], color=colors)
    axes[0].set_xlabel('R² Score')
    axes[0].set_title('Test R² Score (Higher is Better)')
    for i, v in enumerate(df['Test R²']):
        axes[0].text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)

    # MAE
    axes[1].barh(df['Model'], df['Test MAE'], color=colors)
    axes[1].set_xlabel('MAE (Lakhs)')
    axes[1].set_title('Test MAE (Lower is Better)')
    for i, v in enumerate(df['Test MAE']):
        axes[1].text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9)

    # RMSE
    axes[2].barh(df['Model'], df['Test RMSE'], color=colors)
    axes[2].set_xlabel('RMSE (Lakhs)')
    axes[2].set_title('Test RMSE (Lower is Better)')
    for i, v in enumerate(df['Test RMSE']):
        axes[2].text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, 'model_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Saved: {path}")


def plot_actual_vs_predicted(y_true, y_pred, model_name, save_dir='plots'):
    """Scatter plot of actual vs predicted values."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.4, s=20, color='#7b2ff7', edgecolors='white', linewidth=0.3)

    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('Actual Price (Lakhs)', fontsize=12)
    ax.set_ylabel('Predicted Price (Lakhs)', fontsize=12)
    ax.set_title(f'{model_name} — Actual vs Predicted', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)

    plt.tight_layout()
    safe_name = model_name.lower().replace(' ', '_')
    path = os.path.join(save_dir, f'actual_vs_predicted_{safe_name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Saved: {path}")


def plot_residuals(y_true, y_pred, model_name, save_dir='plots'):
    """Residual distribution plot."""
    os.makedirs(save_dir, exist_ok=True)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} — Residual Analysis', fontsize=14, fontweight='bold')

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.4, s=15, color='#00d2ff', edgecolors='white', linewidth=0.3)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_xlabel('Predicted Price (Lakhs)')
    axes[0].set_ylabel('Residual (Lakhs)')
    axes[0].set_title('Residuals vs Predicted')

    # Residual distribution
    axes[1].hist(residuals, bins=40, color='#7b2ff7', edgecolor='white', alpha=0.8)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel('Residual (Lakhs)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')

    plt.tight_layout()
    safe_name = model_name.lower().replace(' ', '_')
    path = os.path.join(save_dir, f'residuals_{safe_name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Saved: {path}")


def plot_feature_importance(pipeline, model_name, save_dir='plots', top_n=15):
    """
    Bar chart of feature importance (tree-based models only).
    Extracts feature names from the ColumnTransformer.
    """
    os.makedirs(save_dir, exist_ok=True)
    model = pipeline.named_steps['model']

    if not hasattr(model, 'feature_importances_'):
        print(f"  ⚠️  {model_name} does not support feature_importances_")
        return

    preprocessor = pipeline.named_steps['preprocessor']
    num_features = list(preprocessor.transformers_[0][2])
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_features = list(cat_encoder.get_feature_names_out())
    all_names = num_features + cat_features

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette('viridis', n_colors=top_n)
    ax.barh(
        [all_names[i] for i in indices][::-1],
        importances[indices][::-1],
        color=colors,
    )
    ax.set_xlabel('Importance')
    ax.set_title(f'{model_name} — Top {top_n} Feature Importances', fontsize=14, fontweight='bold')

    plt.tight_layout()
    safe_name = model_name.lower().replace(' ', '_')
    path = os.path.join(save_dir, f'feature_importance_{safe_name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Saved: {path}")
