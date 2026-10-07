"""
Model Training Module
======================
Defines 7 individual ML models + 1 Stacking Ensemble.
Uses sklearn Pipeline + ColumnTransformer for clean,
leak-proof preprocessing within each model.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.feature_engineering import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def create_preprocessor():
    """
    Build a ColumnTransformer that scales numericals
    and one-hot-encodes categoricals.
    """
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            (
                'cat',
                OneHotEncoder(
                    drop='first',
                    sparse_output=False,
                    handle_unknown='ignore',
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def get_model_configs():
    """Return a dict of model name → unfitted estimator."""
    return {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, verbosity=0,
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, verbose=-1,
        ),
    }


def compute_metrics(y_true, y_pred):
    """Compute regression metrics for a single set of predictions."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    return {
        'R2 Score': round(r2_score(y_true_arr, y_pred_arr), 4),
        'MAE': round(mean_absolute_error(y_true_arr, y_pred_arr), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)), 4),
        'MAPE': round(
            np.mean(np.abs((y_true_arr - y_pred_arr) / np.clip(y_true_arr, 0.01, None))) * 100, 2
        ),
    }


def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train all 7 individual models inside sklearn Pipelines.
    
    Returns
    -------
    results : dict
        {model_name: {'train': metrics, 'test': metrics}}
    pipelines : dict
        {model_name: fitted Pipeline}
    """
    models = get_model_configs()
    results = {}
    pipelines = {}

    print("\n" + "=" * 60)
    print("🤖  MODEL TRAINING")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n  🔄 Training {name} ...")
        pipeline = Pipeline([
            ('preprocessor', create_preprocessor()),
            ('model', model),
        ])
        pipeline.fit(X_train, y_train)

        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        results[name] = {
            'train': compute_metrics(y_train, y_pred_train),
            'test': compute_metrics(y_test, y_pred_test),
        }
        pipelines[name] = pipeline

        tr = results[name]['train']
        te = results[name]['test']
        print(f"     Train R²: {tr['R2 Score']:.4f}  |  Test R²: {te['R2 Score']:.4f}")
        print(f"     Train MAE: {tr['MAE']:.4f} |  Test MAE: {te['MAE']:.4f}")

    return results, pipelines


def train_stacking_ensemble(X_train, y_train, X_test, y_test):
    """
    Train a Stacking Ensemble using Random Forest, Gradient Boosting,
    and XGBoost as base learners with Ridge as the meta-learner.
    
    Returns
    -------
    result : dict
        {'train': metrics, 'test': metrics}
    stacking_model : StackingRegressor
        Fitted stacking model.
    """
    print("\n  🔄 Training Stacking Ensemble ...")

    estimators = [
        ('rf', Pipeline([
            ('preprocessor', create_preprocessor()),
            ('model', RandomForestRegressor(
                n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
            )),
        ])),
        ('gb', Pipeline([
            ('preprocessor', create_preprocessor()),
            ('model', GradientBoostingRegressor(
                n_estimators=100, max_depth=5, random_state=42
            )),
        ])),
        ('xgb', Pipeline([
            ('preprocessor', create_preprocessor()),
            ('model', XGBRegressor(
                n_estimators=100, max_depth=6, random_state=42, verbosity=0
            )),
        ])),
    ]

    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1,
    )

    stacking.fit(X_train, y_train)

    y_pred_train = stacking.predict(X_train)
    y_pred_test = stacking.predict(X_test)

    result = {
        'train': compute_metrics(y_train, y_pred_train),
        'test': compute_metrics(y_test, y_pred_test),
    }

    tr = result['train']
    te = result['test']
    print(f"     Train R²: {tr['R2 Score']:.4f}  |  Test R²: {te['R2 Score']:.4f}")
    print(f"     Train MAE: {tr['MAE']:.4f} |  Test MAE: {te['MAE']:.4f}")

    return result, stacking


def get_feature_names_from_pipeline(pipeline):
    """Extract feature names after ColumnTransformer transformation."""
    preprocessor = pipeline.named_steps['preprocessor']
    num_features = list(preprocessor.transformers_[0][2])
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_features = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return num_features + cat_features
