import joblib
import numpy as np
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import lightgbm as lgb

def train_model(X_train, y_train,
                ridge_alpha=1.0,
                lgb_params=None,
                xgb_params=None,
                final_estimator_params=None,
                random_state=42):
    """
    Train a Stacking Regressor with LightGBM, XGBoost, and Ridge.
    """

    # Default LightGBM parameters
    if lgb_params is None:
        lgb_params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state
        }

    # Default XGBoost parameters
    if xgb_params is None:
        xgb_params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state
        }

    # Default Ridge as final estimator
    if final_estimator_params is None:
        final_estimator_params = {
            'alpha': ridge_alpha,
            'random_state': random_state
        }

    # Base estimators
    estimators = [
        ('lgb', lgb.LGBMRegressor(**lgb_params)),
        ('xgb', XGBRegressor(**xgb_params))
    ]

    final_estimator = Ridge(**final_estimator_params)

    model = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        n_jobs=-1,
        passthrough=False
    )

    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the Stacking model and print metrics.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Test MSE: {mse:.3f}")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test R²: {r2:.3f}")

    return mse, rmse, r2

def save_model(model, path):
    """
    Save the stacking model to disk.
    """
    joblib.dump(model, path)
    print(f"Model saved to: {path}")

def load_model(path):
    """
    Load the saved stacking model.
    """
    model = joblib.load(path)
    print(f"Model loaded from: {path}")
    return model