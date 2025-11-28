import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

def train_model(X_train, y_train, alpha=1.0, random_state=42):
    """
    Train a Ridge Regression model.
    """
    model = Ridge(alpha=alpha, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a Ridge Regression model and print metrics.
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
    Save a trained Ridge model to disk.
    """
    joblib.dump(model, path)
    print(f"Model saved to: {path}")

def load_model(path):
    """
    Load a saved Ridge model from disk.
    """
    model = joblib.load(path)
    print(f"Model loaded from: {path}")
    return model
