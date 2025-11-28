import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_model(X_train, y_train, n_estimators=200, max_depth=None, random_state=42):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Test MSE: {mse:.3f}")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test R²: {r2:.3f}")
    return mse, rmse, r2

def save_model(model, path):
    joblib.dump(model, path)
    print(f"Model saved to: {path}")

def load_model(path):
    model = joblib.load(path)
    print(f"Model loaded from: {path}")
    return model
