import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_model(X_train, y_train, n_neighbors=5):
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Calculating RMSE
    r2 = r2_score(y_test, y_pred)

    print(f"Test MSE: {mse:.3f}")
    print(f"Test RMSE: {rmse:.3f}")  # Printing RMSE
    print(f"Test R²: {r2:.3f}")
    return mse, rmse, r2

def save_model(model, path):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path):
    return joblib.load(path)
