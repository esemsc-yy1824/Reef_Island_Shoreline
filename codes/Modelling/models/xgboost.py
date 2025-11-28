import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_model(X_train, y_train,
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=42):
    
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
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
