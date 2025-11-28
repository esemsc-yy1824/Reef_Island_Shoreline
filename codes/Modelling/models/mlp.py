import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_model(X_train, y_train,
                hidden_layer_sizes=(64, 64),
                activation='relu',
                alpha=0.0001,
                learning_rate_init=0.001,
                max_iter=1000,
                random_state=42):
    """
    Train an MLPRegressor.
    """
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate an MLPRegressor and print metrics.
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
    Save the trained MLP model to disk.
    """
    joblib.dump(model, path)
    print(f"Model saved to: {path}")

def load_model(path):
    """
    Load a saved MLP model from disk.
    """
    model = joblib.load(path)
    print(f"Model loaded from: {path}")
    return model
