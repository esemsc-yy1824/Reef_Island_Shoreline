import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

def permutation_importance_nn(model, X_test, y_test, metric_fn, n_repeats=10):
    if isinstance(X_test, pd.DataFrame):
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
    elif isinstance(X_test, np.ndarray):
        X_test = torch.tensor(X_test, dtype=torch.float32)

    if isinstance(y_test, pd.DataFrame):
        y_test = torch.tensor(y_test.values, dtype=torch.float32)
    elif isinstance(y_test, np.ndarray):
        y_test = torch.tensor(y_test, dtype=torch.float32)

    baseline_score = metric_fn(model, X_test, y_test)
    importances = np.zeros(X_test.shape[1])

    for i in range(X_test.shape[1]):
        feature_values = X_test[:, i].clone()
        scores = []
        for _ in range(n_repeats):
            X_test_perm = X_test.clone()
            X_test_perm[:, i] = feature_values[torch.randperm(X_test.size(0))]
            score = metric_fn(model, X_test_perm, y_test)
            scores.append(score)
        
        importances[i] = baseline_score - np.mean(scores)

    return importances


def r2_score_fn(model, X_test, y_test):
    # Convert X_test and y_test to tensors if needed
    if isinstance(X_test, pd.DataFrame):
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
    elif isinstance(X_test, np.ndarray):
        X_test = torch.tensor(X_test, dtype=torch.float32)

    if isinstance(y_test, pd.DataFrame):
        y_test = torch.tensor(y_test.values, dtype=torch.float32)
    elif isinstance(y_test, np.ndarray):
        y_test = torch.tensor(y_test, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        # Forward pass
        y_pred = model(X_test)

        # Ensure tensors are moved to CPU and converted to numpy
        y_test_np = y_test.detach().cpu().numpy() if torch.is_tensor(y_test) else y_test
        y_pred_np = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred

        # Compute and return R² score
        return r2_score(y_test_np, y_pred_np)
    
def calculate_avg_feature_importance(feature_importance_dfs):
    """
    Calculate average feature importance after normalizing all individual importance scores to [0, 1] range
    """
    
    normalized_dfs = []
    for df in feature_importance_dfs:
        df_norm = df.copy()
        imp = df_norm['Importance']
        min_val, max_val = imp.min(), imp.max()
        df_norm['Importance'] = (imp - min_val) / (max_val - min_val) if max_val != min_val else 0.5
        normalized_dfs.append(df_norm)
    
    df_all = pd.concat(normalized_dfs)
    avg_importance = (df_all.groupby('Feature')['Importance']
                         .mean()
                         .sort_values(ascending=False)
                         .reset_index())
    
    return avg_importance
