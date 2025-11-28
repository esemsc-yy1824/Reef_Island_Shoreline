import os
import json
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.metrics import mean_squared_error, r2_score
from models import neural_net


def evaluate_on_csv(
    eval_path: str,
    model_path: str,
    pipeline_path: str = './Dataset/preprocessing_pipeline.pkl',
    scaler_path: str = './Dataset/target_scaler.pkl',
    feature_cols_path: str = './Dataset/feature_cols.pkl',
    drop_cols=('shoreline_pos','longitude','latitude'),
    save_pred_path: str | None = None,
):
    """
    Read evaluation CSV -> replicate pre-training preprocessing -> predict z -> inverse standardize to get Δd′
    Output metrics & optionally save predictions.
    Returns: metrics(dict), delta_dp_pred(np.ndarray)
    """
    df = pd.read_csv(eval_path).dropna(how='any')
    if 'shoreline_pos' not in df.columns:
        raise ValueError("Evaluation file is missing 'shoreline_pos' (should be Δd′).")
    y_true = df['shoreline_pos'].astype(float).to_numpy()

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    if 'date' in X.columns:
        X['date'] = pd.to_datetime(X['date'], errors='coerce')
        if 'month' not in X.columns:
            X['month'] = X['date'].dt.month
        date_ns = X['date'].view('int64')
        X['date_ordinal'] = (date_ns / 86_400_000_000_000.0).astype('float64')
        X.loc[X['date'].isna(), 'date_ordinal'] = np.nan
        X = X.drop(columns=['date'])

    pipeline = joblib.load(pipeline_path)
    X_t = pipeline.transform(X)

    cols_ref = None
    if os.path.exists(feature_cols_path):
        try:
            cols_ref = joblib.load(feature_cols_path)
        except Exception:
            cols_ref = None

    model = joblib.load(model_path)
    if cols_ref is None and hasattr(model, 'feature_names_in_'):
        cols_ref = list(model.feature_names_in_)

    if cols_ref is not None and hasattr(X_t, 'columns'):
        extra = [c for c in X_t.columns if c not in cols_ref]
        if extra:
            X_t = X_t.drop(columns=extra)
        missing = [c for c in cols_ref if c not in X_t.columns]
        if missing:
            raise ValueError(f"Prediction is missing feature columns from training: {missing}")
        X_t = X_t[cols_ref]

    z_pred = model.predict(X_t)
    y_scaler = joblib.load(scaler_path)
    delta_dp_pred = y_scaler.inverse_transform(np.asarray(z_pred).reshape(-1,1)).ravel()

    mse = mean_squared_error(y_true, delta_dp_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, delta_dp_pred)
    metrics = {"r2": float(r2), "mse": float(mse), "rmse": rmse, "n": int(len(y_true))}

    if save_pred_path:
        out = pd.DataFrame({
            "delta_dp_true": y_true,
            "z_pred": z_pred,
            "delta_dp_pred": delta_dp_pred,
        })
        if 'date' in df.columns:
            out.insert(0, "date", pd.to_datetime(df['date'], errors='coerce').astype(str))
        out.to_csv(save_pred_path, index=False, encoding="utf-8")

    return metrics, delta_dp_pred

def evaluate_neural_net_model(eval_csv_path, model_type, model_save_dir='./outputs', 
                            pipeline_path='./Dataset/preprocessing_pipeline.pkl',
                            feature_cols_path='./Dataset/feature_cols.pkl',
                            params_json_path='./outputs/nn_tuned_best_params.json'):
    """
    Evaluate a neural network model on evaluation data from a CSV file.
    
    Returns:
    dict
        Dictionary containing evaluation metrics (R², MSE, RMSE, n)
    numpy.ndarray
        Array of predicted values
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Read data
        pre_data = pd.read_csv(eval_csv_path).dropna()
        
        # Separate features and target
        X_pre = pre_data.drop(columns=['shoreline_pos'])
        y_true = pre_data['shoreline_pos']
        
        # Apply preprocessing pipeline
        pipeline = joblib.load(pipeline_path)
        X_pre_transformed = pipeline.transform(X_pre)
        
        # Handle feature dimension consistency
        if os.path.exists(feature_cols_path):
            try:
                feature_cols = joblib.load(feature_cols_path)
                print(f"Loaded feature columns: {len(feature_cols)}")
                
                # Ensure feature consistency
                if hasattr(X_pre_transformed, 'columns'):
                    # Remove extra features
                    extra_cols = [c for c in X_pre_transformed.columns if c not in feature_cols]
                    if extra_cols:
                        X_pre_transformed = X_pre_transformed.drop(columns=extra_cols)
                    
                    # Check for missing features
                    missing_cols = [c for c in feature_cols if c not in X_pre_transformed.columns]
                    if missing_cols:
                        print(f"Warning: Missing features: {missing_cols}")
                    
                    # Ensure consistent feature order
                    X_pre_transformed = X_pre_transformed[feature_cols]
            except Exception as e:
                print(f"Error loading feature columns: {e}")
        
        # Convert to model-compatible format
        if hasattr(X_pre_transformed, 'toarray'):
            X_array = X_pre_transformed.toarray().astype(np.float32)
        elif isinstance(X_pre_transformed, pd.DataFrame):
            X_array = X_pre_transformed.values.astype(np.float32)
        else:
            X_array = X_pre_transformed.astype(np.float32)
        
        # Get input dimension
        input_dim = X_array.shape[1]
        print(f"Input dimension: {input_dim}")
        
        # Load model parameters
        with open(params_json_path, 'r') as f:
            best_params = json.load(f)
        
        # Set hidden layer dimension
        if model_type == 'neural_net_model':
            hidden_dim = 64
        else:
            hidden_dim = best_params['hidden_dim']
        
        print(f"Hidden layer dimension: {hidden_dim}")
        
        # Load model
        model_path = os.path.join(model_save_dir, f"{model_type}.pth")
        model = neural_net.load_model(model_path, 
                                     input_dim=input_dim, 
                                     hidden_dim=hidden_dim, 
                                     device=device)
        
        # Make predictions
        with torch.no_grad():
            X_tensor = torch.tensor(X_array).to(device)
            y_pred = model(X_tensor).cpu().numpy().squeeze()
        
        # Calculate evaluation metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        n_samples = len(y_true)
        
        # Print results
        print(f"\n📊 Model Evaluation on Target Island:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  Mean Squared Error (MSE): {mse:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"  Number of samples: {n_samples}")
        
        # Return results
        metrics = {
            "r2": r2,
            "mse": mse,
            "rmse": rmse,
            "n": n_samples
        }
        
        return metrics, y_pred
        
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        raise