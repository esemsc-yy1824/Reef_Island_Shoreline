import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from preprocessing.data_preprocessing import build_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import json
from datetime import datetime

def split_and_save(
    data_path='./data/final_data.csv',
    output_path='./Dataset/',
    test_size=0.2,
    random_state=42,
    sitename='Madhirivaadhoo'
):
    df = pd.read_csv(data_path)
    print(f"Original data shape: {df.shape}")

    # Drop missing values
    df = df.dropna()
    print(f"Shape after dropping missing values: {df.shape}")

    # Target variable
    y = df['shoreline_pos']

    # Drop target column from features
    X = df.drop(columns=['shoreline_pos', 'longitude', 'latitude'])
    X['date'] = pd.to_datetime(X['date'], errors='coerce')
    d = X['date']
    date_ordinal = d.view('int64') / 86_400_000_000_000  # 86400 * 1e9
    date_ordinal = date_ordinal.astype('float64')
    date_ordinal[d.isna()] = np.nan
    X['date_ordinal'] = date_ordinal

    X = X.drop(columns=['date'])

    # Split dataset
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=test_size, random_state=random_state
    )

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

        # 只在训练集拟合特征流水线
    pipeline = build_pipeline()
    X_train_t = pipeline.fit_transform(X_train)
    X_test_t  = pipeline.transform(X_test)

    # —— 只在训练集拟合“目标 y=Δd′ 的标准化器”（μ、σ）
    y_scaler = StandardScaler(with_mean=True, with_std=True)
    y_train = y_scaler.fit_transform(y_train_raw.to_numpy().reshape(-1, 1)).ravel()
    y_test  = y_scaler.transform(y_test_raw.to_numpy().reshape(-1, 1)).ravel()

    mu = float(y_scaler.mean_[0])
    sigma = float(y_scaler.scale_[0])
    print(f"Target standardization (train-only): mu={mu:.6f}, sigma={sigma:.6f}")

    # —— 保存
    # os.makedirs(output_path, exist_ok=True)
    joblib.dump(pipeline, os.path.join(output_path, 'preprocessing_pipeline.pkl'))
    # 若 X_train_t 是 DataFrame 则保存列名
    try:
        feature_cols = X_train_t.columns.tolist()
        joblib.dump(feature_cols, os.path.join(output_path, 'feature_cols.pkl'))
    except Exception:
        pass

    joblib.dump(X_train_t, os.path.join(output_path, 'X_train.pkl'))
    joblib.dump(X_test_t,  os.path.join(output_path, 'X_test.pkl'))
    joblib.dump(y_train,   os.path.join(output_path, 'y_train.pkl'))   # 标准化后的 Δd′ (z)
    joblib.dump(y_test,    os.path.join(output_path, 'y_test.pkl'))    # 标准化后的 Δd′ (z)

    # 便于诊断：保存未标准化的 Δd′
    joblib.dump(y_train_raw.to_numpy(), os.path.join(output_path, 'y_train_raw_delta_dp.npy'))
    joblib.dump(y_test_raw.to_numpy(),  os.path.join(output_path, 'y_test_raw_delta_dp.npy'))


    # 保存 y 的标准化器（推理/逆变换要用）
    joblib.dump(y_scaler, os.path.join(output_path, 'target_scaler.pkl'))

    print("All data has been saved!")

    
    meta_path = os.path.join(f"../CoastSat/data/{sitename}/{sitename}_reconstruction_metadata.json")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta.setdefault("standardization", {})
    meta["standardization"].update({
        "mu":   float(y_scaler.mean_[0]),
        "sigma":float(y_scaler.scale_[0]),
        "definition": "z = (Δd' - μ) / σ",
        "fitted_on":  "train_only",  # 说明只用训练集拟合
    })
    meta["updated_at"] = datetime.utcnow().isoformat() + "Z"

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Updated μ/σ in: {meta_path}")

if __name__ == "__main__":
    split_and_save()
