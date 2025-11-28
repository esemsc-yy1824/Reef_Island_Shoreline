import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit
from preprocessing.data_preprocessing import build_pipeline

def split_and_save(
    data_path='./data/final_data.csv',
    output_path='./Dataset/',
    test_size=0.2,
    random_state=42
):
    df = pd.read_csv(data_path)
    print(f"Original data shape: {df.shape}")

    # Drop missing values
    df = df.dropna()
    print(f"Shape after dropping missing values: {df.shape}")

    # Ensure 'date' column exists
    if 'date' not in df.columns:
        raise ValueError("Your dataframe must have a 'date' column for time-based splitting.")

    # Target variable
    y = df['shoreline_pos']

    # Features (drop target)
    X = df.drop(columns=['shoreline_pos'])

    # Group info for GroupShuffleSplit
    groups = df['date']

    # Build and apply pipeline
    pipeline = build_pipeline()
    X_transformed = pipeline.fit_transform(X)

    # Save pipeline
    joblib.dump(pipeline, output_path + 'preprocessing_pipeline.pkl')

    # Save feature column names
    if hasattr(X_transformed, 'columns'):
        feature_cols = X_transformed.columns.tolist()
    else:
        feature_cols = [f"feature_{i}" for i in range(X_transformed.shape[1])]
    joblib.dump(feature_cols, output_path + 'feature_cols.pkl')

    # ------------------ Group-aware split ------------------
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, y, groups))  # <--- NOTE: Use df here for correct indexing

    # Reset index to make sure row positions align with idx
    df = df.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if isinstance(X_transformed, pd.DataFrame):
        X_transformed = X_transformed.reset_index(drop=True)
        X_array = X_transformed.to_numpy()
    else:
        X_array = X_transformed

    # Final split
    X_train = X_array[train_idx]
    X_test = X_array[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # ------------------ Save ------------------
    joblib.dump(X_train, output_path + 'X_train.pkl')
    joblib.dump(X_test, output_path + 'X_test.pkl')
    joblib.dump(y_train, output_path + 'y_train.pkl')
    joblib.dump(y_test, output_path + 'y_test.pkl')

    print("All data has been saved!")

if __name__ == "__main__":
    split_and_save()
