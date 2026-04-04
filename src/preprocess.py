import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Get base directory of project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data(path):
    df = pd.read_csv(path, sep=" ", header=None)
    df = df.dropna(axis=1)

    df.columns = (
        ['engine_id', 'cycle'] +
        [f'op_setting_{i}' for i in range(1, 4)] +
        [f'sensor_{i}' for i in range(1, 22)]
    )

    return df


def add_rul_and_label(df, threshold=30):
    max_cycle = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycle.columns = ['engine_id', 'max_cycle']

    df = df.merge(max_cycle, on='engine_id')

    df['RUL'] = df['max_cycle'] - df['cycle']
    df['label'] = (df['RUL'] <= threshold).astype(int)

    return df


def split_by_engine(df):
    engine_ids = df['engine_id'].unique()

    np.random.seed(42)
    np.random.shuffle(engine_ids)

    train_ids = engine_ids[:70]
    val_ids = engine_ids[70:85]
    test_ids = engine_ids[85:]

    train_df = df[df['engine_id'].isin(train_ids)].copy()
    val_df = df[df['engine_id'].isin(val_ids)].copy()
    test_df = df[df['engine_id'].isin(test_ids)].copy()

    return train_df, val_df, test_df


def get_useful_columns(train_df):
    variance = train_df.var(numeric_only=True)
    useful_cols = variance[variance > 1e-5].index.tolist()

    required_cols = ['engine_id', 'cycle', 'max_cycle', 'RUL', 'label']
    for col in required_cols:
        if col in train_df.columns and col not in useful_cols:
            useful_cols.append(col)

    return useful_cols


def clean_dataset(df, useful_cols):
    df = df[useful_cols].copy()
    df = df.drop(columns=['engine_id', 'cycle', 'max_cycle', 'RUL'], errors='ignore')
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def preprocess_pipeline(path):
    df = load_data(path)
    df = add_rul_and_label(df)

    train_df, val_df, test_df = split_by_engine(df)

    useful_cols = get_useful_columns(train_df)

    train_df = clean_dataset(train_df, useful_cols)
    val_df = clean_dataset(val_df, useful_cols)
    test_df = clean_dataset(test_df, useful_cols)

    return train_df, val_df, test_df


def save_processed_data(train_df, val_df, test_df):
    output_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train data saved to: {train_path}")
    print(f"Validation data saved to: {val_path}")
    print(f"Test data saved to: {test_path}")


if __name__ == "__main__":
    raw_data_path = os.path.join(BASE_DIR, "data", "raw", "train_FD001.txt")

    train_df, val_df, test_df = preprocess_pipeline(raw_data_path)

    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    print("\nTrain label distribution:")
    print(train_df['label'].value_counts())

    save_processed_data(train_df, val_df, test_df)