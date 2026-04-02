from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = ROOT_DIR / "data" / "processed"


class DataFileNotFoundError(FileNotFoundError):
    pass


def _validate_file(path: Path) -> None:
    if not path.exists():
        raise DataFileNotFoundError(f"Expected file not found: {path}")


def load_split(split_name: str, processed_dir: Path | str = DEFAULT_PROCESSED_DIR) -> pd.DataFrame:
    processed_dir = Path(processed_dir)
    path = processed_dir / f"{split_name}.csv"
    _validate_file(path)
    return pd.read_csv(path)


def load_processed_splits(
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        load_split("train", processed_dir),
        load_split("val", processed_dir),
        load_split("test", processed_dir),
    )


def split_features_target(df: pd.DataFrame, target_col: str = "label"):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y