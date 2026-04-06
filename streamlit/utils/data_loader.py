from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "test.csv"
TARGET_CANDIDATES = ["Irrigation_Need"]


def get_data_paths() -> tuple[Path, Path]:
    return TRAIN_DATA_PATH, TEST_DATA_PATH


@st.cache_data(show_spinner=False)
def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def detect_target_column(df: pd.DataFrame) -> str | None:
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def summarize_dataset(df: pd.DataFrame) -> dict:
    target_column = detect_target_column(df)
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    feature_columns = [
        column for column in df.columns if target_column is None or column != target_column
    ]

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "feature_count": len(feature_columns),
        "target_column": target_column,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_values": int(df.isna().sum().sum()),
    }
