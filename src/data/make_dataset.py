import pandas as pd
from sklearn.datasets import fetch_openml

from src.utils.logger import logger


def load_data(dataset_name: str, columns_to_lower: bool = False) -> pd.DataFrame:
    """
    Args:
        dataset_name: OpenML dataset name (e.g. "disaster-tweets")
        columns_to_lower: lowercase all column names

    Returns:
        pd.DataFrame
    """
    raw = fetch_openml(
        name=dataset_name,
        as_frame=True,
        version="active",
        return_X_y=False,
        target_column=None,
    )

    data = raw["data"]

    if columns_to_lower:
        data.columns = data.columns.str.lower()

    _cast_dtypes(data)

    logger.info(f"Loaded '{dataset_name}' dataset")
    logger.info(f"Data shape: {data.shape}")

    return data


def _cast_dtypes(data: pd.DataFrame) -> None:
    """Cast columns to their intended types in-place."""
    if "target" in data.columns:
        data["target"] = data["target"].astype(bool)

    for col in ("text", "keyword", "location"):
        if col in data.columns:
            data[col] = data[col].astype(pd.StringDtype())
