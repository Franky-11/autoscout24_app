from pathlib import Path

import pandas as pd

from autoscout24.config import CAR_DEALER_IMAGE_PATH, RAW_DATA_PATH
from autoscout24.data.cleaning import drop_missing_and_duplicates, prepare_modeling_dataset
from autoscout24.data.schema import validate_vehicle_schema


def load_raw_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_vehicle_schema(df)
    return df


def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return drop_missing_and_duplicates(load_raw_dataset(path))


def load_modeling_dataset(path: Path = RAW_DATA_PATH, factor: float = 1.5) -> pd.DataFrame:
    return prepare_modeling_dataset(load_raw_dataset(path), factor=factor)


def get_image_path() -> Path:
    return CAR_DEALER_IMAGE_PATH
