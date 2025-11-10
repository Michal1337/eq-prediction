from typing import Dict, Union, List
import pandas as pd

SEED: int = 1337

# Parameters for initial data download
MIN_TIME: str = "100-01-01"
START_TIME: str = "1940-01-01"
END_TIME: str = "2023-10-01"
MIN_LAT: int = -90
MAX_LAT: int = 90
MIN_LON: int = -180
MAX_LON: int = 180

# Parameters for initial preprocessing and adding features
TIME_CUT: str = "1973-01-01"
MAG_TH: int = 5
GEO_SPLIT: int = 1
TIME_DIFF: pd.DateOffset = pd.DateOffset(months=1)

# Parameters for dataset splitting and creation
SPLIT_DATE_TRAIN: str = "2020-01-01"
SPLIT_DATE_VAL: str = "2023-01-01"
RADIUS: int = 300
THRESHOLD: int = 150
BLOCK_SIZE: int = 64

# Parameters for data preprocessing
PREPROC_PARAMS = {
    "mag_low": -1,
    "mag_high": 7,
    "depth_low": 2,
    "depth_high": 1e8,
    "dist_low": 1,
    "dist_high": 1e8,
    "dist_region_low": 2,
    "dist_region_high": 1e8,
    "scale_distance": 78.28,
    "scale_distance_lag": 300,
    "bins": [0, 1, 2, 3, 4, 5, 6, 7, 10, 14, 21, 30, 60, 180, 1e8],
}

# Features
FEATURES_REGION: List[str] = ["lat_cent", "lon_cent", "dist_region", "plate_region"]
FEATURES: List[str] = [
    "mag",
    "depth",
    "latitude_new",
    "longitude_new",
    "dist",
    "distance",
    "plate",
    "diff_days",
    "magType",
]

# Parameters for datasets merging
INPUT_DIRECTORY: str = "../data/npys/"
OUTPUT_DIRECTORY: str = "../data/datasets/"

# Model parameters
D_MODEL: int = 126
NUM_LAYERS: int = 2
NUM_HEADS: int = 2

# Training parameters
BATCH_SIZE: int = 1024
EPOCHS: int = 30
