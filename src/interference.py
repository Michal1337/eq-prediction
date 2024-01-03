import datetime as dt
import pickle
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

warnings.filterwarnings("ignore")

from typing import Dict, List, Tuple, Union

from sklearn.preprocessing import MinMaxScaler

from add_features import add_region_info, add_tectonic_info, haversine_distance
from get_data import get_earthquake_count, get_earthquake_data, make_df, make_params
from make_npys import make_feature_order
from model import MyModel
from params import (
    BLOCK_SIZE,
    FEATURES,
    FEATURES_REGION,
    GEO_SPLIT,
    PREPROC_PARAMS,
    RADIUS,
)

def check_cords(X: float, Y: float) -> bool:
    if X < 0 or X > 360 or Y < -90 or Y > 90:
        return False
    return True

def get_data(
    X: float,
    Y: float,
    START_TIME: str,
    END_TIME: str,
    MIN_LAT: float,
    MAX_LAT: float,
    MIN_LON: float,
    MAX_LON: float,
) -> Tuple[pd.DataFrame, str, List[str]]:
    while True:
        count = get_earthquake_count(
            make_params(START_TIME, END_TIME, MIN_LAT, MAX_LAT, MIN_LON, MAX_LON)
        )
        if count < 65:
            START_TIME = START_TIME - dt.timedelta(days=300)
        else:
            resp = get_earthquake_data(
                make_params(START_TIME, END_TIME, MIN_LAT, MAX_LAT, MIN_LON, MAX_LON)
            )
            df, errors = make_df(
                resp,
                make_params(START_TIME, END_TIME, MIN_LAT, MAX_LAT, MIN_LON, MAX_LON),
                [],
            )
            df = df[df["type"] == "earthquake"]
            df = df[["time", "longitude", "latitude", "depth", "mag", "magType"]]
            df["time"] = df["time"].apply(lambda x: dt.datetime.fromtimestamp(x / 1000))
            df.dropna(inplace=True)
            df.drop_duplicates(inplace=True)
            df["pos"] = "0_0"
            df.loc[
                (df["latitude"] >= Y - GEO_SPLIT / 2)
                & (df["latitude"] <= Y + GEO_SPLIT / 2)
                & (df["longitude"] >= X - GEO_SPLIT / 2)
                & (df["longitude"] <= X + GEO_SPLIT / 2),
                "pos",
            ] = (
                str(Y - GEO_SPLIT / 2) + "_" + str(X - GEO_SPLIT / 2)
            )
            df["distance"] = haversine_distance(df["latitude"], df["longitude"], Y, X)
            df["latitude_disc"] = Y - GEO_SPLIT / 2
            df["longitude_disc"] = X - GEO_SPLIT / 2
            tmp1 = df[df["pos"] != "0_0"]
            if len(tmp1) > 0:
                tmp2 = df[df["distance"] <= 300]
                tmp2 = tmp2[tmp2["time"] <= tmp1["time"].max()]
                if len(tmp2) >= BLOCK_SIZE + 1:
                    return df[df["time"] <= tmp1["time"].max()], START_TIME, errors
            START_TIME = START_TIME - dt.timedelta(days=300)


def map_col(df: pd.DataFrame, col: str, mapping: Dict[str, int]) -> pd.DataFrame:
    mapping = dict(zip(mapping.iloc[:, 0], mapping.iloc[:, 1]))
    df[col] = df[col].map(mapping)
    return df


def preprocess_df(
    df: pd.DataFrame,
    preproc_params: Dict[str, Union[int, float, List[int]]],
    scaler_dict: Dict[str, MinMaxScaler],
) -> pd.DataFrame:
    scaler = scaler_dict["mag"]
    df["mag"] = scaler.transform(
        np.clip(
            df["mag"].values, preproc_params["mag_low"], preproc_params["mag_high"]
        ).reshape(-1, 1)
    )

    scaler = scaler_dict["depth"]
    df["depth"] = np.log(df["depth"] + np.abs(df["depth"].min()) + 1)
    df["depth"] = scaler.transform(
        np.clip(
            df["depth"].values,
            preproc_params["depth_low"],
            preproc_params["depth_high"],
        ).reshape(-1, 1)
    )

    scaler = scaler_dict["latitude_new"]
    df["latitude_new"] = scaler.transform(df["latitude"].values.reshape(-1, 1))

    scaler = scaler_dict["longitude_new"]
    df["longitude_new"] = scaler.transform(df["longitude"].values.reshape(-1, 1))

    scaler = scaler_dict["lat_cent"]
    df["lat_cent"] = scaler.transform(df["lat_cent"].values.reshape(-1, 1))

    scaler = scaler_dict["lon_cent"]
    df["lon_cent"] = scaler.transform(df["lon_cent"].values.reshape(-1, 1))

    scaler = scaler_dict["dist"]
    df["dist"] = df["dist"].astype(float)
    df["dist"] = scaler.transform(
        np.clip(
            np.log(df["dist"] + 1).values.reshape(-1, 1),
            preproc_params["dist_low"],
            preproc_params["dist_high"],
        )
    )

    scaler = scaler_dict["dist_region"]
    df["dist_region"] = scaler.transform(
        np.clip(
            np.log(df["dist_region"] + 1).values.reshape(-1, 1),
            preproc_params["dist_region_low"],
            preproc_params["dist_region_high"],
        )
    )

    return df


def make_block(
    df: pd.DataFrame,
    pos: str,
    radius: int,
    block_size: int,
    preproc_params: Dict[str, Union[int, float, List[int]]],
) -> pd.DataFrame:
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 10, 14, 21, 30, 60, 180, 1e8]
    lat, lon = pos.split("_")
    lat, lon = float(lat), float(lon)
    tmp1 = df[df["pos"] == pos]
    diff = int(radius / 111) + 3
    tmp2 = df[
        (
            (df["latitude"] >= lat - diff)
            & (df["latitude"] <= lat + diff)
            & (df["longitude"] >= lon - diff)
            & (df["longitude"] <= lon + diff)
        )
        & (df["pos"] != pos)
    ]
    tmp1["label"] = 0
    tmp2["label"] = -1
    tmp = pd.concat([tmp1, tmp2], axis=0)
    tmp = tmp[tmp["distance"] <= radius]
    tmp.sort_values(by=["time"], inplace=True)
    tmp["diff_days"] = (tmp["time"] - tmp["time"].shift(1)).dt.days
    tmp.dropna(inplace=True)
    tmp["diff_days"] = np.digitize(tmp["diff_days"], bins=bins) - 1
    for idx in range(1, block_size):
        tmp["mag_" + str(idx)] = tmp["mag"].shift(idx)
        tmp["depth_" + str(idx)] = tmp["depth"].shift(idx)
        tmp["latitude_new_" + str(idx)] = tmp["latitude_new"].shift(idx)
        tmp["longitude_new_" + str(idx)] = tmp["longitude_new"].shift(idx)
        tmp["dist_" + str(idx)] = tmp["dist"].shift(idx)
        tmp["distance_" + str(idx)] = (
            tmp["distance"].shift(idx) / preproc_params["scale_distance_lag"]
        )
        tmp["plate_" + str(idx)] = tmp["plate"].shift(idx)
        tmp["diff_days_" + str(idx)] = tmp["diff_days"].shift(idx)
        tmp["magType_" + str(idx)] = tmp["magType"].shift(idx)
    tmp = tmp[tmp["label"] != -1]
    tmp["distance"] = tmp["distance"] / preproc_params["scale_distance"]
    tmp.dropna(inplace=True)
    return tmp


def reshape(df, block_size, feature_order, featrues_region):
    x_ts = (
        df[feature_order]
        .to_numpy()
        .reshape(-1, block_size, len(feature_order) // block_size)
    )
    x_region = df[featrues_region].to_numpy().reshape(-1, len(featrues_region))
    return x_ts, x_region


def make_timeseries(
    df: pd.DataFrame,
    x: float,
    y: float,
    radius: int,
    block_size: int,
    features_order: List[str],
    features_region: List[str],
    preproc_params: Dict[str, Union[int, float, List[int]]],
    scaler_dict: Dict[str, MinMaxScaler],
) -> Tuple[np.ndarray, np.ndarray]:
    df["time"] = pd.to_datetime(df["time"], format="mixed")
    df.sort_values(by="time", inplace=True)
    pos = (
        str(y - GEO_SPLIT / 2) + "_" + str(x - GEO_SPLIT / 2)
    )
    df = preprocess_df(df, preproc_params, scaler_dict)
    df_pos = make_block(df, pos, radius, block_size, preproc_params)
    df_pos = df_pos.iloc[-1]
    x_ts, x_region = reshape(df_pos, block_size, features_order, features_region)
    return x_ts, x_region


def prepare_data(df: pd.DataFrame, geo_split: int) -> pd.DataFrame:
    df_tp = pd.read_csv("../data/all.csv")
    df_tp.drop_duplicates(inplace=True)

    df = add_region_info(df, df_tp, geo_split)
    df = add_tectonic_info(df, df_tp)

    mapping1 = pd.read_csv("../data/magtype2id.csv")
    mapping2 = pd.read_csv("../data/plate2id.csv")
    mapping3 = pd.read_csv("../data/plate_region2id.csv")

    df = map_col(df, "magType", mapping1)
    df = map_col(df, "plate", mapping2)
    df = map_col(df, "plate_region", mapping3)
    return df


def make_prediction(X: float, Y: float) -> float:
    if not check_cords(X, Y):
        return -1
    now = dt.datetime.now()
    START_TIME = now - dt.timedelta(days=300)
    END_TIME = now
    X_mid = X // GEO_SPLIT * GEO_SPLIT + GEO_SPLIT / 2
    Y_mid = Y // GEO_SPLIT * GEO_SPLIT + GEO_SPLIT / 2
    MIN_LAT = Y_mid - 5
    MAX_LAT = Y_mid + 5
    MIN_LON = X_mid - 5
    MAX_LON = X_mid + 5

    df, START_TIME, errors = get_data(
        X, Y, START_TIME, END_TIME, MIN_LAT, MAX_LAT, MIN_LON, MAX_LON
    )
    df = prepare_data(df, GEO_SPLIT)

    scalers = pickle.load(open("../data/scalers_for_npys.pkl", "rb"))
    features_order = make_feature_order(FEATURES, BLOCK_SIZE)
    x_ts, x_region = make_timeseries(
        df,
        X,
        Y,
        RADIUS,
        BLOCK_SIZE,
        features_order,
        FEATURES_REGION,
        PREPROC_PARAMS,
        scalers,
    )

    model = tf.keras.models.load_model("../models/model.keras", custom_objects={"MyModel": MyModel})

    return model.predict(x_ts).numpy()[0][0]


if __name__ == "__main__":
    X = 25.814000
    Y = 35.373000
    make_prediction(X, Y)
