"""
Script used for inference. The script takes in a latitude and longitude and returns the probability of an earthquake occurring in the next 30 days at that location.
The script also returns the date of the last earthquake that occurred at that location. If the last earthquake occurred more than 30 days ago, the script returns None.
If the coordinates are not in the expected regions the script returns None.
"""
import datetime as dt
import pickle
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

warnings.filterwarnings("ignore")

from typing import Dict, List, Tuple, Union

from sklearn.preprocessing import MinMaxScaler

from .add_features import add_region_info, add_tectonic_info, haversine_distance
from .get_data import get_earthquake_count, get_earthquake_data, make_df, make_params
from .make_npys import make_feature_order
from .model import MyModel
from .params import (
    BLOCK_SIZE,
    FEATURES,
    FEATURES_REGION,
    GEO_SPLIT,
    PREPROC_PARAMS,
    RADIUS,
)


def make_params_circle(
    starttime: str, endtime: str, latitude: float, longitude: float, maxradiuskm: float
) -> Dict[str, Union[str, float]]:
    """
    Construct a dictionary of parameters for USGS API request. https://earthquake.usgs.gov/fdsnws/event/1/#parameters.

    Parameters:
    - starttime (str): Start time for search
    - endtime (str): End time for search
    - latitude (float): Latitude for search
    - longitude (float): Longitude for search
    - maxradiuskm (float): Maximum radius for search

    Returns:
    Dict[str, Union[str, int]]: A dictionary containing the parameters for the geojson request.

    Example:
    >>> make_params("2023-01-01T00:00:00", "2023-12-31T23:59:59", 35.0, 45.0, "-120.0", -110.0)
    {'format': 'geojson',
        'starttime': '2023-01-01T00:00:00',
        'endtime': '2023-12-31T23:59:59',
        'latitude': 35.0,
        'longitude': 45.0,
        'maxradiuskm': '-120.0',
    }
    """
    params = {
        "format": "geojson",
        "starttime": starttime,
        "endtime": endtime,
        "latitude": latitude,
        "longitude": longitude,
        "maxradiuskm": maxradiuskm,
    }
    return params


def get_data(
    X: float,
    Y: float,
    start_time: dt.datetime,
    end_time: dt.datetime,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    geo_split: int,
    radius: int,
    block_size: int,
) -> Tuple[pd.DataFrame, dt.datetime, List[str]]:
    """
    Download the necessary data to make a prediction. If the last earthquake occurred more than 30 days ago, the function returns None.
    If it takes too long to download the data, the function returns None. Download the data in 3 steps:
    1. Check if there is an earthquake in the region in the last 30 days. If there is no earthquake, return None.
    2. Check how far back we need to go to get enough data to make a prediction. If it takes too long to download the data, return None.
    3. Download the data based on the found start and end time.

    These steps ensure that the function makes minimal number of requests to the USGS API.

    Parameters:
    - X (float): Longitude of the center of the region.
    - Y (float): Latitude of the center of the region.
    - start_time (datetime): Start time for search.
    - end_time (datetime): End time for search.
    - min_lat (float): Minimum latitude for search.
    - max_lat (float): Maximum latitude for search.
    - min_lon (float): Minimum longitude for search.
    - max_lon (float): Maximum longitude for search.
    - geo_split (int): Size of the region.
    - radius (int): Radius of the region.
    - block_size (int): Size of the block.

    Returns:
    Tuple[pd.DataFrame, dt.datetime, List[str]]: A tuple containing the following elements:
        - pd.DataFrame: A dataframe containing the data.
        - dt.datetime: The end time of the data.
        - List[str]: A list of errors that occurred during the data download.
    """
    errors = []
    count = get_earthquake_count(
        make_params(start_time, end_time, min_lat, max_lat, min_lon, max_lon)
    )
    if count == 0:
        print("Last earthquake was more than 30 days ago")
        return None, None, None
    resp = get_earthquake_data(
        make_params(start_time, end_time, min_lat, max_lat, min_lon, max_lon)
    )
    df, errors = make_df(
        resp,
        make_params(start_time, end_time, min_lat, max_lat, min_lon, max_lon),
        errors,
    )
    df = df[df["type"] == "earthquake"]
    df["time"] = df["time"].apply(lambda x: dt.datetime.fromtimestamp(x / 1000))
    df["time"] = pd.to_datetime(df["time"], format="mixed")
    df = df[["time", "longitude", "latitude", "depth", "mag", "magType"]]
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    if len(df) == 0:
        print("Last earthquake was more than 30 days ago v2")
        return None, None, None  # No data
    end_time = df["time"].max()
    count = 0
    cnt = 0
    while count < 65:
        if cnt > 50:
            print("No data")
            return None, None, None
        start_time = start_time - dt.timedelta(days=360)
        count = get_earthquake_count(
            make_params_circle(start_time, end_time, Y, X, radius + 10)
        )
        cnt += 1

    df = pd.DataFrame()
    while cnt < 20:
        cnt += 1
        n = count // 20000
        dates = pd.date_range(
            start_time, end_time, periods=2 if count < 20000 else n + 4
        )
        for i in range(len(dates) - 1):
            resp = get_earthquake_data(
                make_params_circle(
                    dates[i].to_pydatetime(),
                    dates[i + 1].to_pydatetime(),
                    Y,
                    X,
                    radius + 10,
                )
            )
            df_small, errors = make_df(
                resp,
                make_params_circle(
                    dates[i].to_pydatetime(),
                    dates[i + 1].to_pydatetime(),
                    Y,
                    X,
                    radius + 10,
                ),
                errors,
            )
            df_small = df_small[df_small["type"] == "earthquake"]
            df_small = df_small[
                ["time", "longitude", "latitude", "depth", "mag", "magType"]
            ]
            df_small["time"] = df_small["time"].apply(
                lambda x: dt.datetime.fromtimestamp(x / 1000)
            )
            df_small.dropna(inplace=True)
            df_small.drop_duplicates(inplace=True)
            df = pd.concat([df, df_small], axis=0).reset_index(drop=True)
        df["pos"] = "0_0"
        df.loc[
            (df["latitude"] >= Y - geo_split / 2)
            & (df["latitude"] <= Y + geo_split / 2)
            & (df["longitude"] >= X - geo_split / 2)
            & (df["longitude"] <= X + geo_split / 2),
            "pos",
        ] = (
            str(Y - geo_split / 2) + "_" + str(X - geo_split / 2)
        )
        df["distance"] = haversine_distance(df["latitude"], df["longitude"], Y, X)
        df["latitude_disc"] = Y - geo_split / 2
        df["longitude_disc"] = X - geo_split / 2
        tmp = df[df["distance"] <= radius]
        if len(tmp) >= block_size + 1:
            return (
                tmp,
                end_time,
                errors,
            )
        end_time = start_time
        start_time = start_time - dt.timedelta(days=360)
    print("No data")
    return None, None, None


def check_coords(X: float, Y: float, regions: List[str], geo_split: int) -> bool:
    """
    Check if the coordinates are in the region.

    Parameters:
    - X (float): Longitude of the center of the region.
    - Y (float): Latitude of the center of the region.
    - regions (List[str]): List of regions.
    - geo_split (int): Size of the region.

    Returns:
    bool: True if the coordinates are in the region, False otherwise.

    Example:
    >>> check_coords(0, 0, ["0_0"], 1)
    True
    """
    if X < -180 or X > 180 or Y < -90 or Y > 90:
        return False
    for pos in regions:
        y, x = pos.split("_")
        y, x = float(y), float(x)
        if X >= x and X <= x + geo_split and Y >= y and Y <= y + geo_split:
            return True
    return False


def map_col(df: pd.DataFrame, col: str, mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Maps the values in a column to integers based on a mapping provided in a pd.DataFrame.

    Parameters:
    - df (pd.DataFrame): Dataframe containing the data.
    - col (str): Name of the column to be mapped.
    - mapping (pd.DataFrame): Dataframe containing the mapping.

    Returns:
    pd.DataFrame: Dataframe with the mapped column.
    """
    mapping = dict(zip(mapping.iloc[:, 0], mapping.iloc[:, 1]))
    df[col] = df[col].map(mapping)
    return df


def preprocess_df(
    df: pd.DataFrame,
    preproc_params: Dict[str, Union[int, float, List[int]]],
    scaler_dict: Dict[str, MinMaxScaler],
) -> pd.DataFrame:
    """
    Preprocess the dataframe based on the provided parameters and scalers. Similar preprocessing took place before training the model.

    Parameters:
    - df (pd.DataFrame): Dataframe containing the data.
    - preproc_params (Dict[str, Union[int, float, List[int]]]): Dictionary containing the preprocessing parameters.
    - scaler_dict (Dict[str, MinMaxScaler]): Dictionary containing the scalers.

    Returns:
    pd.DataFrame: Preprocessed dataframe.
    """
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
    """
    Creates a time series block for the given dataframe and position.

    Parameters:
    - df (pd.DataFrame): Dataframe containing the data.
    - pos (str): Position of the block.
    - radius (int): Radius of the region.
    - block_size (int): Size of the block.
    - preproc_params (Dict[str, Union[int, float, List[int]]]): Dictionary containing the preprocessing parameters.

    Returns:
    pd.DataFrame: Dataframe containing the time series block.
    """
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 10, 14, 21, 30, 60, 180, 1e8]
    lat, lon = pos.split("_")
    lat, lon = float(lat), float(lon)
    tmp1 = df[df["pos"] == pos]
    tmp2 = df[df["pos"] != pos]
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


def reshape(
    df: pd.DataFrame,
    block_size: int,
    feature_order: List[str],
    featrues_region: List[str],
):
    """
    Splits the dataframe into time series and region features. Reshapes the time series features into a 3D array.

    Parameters:
    - df (pd.DataFrame): Dataframe containing the data.
    - block_size (int): Size of the block.
    - feature_order (List[str]): List of features.
    - featrues_region (List[str]): List of region features.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing the following elements:
        - np.ndarray: A 3D array containing the time series features.
        - np.ndarray: A 2D array containing the region features.
    """
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
    """
    Creates a time series block for the given dataframe and position. Uses the provided scalers and preprocessing parameters.
    Takes the most recent time series block.

    Parameters:
    - df (pd.DataFrame): Dataframe containing the data.
    - x (float): Longitude of the center of the region.
    - y (float): Latitude of the center of the region.
    - radius (int): Radius of the region.
    - block_size (int): Size of the block.
    - features_order (List[str]): List of features.
    - features_region (List[str]): List of region features.
    - preproc_params (Dict[str, Union[int, float, List[int]]]): Dictionary containing the preprocessing parameters.
    - scaler_dict (Dict[str, MinMaxScaler]): Dictionary containing the scalers.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing the following elements:
        - np.ndarray: A 3D array containing the time series features.
        - np.ndarray: A 2D array containing the region features.
        - dt.datetime: The date of the last earthquake that occurred at that location.
    """
    df["time"] = pd.to_datetime(df["time"], format="mixed")
    df.sort_values(by="time", inplace=True)
    pos = str(y - GEO_SPLIT / 2) + "_" + str(x - GEO_SPLIT / 2)
    df = preprocess_df(df, preproc_params, scaler_dict)
    df_pos = make_block(df, pos, radius, block_size, preproc_params)
    df_pos = df_pos.iloc[-1]
    time = df_pos["time"]
    x_ts, x_region = reshape(df_pos, block_size, features_order, features_region)
    return x_ts, x_region, time


def prepare_data(df: pd.DataFrame, geo_split: int) -> pd.DataFrame:
    """
    Prepare the data for inference. Add region and tectonic features. Map the categorical features to integers.
    If the region or tectonic features are missing, fill them with intager corresponding to the "Other" category.

    Parameters:
    - df (pd.DataFrame): Dataframe containing the data.
    - geo_split (int): Size of the region.

    Returns:
    pd.DataFrame: Dataframe containing the prepared data.

    Example:
    >>> prepare_data(pd.DataFrame({"latitude": [0, 1], "longitude": [0, 0]}), 1)
        latitude  longitude  lat_cent  lon_cent  plate  plate_region  dist_region  dist  magType
    0          0          0       0.0       0.0   61.0          51.0          0.0   0.0      0.0
    1          1          0       1.0       0.0   61.0          51.0          0.0   0.0      0.0
    """
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

    df["plate_region"] = df["plate_region"].fillna(51)
    df["plate"] = df["plate"].fillna(61)
    return df


def make_prediction(X: float, Y: float) -> Tuple[float, dt.datetime]:
    """
    Make a prediction for the given coordinates. Downloads the necessary data, preprocesses it and makes a prediction using the trained model.

    Parameters:
    - X (float): Longitude of the center of the region.
    - Y (float): Latitude of the center of the region.

    Returns:
    Tuple[float, dt.datetime]: A tuple containing the following elements:
        - float: The probability of an earthquake occurring in the next 30 days at that location.
        - dt.datetime: The date of the last earthquake that occurred at that location.

    """
    regions = np.load("../data/regions.npy", allow_pickle=True)
    if not check_coords(X, Y, regions, GEO_SPLIT):
        print("Wrong coords")
        return None, None
    now = dt.datetime.now()
    START_TIME = now - dt.timedelta(days=30)
    END_TIME = now
    MIN_LAT = Y - GEO_SPLIT / 2
    MAX_LAT = Y + GEO_SPLIT / 2
    MIN_LON = X - GEO_SPLIT / 2
    MAX_LON = X + GEO_SPLIT / 2

    df, START_TIME, errors = get_data(
        X,
        Y,
        START_TIME,
        END_TIME,
        MIN_LAT,
        MAX_LAT,
        MIN_LON,
        MAX_LON,
        GEO_SPLIT,
        RADIUS,
        BLOCK_SIZE,
    )
    if df is None:
        return None, None
    df = prepare_data(df, GEO_SPLIT)

    scalers = pickle.load(open("../data/scalers_for_npys.pkl", "rb"))
    features_order = make_feature_order(FEATURES, BLOCK_SIZE)
    x_ts, x_region, last_eq_time = make_timeseries(
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

    model = tf.keras.models.load_model(
        "../models/model_v1.keras", custom_objects={"MyModel": MyModel}
    )

    return model.predict(x_ts.astype(np.float32))[0][0], last_eq_time
