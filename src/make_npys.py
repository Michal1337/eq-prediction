"""
Script for creating the time-series blocks for all regions with more than THRESHOLD earthquakes in the training set.
Save the blocks as .npy files in INPUT_DIRECTORY. Save the scalers as .pkl file in ../data/scalers_for_npys.pkl.

Note that time-series blocks are based not only on the earthquakes from a set region, but also on the earthquakes from a radius RADIUS around the region.
This ensures that the model can also learn from earthquakes that are close to the region, but not in the region itself.
"""
import pickle
import warnings

import numpy as np
import pandas as pd
import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

from typing import Dict, List, Tuple, Union

from .add_features import haversine_distance
from .params import (
    BLOCK_SIZE,
    FEATURES,
    FEATURES_REGION,
    INPUT_DIRECTORY,
    PREPROC_PARAMS,
    RADIUS,
    SEED,
    SPLIT_DATE_TRAIN,
    SPLIT_DATE_VAL,
    THRESHOLD,
)


def filter_regions(
    df: pd.DataFrame, threshold: int, split_date_train: str
) -> np.ndarray[str]:
    """
    Filters out regions with less than threshold earthquakes in the training set.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the earthquake data.
    - threshold (int): The minimum number of earthquakes in a region.
    - split_date_train (str): The date used to split the data into train and test.

    Returns:
    - regions (np.array): The regions with more than threshold earthquakes in the training set.

    Example:
    >>> df = pd.DataFrame({"time": ["2020-01-01", "2020-01-01", "2020-01-01"], "pos": ["0_0", "0_0", "1_1"]})
    >>> filter_regions(df, 2, "2020-01-01")
    array(['0_0'], dtype=object)
    """
    df_f = df[df["time"] <= split_date_train]
    df_agg = df_f.groupby(["pos"]).agg({"mag": "count"}).reset_index()
    regions = df_agg.loc[df_agg["mag"] >= threshold, "pos"].values
    return regions


def preprocess_df(
    df: pd.DataFrame,
    preproc_params: Dict[str, Union[int, float, List[int]]],
    split_date_train: str,
) -> Tuple[pd.DataFrame, Dict[str, MinMaxScaler]]:
    """
    Preprocesses the dataframe by scaling the features. Explenation of the preprocessing can be found in ../utils/check_dist.ipynb.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the earthquake data.
    - preproc_params (Dict[str, Union[int, float, List[int]]]): The parameters used for preprocessing.
    - split_date_train (str): The date used to split the data into train and test.

    Returns:
    - df (pd.DataFrame): The dataframe containing the earthquake data with scaled features.
    - scaler_dict (Dict[str, MinMaxScaler]): The scalers used for scaling the features.
    """
    scaler_dict = {}
    df_train = df[df["time"] < split_date_train]

    scaler = MinMaxScaler()
    mag = np.clip(
        df_train["mag"].values, preproc_params["mag_low"], preproc_params["mag_high"]
    )
    scaler.fit(mag.reshape(-1, 1))
    df["mag"] = scaler.transform(
        np.clip(
            df["mag"].values, preproc_params["mag_low"], preproc_params["mag_high"]
        ).reshape(-1, 1)
    )
    scaler_dict["mag"] = scaler

    scaler = MinMaxScaler()
    depth = np.log(df_train["depth"] + np.abs(df["depth"].min()) + 1).values
    depth = np.clip(depth, preproc_params["depth_low"], preproc_params["depth_high"])
    scaler.fit(depth.reshape(-1, 1))
    df["depth"] = np.log(df["depth"] + np.abs(df["depth"].min()) + 1)
    df["depth"] = scaler.transform(
        np.clip(
            df["depth"].values,
            preproc_params["depth_low"],
            preproc_params["depth_high"],
        ).reshape(-1, 1)
    )
    scaler_dict["depth"] = scaler

    scaler = MinMaxScaler()
    scaler.fit(df_train["latitude"].values.reshape(-1, 1))
    df["latitude_new"] = scaler.transform(df["latitude"].values.reshape(-1, 1))
    scaler_dict["latitude_new"] = scaler

    scaler = MinMaxScaler()
    scaler.fit(df_train["longitude"].values.reshape(-1, 1))
    df["longitude_new"] = scaler.transform(df["longitude"].values.reshape(-1, 1))
    scaler_dict["longitude_new"] = scaler

    scaler = MinMaxScaler()
    scaler.fit(df_train["lat_cent"].values.reshape(-1, 1))
    df["lat_cent"] = scaler.transform(df["lat_cent"].values.reshape(-1, 1))
    scaler_dict["lat_cent"] = scaler

    scaler = MinMaxScaler()
    scaler.fit(df_train["lon_cent"].values.reshape(-1, 1))
    df["lon_cent"] = scaler.transform(df["lon_cent"].values.reshape(-1, 1))
    scaler_dict["lon_cent"] = scaler

    scaler = MinMaxScaler()
    dist = np.log(df_train["dist"] + 1).values.reshape(-1, 1)
    dist = np.clip(dist, PREPROC_PARAMS["dist_low"], PREPROC_PARAMS["dist_high"])
    scaler.fit(dist)
    df["dist"] = scaler.transform(
        np.clip(
            np.log(df["dist"] + 1).values.reshape(-1, 1),
            PREPROC_PARAMS["dist_low"],
            PREPROC_PARAMS["dist_high"],
        )
    )
    scaler_dict["dist"] = scaler

    scaler = MinMaxScaler()
    dist_region = np.log(df_train["dist_region"] + 1).values.reshape(-1, 1)
    dist_region = np.clip(
        dist_region,
        PREPROC_PARAMS["dist_region_low"],
        PREPROC_PARAMS["dist_region_high"],
    )
    scaler.fit(dist_region)
    df["dist_region"] = scaler.transform(
        np.clip(
            np.log(df["dist_region"] + 1).values.reshape(-1, 1),
            PREPROC_PARAMS["dist_region_low"],
            PREPROC_PARAMS["dist_region_high"],
        )
    )
    scaler_dict["dist_region"] = scaler

    return df, scaler_dict


def make_block(
    df: pd.DataFrame,
    pos: str,
    radius: int,
    block_size: int,
    preproc_params: Dict[str, Union[int, float, List[int]]],
) -> pd.DataFrame:
    """
    Creates a time-series block for a given region. The block is created by taking all earthquakes in the region and all earthquakes in a radius radious around the region.
    Additionally, a distance feature is added to the block, which is the distance between the center of the region and the earthquake.
    Also, diff_days is added, which is the number of days between the current earthquake and the previous earthquake.
    The diff_days feature is discretized into bins, which are defined in preproc_params.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the earthquake data.
    - pos (str): The position of the region.
    - radius (int): The radius around the region.
    - block_size (int): The size of the block.
    - preproc_params (Dict[str, Union[int, float, List[int]]]): The parameters used for preprocessing.

    Returns:
    - tmp (pd.DataFrame): The dataframe containing the block.
    """
    lat, lon = pos.split("_")
    lat, lon = float(lat), float(lon)
    tmp1 = df[df["pos"] == pos]
    tmp2 = df[df["pos"] != pos]
    tmp2["label"] = -1
    tmp = pd.concat([tmp1, tmp2], axis=0)
    tmp["distance"] = haversine_distance(
        tmp["latitude"], tmp["longitude"], lat + 0.5, lon + 0.5
    )
    tmp = tmp[tmp["distance"] <= radius]
    tmp.sort_values(by=["time"], inplace=True)
    tmp["diff_days"] = (tmp["time"] - tmp["time"].shift(1)).dt.days
    tmp.dropna(inplace=True)
    tmp["diff_days"] = np.digitize(tmp["diff_days"], bins=preproc_params["bins"]) - 1
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
    tmp = tmp[tmp["label"] != -1]
    tmp["distance"] = tmp["distance"] / preproc_params["scale_distance"]
    tmp.dropna(inplace=True)
    return tmp


def reshape(
    df: pd.DataFrame,
    block_size: int,
    feature_order: List[str],
    featrues_region: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the dataframe into x, x_region and y. x is the time-series block, x_region is the region features and y is the label.
    x is in shape (n_samples, block_size, n_features), x_region is in shape (n_samples, n_features_region) and y is in shape (n_sample, 1).

    Parameters:
    - df (pd.DataFrame): The dataframe containing the earthquake data.
    - block_size (int): The size of the block.
    - feature_order (List[str]): The order of the features in the block.
    - featrues_region (List[str]): The features of the region.

    Returns:
    - x (np.ndarray): The time-series block.
    - x_region (np.ndarray): The region features.
    - y (np.ndarray): The labels.

    Example:
    >>> df = pd.DataFrame({"mag": [1, 3, 4], "mag_1": [3, 4, 6], "label": [0, 1, 0]})
    >>> reshape(df, 2, ["mag_1", "mag"], [])
    (array([[[3, 1],
             [4, 3],
             [4, 6]]]), array([], shape=(3, 0), dtype=float64), array([[0], [0], [1]]))
    """
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    x = (
        df[feature_order]
        .to_numpy()
        .reshape(-1, block_size, len(feature_order) // block_size)
    )
    x_region = df[featrues_region].to_numpy().reshape(-1, len(featrues_region))
    y = df["label"].to_numpy().reshape(-1, 1)
    return x, x_region, y


def split_all(
    df: pd.DataFrame,
    block_size: int,
    feature_order: List[str],
    features_region: List[str],
    split_date_train: str,
    split_date_val: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Splits the dataframe into train, validation and test set based on the split dates. All x blocks are reshaped into (n_samples, block_size, n_features) and all x_region blocks are reshaped into (n_samples, n_features_region).
    The y blocks are reshaped into (n_samples, 1). The x blocks have features in the order defined by feature_order and the x_region blocks have features in the order defined by features_region.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the earthquake data.
    - block_size (int): The size of the block.
    - feature_order (List[str]): The order of the features in the block.
    - featrues_region (List[str]): The features of the region.
    - split_date_train (str): The date used to split the data into train and test.
    - split_date_val (str): The date used to split the data into train and test.

    Returns:
    - x_train (np.ndarray): The time-series block of the train set.
    - x_train_region (np.ndarray): The region features of the train set.
    - y_train (np.ndarray): The labels of the train set.
    - x_val (np.ndarray): The time-series block of the validation set.
    - x_val_region (np.ndarray): The region features of the validation set.
    - y_val (np.ndarray): The labels of the validation set.
    - x_test (np.ndarray): The time-series block of the test set.
    - x_test_region (np.ndarray): The region features of the test set.
    - y_test (np.ndarray): The labels of the test set.
    """
    df_train = df[df["time"] < split_date_train]
    df_val = df[(df["time"] >= split_date_train) & (df["time"] < split_date_val)]
    df_test = df[df["time"] >= split_date_val]
    x_train, x_train_region, y_train = reshape(
        df_train, block_size, feature_order, features_region
    )
    x_val, x_val_region, y_val = reshape(
        df_val, block_size, feature_order, features_region
    )
    x_test, x_test_region, y_test = reshape(
        df_test, block_size, feature_order, features_region
    )
    return (
        x_train,
        x_train_region,
        y_train,
        x_val,
        x_val_region,
        y_val,
        x_test,
        x_test_region,
        y_test,
    )


def make_npys(
    df: pd.DataFrame,
    radius: int,
    th: int,
    block_size: int,
    features_order: List[str],
    features_region: List[str],
    preproc_params: Dict[str, Union[int, float, List[int]]],
    split_date_train: str,
    split_date_val: str,
    input_directory: str,
) -> Dict[str, MinMaxScaler]:
    """
    Filters out regions with less than th earthquakes in the training set.
    Preprocesses the dataframe by scaling the features.
    Creates the time-series blocks for all filtered out regions. The blocks are saved as .npy files in input_directory.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the earthquake data.
    - radius (int): The radius around the region.
    - th (int): The minimum number of earthquakes in a region.
    - block_size (int): The size of the block.
    - features_order (List[str]): The order of the features in the block.
    - featrues_region (List[str]): The features of the region.
    - preproc_params (Dict[str, Union[int, float, List[int]]]): The parameters used for preprocessing.
    - split_date_train (str): The date used to split the data into train and test.
    - split_date_val (str): The date used to split the data into train and test.
    - input_directory (str): The directory where the blocks are saved.

    Returns:
    - scaler_dict (Dict[str, MinMaxScaler]): The scalers used for scaling the features.
    """
    df["time"] = pd.to_datetime(df["time"], format="mixed")
    df.sort_values(by="time", inplace=True)
    regions = filter_regions(df, th, split_date_train)
    df, scaler_dict = preprocess_df(df, preproc_params, split_date_train)
    np.random.shuffle(regions)
    for idx, pos in enumerate(tqdm.tqdm(regions)):
        df_pos = make_block(df, pos, radius, block_size, preproc_params)
        (
            x_train,
            x_train_region,
            y_train,
            x_val,
            x_val_region,
            y_val,
            x_test,
            x_test_region,
            y_test,
        ) = split_all(
            df_pos,
            block_size,
            features_order,
            features_region,
            split_date_train,
            split_date_val,
        )
        np.save(input_directory + "x_train_" + str(idx) + ".npy", x_train)
        np.save(input_directory + "x_train_region_" + str(idx) + ".npy", x_train_region)
        np.save(input_directory + "y_train_" + str(idx) + ".npy", y_train)
        np.save(input_directory + "x_val_" + str(idx) + ".npy", x_val)
        np.save(input_directory + "x_val_region_" + str(idx) + ".npy", x_val_region)
        np.save(input_directory + "y_val_" + str(idx) + ".npy", y_val)
        np.save(input_directory + "x_test_" + str(idx) + ".npy", x_test)
        np.save(input_directory + "x_test_region_" + str(idx) + ".npy", x_test_region)
        np.save(input_directory + "y_test_" + str(idx) + ".npy", y_test)
    return scaler_dict


def make_feature_order(features: List[str], block_size: int) -> List[str]:
    """
    Creates the feature order for the time-series block. The order is defined as follows: [feature_1_lag_block_size-1, feature_2_lag_block_size-1, ..., feature_1_lag_0, feature_2_lag_0].

    Parameters:
    - features (List[str]): The features of the block.
    - block_size (int): The size of the block.

    Returns:
    - features_order (List[str]): The feature order of the block.

    Example:
    >>> make_feature_order(["mag", "depth"], 2)
    ['mag_1', 'depth_1', 'mag_0', 'depth_0']
    """
    features_order = [
        features[idx] + "_" + str(i)
        for i in range(block_size - 1, 0, -1)
        for idx in range(len(features))
    ]
    features_order = features_order + features
    return features_order


def main():
    """
    Creates the time-series blocks for all regions with more than THRESHOLD earthquakes in the training set.
    Saves the blocks as .npy files in INPUT_DIRECTORY.
    Saves the scalers as .pkl file in ../data/scalers_for_npys.pkl.
    """
    df = pd.read_csv("../data/with_features.csv")
    df.dropna(inplace=True)

    features_order = make_feature_order(FEATURES, BLOCK_SIZE)
    scalers = make_npys(
        df,
        RADIUS,
        THRESHOLD,
        BLOCK_SIZE,
        features_order,
        FEATURES_REGION,
        PREPROC_PARAMS,
        SPLIT_DATE_TRAIN,
        SPLIT_DATE_VAL,
        INPUT_DIRECTORY,
    )

    with open("../data/scalers_for_npys.pkl", "wb") as f:
        pickle.dump(scalers, f)


if __name__ == "__main__":
    main()
