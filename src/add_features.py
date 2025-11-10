"""
Script for adding features to the data. Adds region and tectonic information.
Adds label column with 1 if there is an earthquake with magnitude >= mag_th in the TIME_DIFF in the future.
Note that a second dataframe, with columns [plate, lat, lon] is needed as the tectonic information.
"""
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import tqdm

from .params import GEO_SPLIT, MAG_TH, TIME_CUT, TIME_DIFF, SPLIT_DATE_TRAIN


def haversine_distance(
    lat1: Union[float, List[float]],
    lon1: Union[float, List[float]],
    lat2: Union[float, List[float]],
    lon2: Union[float, List[float]],
) -> Union[float, np.ndarray[float]]:
    """
    Approximation of distance between two points on Earth using the haversine formula.
    Function is vectorized to handle arrays of coordinates.

    Parameters:
    - lat1 (Union[float, List[float]]): Latitude of first point.
    - lon1 (Union[float, List[float]]): Longitude of first point.
    - lat2 (Union[float, List[float]]): Latitude of second point.
    - lon2 (Union[float, List[float]]): Longitude of second point.

    Returns:
    - distance (Union[float, np.ndarray[float]]): Distance between the two points in kilometers.

    Example:
    >>> haversine_distance(0, 0, 0, 1)
    111.195079734
    >>> haversine_distance(0, 0, [1, 1], [0, 0])
    array([111.19507973, 111.19507973])
    """
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance


def find_min_dist(df_tp: pd.DataFrame, x: float, y: float) -> Tuple[float, str]:
    """
    Finds the distance to the nearest tectonic plates boundries and the ids of the nearest plates.
    Concatenates the ids of the nearest plates into one string.

    Parameters:
    - df_tp (pd.DataFrame): Dataframe containing the tectonic plate data.
    - x (float): Latitude of the point.
    - y (float): Longitude of the point.

    Returns:
    - min_dist (float): Distance to the nearest tectonic plate.
    - plates (str): Concatenated ids of the nearest tectonic plates.

    Example:
    >>> df_tp = pd.DataFrame({"lat": [0, 1], "lon": [0, 0], "plate": [1, 2]})
    >>> find_min_dist(df_tp, 0, 0)
    (0.0, '1')
    """
    df_tp["dist"] = haversine_distance(x, y, df_tp["lat"], df_tp["lon"])
    min_dist = df_tp["dist"].min()
    plates = df_tp[df_tp["dist"] == min_dist].sort_values("plate")["plate"].tolist()
    plates = "_".join(plates)
    return min_dist, plates


def add_region_info(
    df: pd.DataFrame, df_tp: pd.DataFrame, geo_split: int
) -> pd.DataFrame:
    """
    Add regional information to df dataframe. Adds cordinates of centers of all sqaures
    and the distance to the nearest tectonic plate, along with id of the nearest plate.

    Parameters:
    - df (pd.DataFrame): Dataframe containing the earthquake data.
    - df_tp (pd.DataFrame): Dataframe containing the tectonic plate data.
    - geo_split (int): Size of the squares in degrees.

    Returns:
    - df (pd.DataFrame): Dataframe containing the earthquake data with added information.

    Example:
    >>> df = pd.DataFrame({"latitude": [0, 1], "longitude": [0, 0]})
    >>> df_tp = pd.DataFrame({"lat": [0, 1], "lon": [0, 0], "plate": [1, 2]})
    >>> add_region_info(df, df_tp, 1)
       latitude  longitude  lat_cent  lon_cent  dist_region  plate_region
    0         0          0       0.5       0.5          0.0             1
    1         1          0       1.5       0.5          0.0             2
    """
    region2plate = {}
    region2dist = {}
    df["lat_cent"] = df["latitude_disc"] + geo_split / 2
    df["lon_cent"] = df["longitude_disc"] + geo_split / 2
    for pos in tqdm.tqdm(df["pos"].unique()):
        x, y = pos.split("_")
        x, y = float(x), float(y)
        dist, plate = find_min_dist(df_tp, x + geo_split / 2, y + geo_split / 2)
        region2plate[pos] = plate
        region2dist[pos] = dist
    df["plate_region"] = df["pos"].map(region2plate)
    df["dist_region"] = df["pos"].map(region2dist)
    return df


def add_tectonic_info(df: pd.DataFrame, df_tp: pd.DataFrame) -> pd.DataFrame:
    """
    Adds distance to the nearest tectonic plate and id of the nearest plate to each earthquake in df dataframe.

    Parameters:
    - df (pd.DataFrame): Dataframe containing the earthquake data.
    - df_tp (pd.DataFrame): Dataframe containing the tectonic plate data.

    Returns:
    - df (pd.DataFrame): Dataframe containing the earthquake data with added information.

    Example:
    >>> df = pd.DataFrame({"latitude": [0, 1], "longitude": [0, 0]})
    >>> df_tp = pd.DataFrame({"lat": [0, 1], "lon": [0, 0], "plate": [1, 2]})
    >>> add_tectonic_info(df, df_tp)
       latitude  longitude  dist  plate
    0         0          0   0.0      1
    1         1          0   0.0      2
    """
    coordinates = list(zip(df["latitude"], df["longitude"]))
    results = list(
        tqdm.tqdm(
            map(lambda x: find_min_dist(df_tp, x[0], x[1]), coordinates),
            total=len(coordinates),
        )
    )
    df[["dist", "plate"]] = results
    return df


def preprocess_magtype(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Converts magtype names from string to int. Returns dataframe with converted names and dictionary with mapping.
    Note that 13 most common magtypes are mapped to integers 1-13, the rest to 14.

    Parameters:
    - df (pd.DataFrame): Dataframe containing the earthquake data.

    Returns:
    - df (pd.DataFrame): Dataframe containing the earthquake data with converted names.
    - magtype2id (Dict[str, int]): Dictionary mapping magtype names to integers.

    Example:
    >>> preprocess_magtype(df)
    """
    magtype2id = {
        magtype: i + 1
        for i, magtype in enumerate(df["magType"].value_counts().index[:17])
    }
    magtype2id.update(
        {magtype: 18 for magtype in df["magType"].value_counts().index[17:]}
    )
    df["magType"] = df["magType"].map(magtype2id)
    return df, magtype2id


def initial_preprocess(df: pd.DataFrame, time_cut: str, geo_split: int) -> pd.DataFrame:
    """
    Initial preproc. Drop duplicates and NaNs, convert time to datetime. Filter data before time_cut, adds time discretized to days.
    Splits coordinates into squares of size geo_split degrees. Adds pos column with string representation of squares.

    Parameters:
    - df (pd.DataFrame): Dataframe containing the earthquake data.
    - time_cut (str): Time to filter data before.
    - geo_split (int): Size of the squares in degrees.

    Returns:
    - df (pd.DataFrame): Dataframe containing the earthquake data with added information.

    Example:
    >>> df = pd.DataFrame({"time": ["1973-01-01T00:00:00", "1973-01-01T00:00:00", "1973-01-01T00:00:00"],
    ...                    "longitude": [0, 0, 0], "latitude": [0, 0, 0]})
    >>> initial_preproces(df, "1973-01-01T00:00:00", 1)
                       time  longitude  latitude    time_disc  longitude_disc  latitude_disc pos
    0  1973-01-01T00:00:00          0         0  1973-01-01T               0              0  0_0
    """
    print(f"Length of df: {len(df)}")
    print(
        f"Percentage of missing values: {round((df.isna().sum().sum() / df.size) * 100, 2)}%"
    )
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    print(f"Length of df after dropping duplicates and NaNs: {len(df)}")

    df["time_disc"] = df["time"].apply(lambda x: x[:10])
    df = df[df["time"] >= time_cut]
    df["time"] = pd.to_datetime(df["time"], format="mixed")
    print(f"Length of df after filtering before {time_cut}: {len(df)}")

    df["longitude_disc"] = (df["longitude"] // geo_split * geo_split).astype(int)
    df["latitude_disc"] = (df["latitude"] // geo_split * geo_split).astype(int)
    df["pos"] = df["latitude_disc"].astype(str) + "_" + df["longitude_disc"].astype(str)

    return df


def add_features(
    df: pd.DataFrame,
    df_tp: pd.DataFrame,
    geo_split: int,
    mag_th: float,
    time_diff: pd.DateOffset,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Adds features to the dataframe. Adds region and tectonic information. Adds label column with 1 if there is an earthquake with magnitude >= mag_th in the time_diff in the future.

    Parameters:
    - df (pd.DataFrame): Dataframe containing the earthquake data.
    - df_tp (pd.DataFrame): Dataframe containing the tectonic plate data.
    - geo_split (int): Size of the squares in degrees.
    - mag_th (float): Threshold magnitude.
    - time_diff (pd.DateOffset): Time difference to look for earthquakes in the future.

    Returns:
    - df (pd.DataFrame): Dataframe containing the earthquake data with added information.
    """
    df_final = None
    df["time"] = pd.to_datetime(df["time"], format="mixed")
    df["time_disc"] = pd.to_datetime(df["time_disc"], format="mixed")
    df = add_region_info(df, df_tp, geo_split)
    df = add_tectonic_info(df, df_tp)
    for pos in tqdm.tqdm(df["pos"].unique()):
        dfs = []
        tmp = df[df["pos"] == pos]
        tmp.sort_values("time", inplace=True)
        for time in tmp["time_disc"].unique():
            tmp_t0 = tmp[tmp["time_disc"] == time]
            t1 = time + time_diff
            tmp_t1 = tmp[(tmp["time_disc"] > time) & (tmp["time_disc"] <= t1)]
            if tmp_t1.empty:
                max_mag = -1e8
            else:
                max_mag = tmp_t1["mag"].max()
            tmp_t0["label"] = 0 if max_mag < mag_th else 1
            dfs.append(tmp_t0)
        df_tmp = pd.concat(dfs)
        df_final = pd.concat([df_final, df_tmp])
    return df_final


def make_mapping(
    df: pd.DataFrame, col: str, n: int, split_date_train: str
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Makes mapping from most common values to integers. Maps the n most common values in the colun col to integers.
    Takes into account only the data before split_date_train. Maps the rest of the values to n+1. Maps values that are not in the training data to n+1.

    Parameters:
    - df (pd.DataFrame): Dataframe containing the earthquake data.
    - col (str): Name of the column to map.
    - n (int): Number of most common values to map.
    - split_date_train (str): Date to split the data before.

    Returns:
    - df (pd.DataFrame): Dataframe containing the earthquake data with mapped column.
    - type2id (Dict[str, int]): Dictionary mapping values to integers.
    """
    type2id = {
        type: i + 1
        for i, type in enumerate(
            df[df["time"] < split_date_train][col].value_counts().index[:n]
        )
    }
    type2id.update(
        {
            type: n + 1
            for type in df[df["time"] < split_date_train][col].value_counts().index[n:]
        }
    )
    type2id.update({x: n + 1 for x in df[col].unique() if x not in type2id})
    df[col] = df[col].map(type2id)
    return df, type2id


def main():
    """
    Main function for adding features to the data. Adds region and tectonic information. Adds label column with 1 if there is an earthquake with magnitude >= mag_th in the TIME_DIFF in the future.
    Saves the data withou mapping to "../data/with_features_notmapped.csv" and the data with mapping to "../data/with_features.csv". Saves the mapping to "../data/magtype2id.csv", "../data/plate2id.csv" and "../data/plate_region2id.csv".
    """
    df = pd.read_csv("../data/usgs_data_small.csv")
    df_tp = pd.read_csv("../data/all.csv")
    df_tp.drop_duplicates(inplace=True)

    df = initial_preprocess(df, TIME_CUT, GEO_SPLIT)
    df_final = add_features(df, df_tp, GEO_SPLIT, MAG_TH, TIME_DIFF)
    df_final.to_csv("../data/with_features_notmapped.csv", index=False)

    df_final, magtype2id = make_mapping(df_final, "magType", 17, SPLIT_DATE_TRAIN)
    df_final, plate2id = make_mapping(df_final, "plate", 60, SPLIT_DATE_TRAIN)
    df_final, plate_region2id = make_mapping(
        df_final, "plate_region", 50, SPLIT_DATE_TRAIN
    )

    df_final.to_csv("../data/with_features.csv", index=False)

    pd.DataFrame(magtype2id.items(), columns=["magType", "magType_id"]).to_csv(
        "../data/magtype2id.csv", index=False
    )
    pd.DataFrame(plate2id.items(), columns=["plate", "plate_id"]).to_csv(
        "../data/plate2id.csv", index=False
    )
    pd.DataFrame(
        plate_region2id.items(), columns=["plate_region", "plate_region_id"]
    ).to_csv("../data/plate_region2id.csv", index=False)


if __name__ == "__main__":
    main()
