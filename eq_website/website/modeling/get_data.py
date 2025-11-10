"""
Script used to download earthquake data from the USGS API based on the parameters from params.py. This is all the earthquake data necessary for the model training. The result of the scripts are two csv files: usgs_data.csv and usgs_data_small.csv.
The first file contains all the earthquake data from the USGS API. The second file contains a subset of the data from the first file, used for model training. Contains the columns "time", "longitude", "latitude", "depth", and "mag".
"""
import datetime as dt
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import requests
import tqdm

from .params import (END_TIME, MAX_LAT, MAX_LON, MIN_LAT, MIN_LON, MIN_TIME,
                     START_TIME)


def make_params(
    starttime: str,
    endtime: str,
    minlatitude: float,
    maxlatitude: float,
    minlongitude: float,
    maxlongitude: float,
) -> Dict[str, Union[str, float]]:
    """
    Construct a dictionary of parameters for USGS API request. https://earthquake.usgs.gov/fdsnws/event/1/#parameters.

    Parameters:
    - starttime (str): Start time for search
    - endtime (str): End time for search
    - minlatitude (float): Minimum latitude for search
    - maxlatitude (float): Maximum latitude for search
    - minlongitude (float): Minimum longitude for search
    - maxlongitude (float): Maximum longitude for search

    Returns:
    Dict[str, Union[str, int]]: A dictionary containing the parameters for the geojson request.
         The dictionary includes the following keys:
         - "format": "geojson"
         - "starttime": The provided start time.
         - "endtime": The provided end time.
         - "minlatitude": The provided minimum latitude.
         - "maxlatitude": The provided maximum latitude.
         - "minlongitude": The provided minimum longitude.
         - "maxlongitude": The provided maximum longitude.

    Example:
    >>> make_params("2023-01-01T00:00:00", "2023-12-31T23:59:59", 35.0, 45.0, "-120.0", -110.0)
    {'format': 'geojson',
     'starttime': '2023-01-01T00:00:00',
     'endtime': '2023-12-31T23:59:59',
     'minlatitude': 35.0,
     'maxlatitude': 45.0,
     'minlongitude': '-120.0',
     'maxlongitude': -110}.0
    """
    params = {
        "format": "geojson",
        "starttime": starttime,
        "endtime": endtime,
        "minlatitude": minlatitude,
        "maxlatitude": maxlatitude,
        "minlongitude": minlongitude,
        "maxlongitude": maxlongitude,
    }
    return params


def make_datarange(start_time: str, end_time: str, min_time: str) -> List[dt.datetime]:
    """
    Construct a list of dates to use for USGS API requests. The list will be used to make requests for each week between the start and end times. Note that the first element in the list will be the min_time,
    which is used to get the spares earthquakes before the start time.

    Parameters:
    - start_time (str): Start time for search
    - end_time (str): End time for search
    - min_time (str): Minimum time to search

    Returns:
    list[dt.datetime]: A list of dates to use for USGS API requests.

    Example:
    >>> make_datarange("2023-01-01T00:00:00", "2023-01-31T23:59:59", "100-01-01")
    ['100-01-01',
     '2023-01-01',
     '2023-01-08',
     '2023-01-15',
     '2023-01-22',
     '2023-01-29']
    """
    datarange = pd.date_range(start_time, end_time, freq="W").tolist()
    datarange = [str(x)[:10] for x in datarange]
    datarange.insert(0, min_time)
    return datarange


def get_earthquake_count(params: Dict[str, Union[str, int]], data_source: str) -> int:
    """
    Get the number of earthquakes for a given set of parameters from the USGS API based on the params dictionary.
    Used to check that the number of earthquakes in the dataframe matches the number of earthquakes in the API response.

    Parameters:
    - params (Dict[str, Union[str, int]]): A dictionary containing the parameters for the geojson request.
         The dictionary includes the following keys:
         - "format": "geojson"
         - "starttime": The provided start time.
         - "endtime": The provided end time.
         - "minlatitude": The provided minimum latitude.
         - "maxlatitude": The provided maximum latitude.
         - "minlongitude": The provided minimum longitude.
         - "maxlongitude": The provided maximum longitude.

    Returns:
    int: The number of earthquakes for the given parameters.

    Example:
    >>> get_earthquake_count({"format": "geojson",
     "starttime": "2023-01-01T00:00:00",
     "endtime": "2023-12-31T23:59:59",
     "minlatitude": 35.0,
     "maxlatitude": 45,
     "minlongitude": "-120.0",
     "maxlongitude": -110})
    100
    """
    if data_source == 'local':
        url = "http://127.0.0.1:8000/api/eqs"
        params['count'] = True
    else:
        url = "https://earthquake.usgs.gov/fdsnws/event/1/count"
    response = requests.get(url, params=params)
    return response.json()["count"]


def get_earthquake_data(params: Dict[str, str | int], data_source: str) -> requests.models.Response:
    """
    Get the earthquake data for a given set of parameters from the USGS API based on the params dictionary.

    Parameters:
    - params (Dict[str, Union[str, int]]): A dictionary containing the parameters for the geojson request.
         The dictionary includes the following keys:
         - "format": "geojson"
         - "starttime": The provided start time.
         - "endtime": The provided end time.
         - "minlatitude": The provided minimum latitude.
         - "maxlatitude": The provided maximum latitude.
         - "minlongitude": The provided minimum longitude.
         - "maxlongitude": The provided maximum longitude.

    Returns:
    requests.models.Response: The response from the API request.

    Example:
    >>> get_earthquake_data({"format": "geojson",
     "starttime": "2023-01-01T00:00:00",
     "endtime": "2023-12-31T23:59:59",
     "minlatitude": 35.0,
     "maxlatitude": 45,
     "minlongitude": "-120.0",
     "maxlongitude": -110})
    <Response [200]>
    """
    if data_source == 'local':
        url = "http://127.0.0.1:8000/api/eqs"
    else:
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    response = requests.get(url, params=params)
    return response


def make_df(
    resp: requests.models.Response,
    params: Dict[str, Union[str, int]],
    errors: List[Dict[str, Union[str, int]]], data_source: str
) -> Tuple[pd.DataFrame, List[Dict[str, Union[str, int]]]]:
    """
    Make a dataframe from the response of a USGS API request.

    Parameters:
    - resp (requests.models.Response): The response from the API request.
    - params (Dict[str, Union[str, int]]): A dictionary containing the parameters for the geojson request.
    - errors (List[Dict[str, Union[str, int]]]): A list of dictionaries containing the parameters for the geojson request that resulted in an error.

    Returns:
    Tuple(pd.DataFrame, List[Dict[str, Union[str, int]]]): A tuple containing the dataframe with the earthquake data and the list with the parameters that resulted in an error.

    Example:
    >>> make_df(<Response [200]>, {"format": "geojson",
     "starttime": "2023-01-01T00:00:00",
     "endtime": "2023-12-31T23:59:59",
     "minlatitude": 35.0,
     "maxlatitude": 45,
     "minlongitude": "-120.0",
     "maxlongitude": -110}, [])
    """
    all_eqs = []
    try:
        for eq in resp.json()["features"]:
            prop = list(eq["properties"].values())
            prop.extend(eq["geometry"]["coordinates"])
            all_eqs.append(prop)
        if data_source == 'local':
            cols = ['time', 'depth', 'mag', 'magType', 'place', 'alert', 'type', 'cdi', 'mmi', 'felt', 'sig', "longitude", "latitude"]
        else:
            cols = list(resp.json()["features"][0]["properties"].keys())
            cols.extend(["longitude", "latitude", "depth"])
        df = pd.DataFrame(all_eqs, columns=cols)
    except Exception as e:
        print("Halo halo")
        print(e)
        errors.append(params)
        df = pd.DataFrame()
    return df, errors
