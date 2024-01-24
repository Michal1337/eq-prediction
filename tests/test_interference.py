import unittest
from datetime import datetime
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from eq_prediction.inference import (
    check_coords,
    make_block,
    make_params_circle,
    make_timeseries,
    map_col,
    preprocess_df,
    reshape,
)
from sklearn.preprocessing import MinMaxScaler


class TestEarthquakeFunctions(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03"]),
                "mag": [5.0, 6.0, 7.0],
                "depth": [10.0, 15.0, 20.0],
                "latitude": [30.0, 31.0, 32.0],
                "longitude": [-120.0, -121.0, -122.0],
                "lat_cent": [30.0, 31.0, 32.0],
                "lon_cent": [-120.0, -121.0, -122.0],
                "dist": [50.0, 55.0, 60.0],
                "dist_region": [100.0, 105.0, 110.0],
                "plate": [1, 2, 3],
                "magType": ["A", "B", "C"],
                "pos": ["30.0_-120.0", "31.0_-121.0", "32.0_-122.0"],
                "distance": [10.0, 15.0, 20.0],
            }
        )

        self.preproc_params = {
            "mag_low": 4.0,
            "mag_high": 8.0,
            "depth_low": 5.0,
            "depth_high": 25.0,
            "dist_low": 5.0,
            "dist_high": 65.0,
            "dist_region_low": 90.0,
            "dist_region_high": 120.0,
            "scale_distance_lag": 2.0,
            "scale_distance": 3.0,
        }

        self.scaler_dict = {
            "mag": MinMaxScaler(),
            "depth": MinMaxScaler(),
            "latitude_new": MinMaxScaler(),
            "longitude_new": MinMaxScaler(),
            "lat_cent": MinMaxScaler(),
            "lon_cent": MinMaxScaler(),
            "dist": MinMaxScaler(),
            "dist_region": MinMaxScaler(),
        }

        # Fit scalers with some example data
        for key, scaler in self.scaler_dict.items():
            if key in self.df.columns:
                scaler.fit(self.df[key].values.reshape(-1, 1))

    def test_make_params_circle(self):
        result = make_params_circle(
            "2023-01-01T00:00:00", "2023-12-31T23:59:59", 35.0, 45.0, -120.0
        )
        expected_result = {
            "format": "geojson",
            "starttime": "2023-01-01T00:00:00",
            "endtime": "2023-12-31T23:59:59",
            "latitude": 35.0,
            "longitude": 45.0,
            "maxradiuskm": -120.0,
        }
        self.assertEqual(result, expected_result)

    def test_check_coords(self):
        result = check_coords(0, 0, ["0_0"], 1)
        self.assertTrue(result)

        result = check_coords(10, 10, ["0_0"], 1)
        self.assertFalse(result)

    def test_map_col(self):
        df = pd.DataFrame({"col1": ["A", "B", "C"], "col2": [1, 2, 3]})
        mapping_df = pd.DataFrame(
            {"col1": ["A", "B", "C"], "mapped_col": [100, 200, 300]}
        )

        result_df = map_col(df, "col1", mapping_df)

        expected_result_df = pd.DataFrame(
            {"col1": ["A", "B", "C"], "col2": [1, 2, 3], "mapped_col": [100, 200, 300]}
        )
        pd.testing.assert_frame_equal(result_df, expected_result_df)

    def test_preprocess_df(self):
        preprocessed_df = preprocess_df(
            self.df.copy(), self.preproc_params, self.scaler_dict
        )
        self.assertIsInstance(preprocessed_df, pd.DataFrame)

    def test_make_block(self):
        block_df = make_block(self.df.copy(), "30.0_-120.0", 15, 5, self.preproc_params)
        self.assertIsInstance(block_df, pd.DataFrame)
        self.assertEqual(len(block_df.columns), 47)

    def test_reshape(self):
        ts_array, region_array = reshape(
            self.df.copy(), 5, ["mag", "depth"], ["lat_cent", "lon_cent"]
        )
        self.assertIsInstance(ts_array, np.ndarray)
        self.assertIsInstance(region_array, np.ndarray)
        self.assertEqual(ts_array.shape, (1, 5, 2))
        self.assertEqual(region_array.shape, (1, 2))

    def test_make_timeseries(self):
        ts_array, region_array, last_time = make_timeseries(
            self.df.copy(),
            -120.0,
            30.0,
            15,
            5,
            ["mag", "depth"],
            ["lat_cent", "lon_cent"],
            self.preproc_params,
            self.scaler_dict,
        )
        self.assertIsInstance(ts_array, np.ndarray)
        self.assertIsInstance(region_array, np.ndarray)
        self.assertIsInstance(last_time, datetime)
        self.assertEqual(ts_array.shape, (1, 5, 2))
        self.assertEqual(region_array.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
