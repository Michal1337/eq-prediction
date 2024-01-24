import unittest

import numpy as np
import pandas as pd
from eq_prediction import filter_regions, make_block, preprocess_df, reshape, split_all
from sklearn.preprocessing import MinMaxScaler


class TestYourFunctions(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
                "pos": ["0_0", "0_0", "1_1"],
                "mag": [3.0, 4.0, 5.0],
                "depth": [10.0, 15.0, 20.0],
            }
        )

    def test_filter_regions(self):
        threshold = 2
        split_date_train = "2020-01-01"
        regions = filter_regions(self.df, threshold, split_date_train)
        expected_regions = np.array(["0_0"])
        np.testing.assert_array_equal(regions, expected_regions)

    def test_preprocess_df(self):
        preproc_params = {
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
        split_date_train = "2020-01-01"
        df_processed, scaler_dict = preprocess_df(
            self.df, preproc_params, split_date_train
        )
        self.assertEqual(len(scaler_dict), 2)
        self.assertIsInstance(scaler_dict["mag"], MinMaxScaler)
        self.assertIsInstance(scaler_dict["depth"], MinMaxScaler)
        self.assertIsInstance(df_processed, pd.DataFrame)

    def test_make_block(self):
        pos = "0_0"
        radius = 50
        block_size = 2
        preproc_params = {
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
        block_df = make_block(self.df, pos, radius, block_size, preproc_params)
        self.assertIsInstance(block_df, pd.DataFrame)
        self.assertEqual(len(block_df), 1)
        self.assertEqual(len(block_df.columns), 6)

    def test_reshape(self):
        block_size = 2
        feature_order = ["mag", "depth", "mag_1", "depth_1"]
        features_region = ["latitude", "longitude"]
        preproc_params = {
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
        block_df = make_block(self.df, "0_0", 50, block_size, preproc_params)
        x, x_region, y = reshape(block_df, block_size, feature_order, features_region)
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(x_region, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(x.shape, (1, block_size, 2))
        self.assertEqual(x_region.shape, (1, 2))
        self.assertEqual(y.shape, (1, 1))

    def test_split_all(self):
        pos = "0_0"
        radius = 50
        block_size = 2
        feature_order = ["mag", "depth", "mag_1", "depth_1"]
        features_region = ["latitude", "longitude"]
        split_date_train = "2020-01-01"
        split_date_val = "2020-01-02"
        preproc_params = {
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
        block_df = make_block(self.df, pos, radius, block_size, preproc_params)
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
            block_df,
            block_size,
            feature_order,
            features_region,
            split_date_train,
            split_date_val,
        )

        self.assertIsInstance(x_train, np.ndarray)
        self.assertIsInstance(x_train_region, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)
        self.assertIsInstance(x_val, np.ndarray)
        self.assertIsInstance(x_val_region, np.ndarray)
        self.assertIsInstance(y_val, np.ndarray)
        self.assertIsInstance(x_test, np.ndarray)
        self.assertIsInstance(x_test_region, np.ndarray)
        self.assertIsInstance(y_test, np.ndarray)
        self.assertEqual(x_train.shape, (1, block_size, 2))
        self.assertEqual(x_train_region.shape, (1, 2))
        self.assertEqual(y_train.shape, (1, 1))
        self.assertEqual(x_val.shape, (1, block_size, 2))
        self.assertEqual(x_val_region.shape, (1, 2))
        self.assertEqual(y_val.shape, (1, 1))
        self.assertEqual(x_test.shape, (1, block_size, 2))
        self.assertEqual(x_test_region.shape, (1, 2))
        self.assertEqual(y_test.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
