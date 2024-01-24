import unittest

import numpy as np
import pandas as pd
from eq_prediction import (
    add_region_info,
    add_tectonic_info,
    find_min_dist,
    haversine_distance,
    initial_preprocess,
    make_mapping,
)


class TestYourFunctions(unittest.TestCase):
    def test_haversine_distance(self):
        self.assertTrue(
            np.isclose(haversine_distance(0, 0, 0, 1), 111.195079734, rtol=1e-8)
        )
        self.assertTrue(
            np.allclose(
                haversine_distance(0, 0, [1, 1], [0, 0]),
                [111.195079734, 111.195079734],
                rtol=1e-8,
            )
        )

    def test_find_min_dist(self):
        df_tp = pd.DataFrame({"lat": [0, 1], "lon": [0, 0], "plate": [1, 2]})
        self.assertEqual(find_min_dist(df_tp, 0, 0), (0.0, "1"))

    def test_add_region_info(self):
        df = pd.DataFrame({"latitude": [0, 1], "longitude": [0, 0]})
        df_tp = pd.DataFrame({"lat": [0, 1], "lon": [0, 0], "plate": [1, 2]})
        result_df = add_region_info(df, df_tp, 1)
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertTrue("lat_cent" in result_df.columns)
        self.assertTrue("lon_cent" in result_df.columns)
        self.assertTrue("plate_region" in result_df.columns)
        self.assertTrue("dist_region" in result_df.columns)

    def test_add_tectonic_info(self):
        df = pd.DataFrame({"latitude": [0, 1], "longitude": [0, 0]})
        df_tp = pd.DataFrame({"lat": [0, 1], "lon": [0, 0], "plate": [1, 2]})
        result_df = add_tectonic_info(df, df_tp)
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertTrue("plate" in result_df.columns)
        self.assertTrue("dist" in result_df.columns)

    def test_initial_preprocess(self):
        df = pd.DataFrame(
            {
                "time": [
                    "1973-01-01T00:00:00",
                    "1973-01-01T00:00:00",
                    "1973-01-01T00:00:00",
                ],
                "longitude": [0, 0, 0],
                "latitude": [0, 0, 0],
            }
        )
        result_df = initial_preprocess(df, "1973-01-01T00:00:00", 1)
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertTrue("time_disc" in result_df.columns)
        self.assertTrue("longitude_disc" in result_df.columns)
        self.assertTrue("latitude_disc" in result_df.columns)
        self.assertTrue("pos" in result_df.columns)

    def test_make_mapping(self):
        df = pd.DataFrame(
            {
                "time": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
                "category": ["A", "B", "A", "C"],
            }
        )

        split_date_train = "2023-01-02"

        result_df, type2id = make_mapping(df, "category", 2, split_date_train)

        expected_mapping = {"A": 1, "B": 2, "C": 3}
        self.assertDictEqual(type2id, expected_mapping)

        expected_result_df = pd.DataFrame(
            {
                "time": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
                "category": [1, 2, 1, 3],
            }
        )
        pd.testing.assert_frame_equal(result_df, expected_result_df)


if __name__ == "__main__":
    unittest.main()
