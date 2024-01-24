import unittest
from unittest.mock import Mock, patch

import pandas as pd
from eq_prediction.get_data import (
    get_earthquake_count,
    get_earthquake_data,
    make_datarange,
    make_df,
    make_params,
)


class TestGetData(unittest.TestCase):
    def test_make_params(self):
        params = make_params(
            "2023-01-01T00:00:00", "2023-12-31T23:59:59", 35.0, 45.0, -120.0, -110.0
        )
        expected_params = {
            "format": "geojson",
            "starttime": "2023-01-01T00:00:00",
            "endtime": "2023-12-31T23:59:59",
            "minlatitude": 35.0,
            "maxlatitude": 45.0,
            "minlongitude": -120.0,
            "maxlongitude": -110.0,
        }
        self.assertEqual(params, expected_params)

    def test_make_datarange(self):
        datarange = make_datarange(
            "2023-01-01T00:00:00", "2023-01-31T23:59:59", "100-01-01"
        )
        expected_datarange = [
            "100-01-01",
            "2023-01-01",
            "2023-01-08",
            "2023-01-15",
            "2023-01-22",
            "2023-01-29",
        ]
        self.assertEqual(datarange, expected_datarange)

    def test_get_earthquake_count(self):
        params = {
            "format": "geojson",
            "starttime": "2023-01-01T00:00:00",
            "endtime": "2023-12-31T23:59:59",
            "minlatitude": 35.0,
            "maxlatitude": 45.0,
            "minlongitude": -120.0,
            "maxlongitude": -110.0,
        }
        count = get_earthquake_count(params)
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)

    @patch("requests.get")
    def test_get_earthquake_data(self, mock_get):
        # Mock the response from the API
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Define test parameters
        test_params = {
            "format": "geojson",
            "starttime": "2023-01-01T00:00:00",
            "endtime": "2023-12-31T23:59:59",
            "minlatitude": 35.0,
            "maxlatitude": 45,
            "minlongitude": "-120.0",
            "maxlongitude": -110,
        }

        result = get_earthquake_data(test_params)
        mock_get.assert_called_with(
            "https://earthquake.usgs.gov/fdsnws/event/1/query", params=test_params
        )
        self.assertEqual(result, mock_response)

    @patch("requests.get")
    def test_make_df(self, mock_get):
        # Mock the response from the API
        mock_response = Mock()
        mock_response.json.return_value = {
            "features": [
                {
                    "properties": {"prop1": "value1", "prop2": "value2"},
                    "geometry": {"coordinates": [1.0, 2.0, 3.0]},
                },
            ]
        }
        mock_get.return_value = mock_response

        test_params = {
            "format": "geojson",
            "starttime": "2023-01-01T00:00:00",
            "endtime": "2023-12-31T23:59:59",
            "minlatitude": 35.0,
            "maxlatitude": 45,
            "minlongitude": "-120.0",
            "maxlongitude": -110,
        }

        result_df, result_errors = make_df(mock_response, test_params, [])

        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(result_errors, [])


if __name__ == "__main__":
    unittest.main()
