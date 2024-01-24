import os
import shutil
import tempfile
import unittest

import numpy as np
import tensorflow as tf
from eq_prediction import save_datasets


class TestSaveDatasets(unittest.TestCase):
    def setUp(self):
        self.input_directory = tempfile.mkdtemp()
        self.output_directory = tempfile.mkdtemp()

        def create_temp_npy_files(directory, prefix, num_files, shape):
            for i in range(num_files):
                np.save(
                    os.path.join(directory, f"{prefix}_{i}.npy"), np.random.rand(*shape)
                )

        create_temp_npy_files(self.input_directory, "x_train", 3, (5, 10))
        create_temp_npy_files(self.input_directory, "x_train_region_", 3, (5, 10))
        create_temp_npy_files(self.input_directory, "y_train", 3, (5, 10))

    def tearDown(self):
        shutil.rmtree(self.input_directory)
        shutil.rmtree(self.output_directory)

    def test_save_datasets(self):
        try:
            save_datasets(self.input_directory, self.output_directory)

            assert os.path.exists(self.output_directory)
            for split in ["train", "val", "test"]:
                for idx in range(3):
                    assert os.path.exists(
                        os.path.join(
                            self.output_directory,
                            f"{split}_{idx}",
                            "data-00000-of-00001",
                        )
                    )

        except Exception as e:
            self.fail(f"Test failed: {e}")


if __name__ == "__main__":
    unittest.main()
