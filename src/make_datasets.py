"""
Script used to create and save tf.data.Datasets into INPUT_DIRECTORY from .npy files from OUTPUT_DIRECTORY.
In the laters stages the datasets will be concatenated into one dataset. Note that 3 .npy files are used for each dataset: x, x_region, and y.
"""
import os
from typing import Tuple

import numpy as np
import tensorflow as tf
import tqdm

from .params import INPUT_DIRECTORY, OUTPUT_DIRECTORY


def data_generator(
    input_directory: str, idx: int, split: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a generator that yields the x, x_region, and y data for a given split and index.
    Note that generator is used to save RAM.

    Parameters:
    - input_directory (str): The directory containing the .npy files.
    - idx (int): The index of the region.
    - split (str): The split of the data. Should be one of ['train', 'val', 'test'].

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the x, x_region, and y data.

    Example:
    >>> data_generator(INPUT_DIRECTORY, 0, 'train')
    """
    x = np.load(os.path.join(input_directory, f"x_{split}_{idx}.npy"))
    x_reg = np.load(os.path.join(input_directory, f"x_{split}_region_{idx}.npy"))
    y = np.load(os.path.join(input_directory, f"y_{split}_{idx}.npy"))
    for i in range(x.shape[0]):
        yield x[i], x_reg[i], y[i]


def make_dataset(
    input_directory: str,
    idx: int,
    split: str,
    shapes: Tuple[Tuple[int], Tuple[int], Tuple[int]],
) -> tf.data.Dataset:
    """
    Returns a tf.data.Dataset for a given split and index created from the data_generator.

    Parameters:
    - input_directory (str): The directory containing the .npy files.
    - idx (int): The index of the region.
    - split (str): The split of the data. Should be one of ['train', 'val', 'test'].

    Returns:
    tf.data.Dataset: A tf.data.Dataset containing the x, x_region, and y data.

    Example:
    >>> make_dataset(INPUT_DIRECTORY, 0, 'train')
    """
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(input_directory, idx, split),
        output_signature=(
            tf.TensorSpec(shape=shapes[0], dtype=tf.float32),
            tf.TensorSpec(shape=shapes[1], dtype=tf.float32),
            tf.TensorSpec(shape=shapes[2], dtype=tf.float32),
        ),
    )
    return dataset


def save_datasets(INPUT_DIRECTORY: str, OUTPUT_DIRECTORY: str) -> None:
    """
    Saves the datasets created from the .npy files from INPUT_DIRECTORY into OUTPUT_DIRECTORY.
    Note that shapes are the same for all regions.

    Parameters:
    - INPUT_DIRECTORY (str): The directory containing the .npy files.
    - OUTPUT_DIRECTORY (str): The directory to save the datasets.

    Returns:
    None
    """
    x_shape = np.load(os.path.join(INPUT_DIRECTORY, f"x_train_0.npy")).shape[1:]
    x_reg_shape = np.load(os.path.join(INPUT_DIRECTORY, f"x_train_region_0.npy")).shape[
        1:
    ]
    y_shape = np.load(os.path.join(INPUT_DIRECTORY, f"y_train_0.npy")).shape[1:]
    shapes = [x_shape, x_reg_shape, y_shape]
    num_regs = len(os.listdir(INPUT_DIRECTORY)) // 9
    for idx in tqdm.trange(num_regs):
        for split in ["train", "val", "test"]:
            dataset = make_dataset(INPUT_DIRECTORY, split, idx, shapes)
            dataset.save(os.path.join(OUTPUT_DIRECTORY, f"{split}_{idx}"))


def main():
    save_datasets(INPUT_DIRECTORY, OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()
