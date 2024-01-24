"""
Script used to concatenate and save all the datasets into 3 datasets: train, val, and test.
"""
import os

import numpy as np
import tensorflow as tf
import tqdm

from .params import OUTPUT_DIRECTORY


def merge_datasets(output_directory: str, split: str) -> tf.data.Dataset:
    """
    Concatenates and returns the datasets for a given split.

    Parameters:
    - split (str): The split of the data. Should be one of ['train', 'val', 'test'].

    Returns:
    tf.data.Dataset: A tf.data.Dataset containing data from all regions.
    """
    dataset = None
    num_regs = len(os.listdir(output_directory)) // 3
    for idx in tqdm.trange(num_regs):
        ds = tf.data.Dataset.load(os.path.join(output_directory, f"{split}_{idx}"))
        if dataset is None:
            dataset = ds
        else:
            dataset = dataset.concatenate(ds)
    return dataset

def main():
    """
    Main function for the script. Saves the datasets to ../data/ds_{split}.
    """
    for split in ["train", "val", "test"]:
        dataset = merge_datasets(OUTPUT_DIRECTORY, split)
        dataset.save(os.path.join(f"../data/ds_{split}"))
        print(f"Saved {split} dataset to ../data/ds_{split}")


if __name__ == "__main__":
    main()
