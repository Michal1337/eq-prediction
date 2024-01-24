# eq_prediction
Repository for earthquake prediction project.

## PIPELINE:
1. eq_prediction/get_data.py - downloads data (~2h, many requests)
2. eq_prediction/add_features.py - filters data and adds features and labels (~5h)
3. eq_prediction/make_npys.py - makes .npy files for each region (~30min)
4. eq_prediction/make_datasets.py - transforms .npy files into tf.Dataset (~50min)
5. eq_prediction/merge_datasets.py - merges datasets into one (~30min)
6. notebooks/train.py - trains the model (~2h - ∞)

Final files are in the eq_prediction folder.