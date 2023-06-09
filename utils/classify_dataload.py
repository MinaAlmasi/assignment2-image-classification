'''
Utils script for Assignment 2, Visual Analytics, Cultural Data Science, F2023

The following script contains functions load_npz_vec_data to aid the classification pipeline (classify_pipeline.py) in loading data 

@MinaAlmasi
'''
import os
import numpy as np

def load_npz_vec_data(input_dir, file_name):
    '''
    Load npz data arrays for model training and fitting that has been vectorized by vectorize.py
    Args
        - input_dir: directory where .npz is located
        - filename: name of the -npz file (e.g., "tfid_500f_data.npz")
    Returns: 
        - X_train_feats, y_train, X_test_feats, y_test, labels: vectorized data arrays for model fitting and evaluation
    '''

    # define path and load data 
    file_path = os.path.join(input_dir, file_name)
    data = np.load(file_path, allow_pickle=True)

    # retrieve arrays
    X_train_feats = data["X_train_feats"]
    X_test_feats = data["X_test_feats"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    labels = data["labels"]

    return X_train_feats, X_test_feats, y_train, y_test, labels