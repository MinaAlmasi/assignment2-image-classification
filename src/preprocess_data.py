'''
Script for Assignment 2, Visual Analytics, Cultural Data Science, F2023
This script is made to load and preprocess CIFAR10 image data. Returns a .npz file in the "in" folder. 

Run the script by typing in the command line: 
    python src/preprocess_data.py

@MinaAlmasi
'''

# system tools 
import pathlib

# custom logger
import sys
sys.path.append(str(pathlib.Path(__file__).parents[1]))
from utils.custom_logging import custom_logger 

# data loader
from tensorflow.keras.datasets import cifar10

# image wrangling 
import numpy as np
import cv2

def load_cifar():
    '''
    Load CIFAR10 data with defined labels.     

    Returns: 
    - (X_train, y_train): tuple containing training data (numpy arrays)
    - (X_test, y_test): tuple containing test data (numpy arrays)
    - labels: labels for classes in data
    '''

    # load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # make labels explicit 
    labels = ["airplane", 
          "automobile", 
          "bird", 
          "cat", 
          "deer", 
          "dog", 
          "frog", 
          "horse", 
          "ship", 
          "truck"]
    
    return (X_train, y_train), (X_test, y_test), labels


def data_preprocess(X_train, X_test):
    '''
    Preprocess image data prior to image classification. Converts image arrays to greyscale, scales and reshapes the arrays.

    Args:
        - X_train: train data array (numpy array)
        - X_test: test data array (numpy array)

    Returns
        - X_train_feats: preprocessed train data (numpy array)
        - X_test_feats: preprocessed test data (numpy array)
    '''

    # greyscale 
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

    # scale (divide by max value)
    X_train_scaled = (X_train_grey)/255.0
    X_test_scaled = (X_test_grey)/255.0

    # reshape 
    nsamples, nx, ny = X_train_scaled.shape
    X_train_feats = X_train_scaled.reshape((nsamples,nx*ny)) # flattening all images  

    nsamples, nx, ny = X_test_scaled.shape
    X_test_feats = X_test_scaled.reshape((nsamples,nx*ny))

    return X_train_feats, X_test_feats


def main():
    # initialize logger
    logging = custom_logger("preprocess_data")

    # define paths 
    path = pathlib.Path(__file__) # path to current file
    save_dir = path.parents[1] / "in"

    # ensure save dir is made
    save_dir.mkdir(exist_ok=True, parents=True)

    # load dataset
    logging.info("Loading raw data ...")
    (X_train, y_train), (X_test, y_test), labels = load_cifar()

    # preprocess data 
    logging.info("Preprocessing raw data ...")
    X_train_feats, X_test_feats = data_preprocess(X_train, X_test)
    
    # save data 
    filepath = save_dir /  "preprocessed_cifar.npz"
    np.savez_compressed(filepath, X_train_feats = X_train_feats, X_test_feats= X_test_feats, y_train = y_train, y_test = y_test, labels=labels)
    
    logging.info(f"Preprocessed data saved! ...")

if __name__ == "__main__":
    main()