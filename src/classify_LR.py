'''
Script for Assignment 2, Visual Analytics, Cultural Data Science, F2023
This script is made to train a logistic regression (LogisticRegression()) on a preprocessed version (following src/preprocess_data.py) of the CIFAR10 dataset. 

Run the script by typing in the command line: 
    python src/classify_LR.py

@MinaAlmasi
'''

# system tools
from pathlib import Path
import time

# custom utils 
import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.classify_dataload import load_npz_vec_data
from utils.custom_logging import custom_logger 

# machine learning
from classify_pipeline import clf_pipeline
from sklearn.linear_model import LogisticRegression

def main():
    #initialise timer, logger 
    start_time = time.time()
    logging = custom_logger("LR_logger")

    # define paths
    path = Path(__file__) # path to current file
    input_dir = path.parents[1] / "in" 
    datafile = "preprocessed_cifar.npz"

    output_dir = path.parents[1] / "out"
    model_path = path.parents[1] / "models"

    # load data
    logging.info("Loading data ...")
    X_train_feats, X_test_feats, y_train, y_test, labels = load_npz_vec_data(input_dir, datafile)

    # initialize classifier 
    logging.info("Initializing classifier ...")
    classifier = LogisticRegression(random_state=129, multi_class="multinomial", solver="saga", tol=0.1, max_iter=1000) 
        

    # run classifier pipeline (steps -> 1: fitting, 2: model eval, 3: save results)
    classifier = clf_pipeline(classifier = classifier, 
                 X_train_feats = X_train_feats, 
                 X_test_feats = X_test_feats, 
                 y_train = y_train, 
                 y_test = y_test,
                 labels = labels,
                 output_dir = output_dir,
                 save_model = True,
                 model_dir = model_path,
                 )

    # print elapsed time
    elapsed = round(time.time() - start_time, 2)
    logging.info(f"Classification finished. Classifier metrics saved to 'out' directory. \n Time elapsed: {elapsed} seconds.")

# run classifier
if __name__ == "__main__":
    main()