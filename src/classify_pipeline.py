'''
Script for Assignment 2, Visual Analytics, Cultural Data Science, F2023
This script comprises several functions which make up a pipeline for fitting and evaluating a scikit-learn classifier. 

To run the entire pipeline, import the function clf_pipeline into your script ! Alternatively, you can pick and choose between the functions below !  

@MinaAlmasi
'''

# system tools
import pathlib

# custom utils
import sys
sys.path.append(str(pathlib.Path(__file__).parents[1]))
from utils.custom_logging import custom_logger 

# machine learning
from sklearn import metrics
from joblib import dump


## functions ##
def clf_evaluate(classifier, X_test_feats, y_test, labels):
    '''
    Evaluates fitted classifier on test data, returns classification report.

    Args: 
        - classifier: classifier already fitted on training data
        - X_test_feats: test data array
        - y_test: test data labels
    Returns: 
        - clf_metrics: classification report containing information such as accuracy, F1, precision and recall 
    '''

    # make predictions
    y_pred = classifier.predict(X_test_feats)

    # evaluate predictions
    clf_metrics = metrics.classification_report(y_test, y_pred, target_names=labels)

    return clf_metrics


def clf_get_name(classifier): 
    '''
    Retrieve name of an instantiated classifier. Useable for logging and saving models or classification metrics. 

    Args: 
        - classifier: instantiated classifier
    
    Returns: 
        - classifier_name: name of instantiated classifier (abbreviation for Logistic Regression and MLPClassifier, full names for other sklearn models)
    '''

    if classifier.__class__.__name__ == "LogisticRegression":
        classifier_name = "LR"
    elif classifier.__class__.__name__ == "MLPClassifier":
        classifier_name = "MLP"
    else: 
        classifier_name = classifier.__class__.__name__
    
    return classifier_name


def clf_metrics_to_txt(txt_name:str, output_dir:pathlib.Path(), clf_metrics, params):
    '''
    Converts scikit-learn's classification report (metrics.classification_report) to a .txt file. 

    Args:
        - txtname: filename for .txt report
        - output_dir: directory where the text file should be stored. 
        - clf_metrics: metrics report (sklearn.metrics.classification_report() or returned from clf_evaluate)
        - params: classifier params. (Returned from .get_params())
    
    Outputs: 
        - .txt file in specified output_dir
    '''

    # define filename 
    filepath = output_dir / txt_name

    # write clf metrics 
    with open(f'{filepath}.txt', "w") as file: 
        file.write(f"Results from model with parameters {params} \n {clf_metrics}")


def clf_pipeline(classifier, X_train_feats, y_train, X_test_feats, y_test, labels, output_dir:pathlib.Path(),
                 save_model:bool=False, model_dir=None):
    '''
    Classifier pipeline which does model fitting and model evaluation of an instantiated classifier in the scikit-learn framework. 
    Saves model metrics to as a .txt file in a specified directory (output_dir) with an option to also save the model.

    Args: 
        - classifier: instantiated classifier (e.g., LogisticRegression() or MLPClassifier())
        - X_train_feats, y_train, X_test_feats, y_test: data arrays for model fitting and evaluation (numpy array)
        - output_dir: directory where classifier metrics should be saved after evaluation
        - save_model: whether the model should be saved. Defaults to false
        - model_dir: directory where model is saved if save_model = True. Defaults to None
        
    Returns: 
        - classifier: fitted classifier

    Output:
        - model metrics as .txt file 
        - model as .joblib (if save_model = True)
    '''
    
    # import logger, get classifier name (for logging and txtfile)
    logging = custom_logger("pipeline") 
    classifier_name = clf_get_name(classifier)   

    # fit classifier 
    logging.info(f"Fitting {classifier_name}")
    classifier = classifier.fit(X_train_feats, y_train.ravel())
    
    # evaluate classifier 
    logging.info(f"Evaluating ... ")
    clf_metrics = clf_evaluate(classifier, X_test_feats, y_test, labels)

    # write metrics report
    clf_metrics_to_txt(f"{classifier_name}_metrics", output_dir, clf_metrics, classifier.get_params())
    
    if save_model == True:
        logging.info(f"Saving ... ")
        model_path = model_dir / f"{classifier_name}_classifier"
        dump(classifier, model_path)
    
    return classifier 