# Image Classification with scikit-learn
This repository forms *assignment 2* by Mina Almasi (202005465) in the subject *Visual Analytics*, *Cultural Data Science*, F2023. The assignment description can be found [here](https://github.com/MinaAlmasi/assignment2-image-classification/blob/master/assignment-desc.md). 

The repository contains code for doing multiclass classification. Concretely, a logistic regression and a neural network  is trained and evaluated using ```scikit-learn```. See the [results](https://github.com/MinaAlmasi/assignment2-image-classification/tree/master#results) section for their final performance.

## Data 
The classifiers are trained and evaluated on the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) (Krizhevsky, 2009). The dataset comprises 60000 color images in the size 32x32 (50000 training and 10000 test images). The images are split in 10 classes (```airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck```) and there are 6000 images per class. 

## Reproducability 
To reproduce the results, follow the instructions in the [*Pipeline*](https://github.com/MinaAlmasi/assignment2-image-classification/tree/master#pipeline) section. 

**NB! Be aware that fitting models (esp. the MLPClassifier) can be computationally intensive and may take several minutes. Cloud computing (e.g., Ucloud) is encouraged**.

## Project Structure 
The repository is structured as such: 
```
├── README.md
├── assignment-desc.md
├── in                          <---    preprocessed data (.npz) is stored here after running preprocess_data.py
│   └── README.md                      
├── models                      <---    classifiers saved here 
│   ├── LR_classifier
│   └── MLP_classifier
├── out                         <---    classifier eval metrics saved here  
│   ├── LR_metrics.txt
│   └── MLP_metrics.txt
├── requirements.txt
├── run.sh                      <---    preprocess data & run classifier pipeline for both classifiers 
├── setup.sh                    <---    create venv and install reqs 
├── src
│   ├── classify_LR.py          <---    run logistic regression (LR)
│   ├── classify_MLP.py         <---    run neural network (MLP)
│   ├── classify_pipeline.py    <---    contains classification pipeline (functions)
│   └── preprocess_data.py      <---    preprocess data
└── utils
    ├── classify_dataload.py    <---    helper to load .npz data
    └── custom_logging.py       <---    custom logger to display user msg 
```

## Pipeline
The pipeline has been tested on Ubuntu ([UCloud](https://cloud.sdu.dk/)). Python's [venv](https://docs.python.org/3/library/venv.html) needs to be installed for the pipeline to work.

### Setup
Before running the classification, please run ```setup.sh``` in the terminal. 
```
bash setup.sh
```
The script installs the necessary packages and its dependencies in a newly created virtual environment (```env```). 

### Running the Classification
To reproduce the results of the repository, please type the following in the terminal: 
```
bash run.sh
```
This will preprocess the data, run the two classifiers, and save their metrics to ```out``` and classifiers to ```models```. 

## Results

### Logistic Regression
```
               precision    recall  f1-score   support

    airplane       0.33      0.40      0.36      1000
  automobile       0.39      0.36      0.37      1000
        bird       0.28      0.17      0.22      1000
         cat       0.23      0.17      0.19      1000
        deer       0.24      0.27      0.25      1000
         dog       0.32      0.31      0.31      1000
        frog       0.29      0.31      0.30      1000
       horse       0.31      0.31      0.31      1000
        ship       0.35      0.40      0.37      1000
       truck       0.39      0.47      0.42      1000

    accuracy                           0.32     10000
   macro avg       0.31      0.32      0.31     10000
weighted avg       0.31      0.32      0.31     10000
```
This model was run with the parameters: *random_state=129, multi_class="multinomial", solver="saga", tol=0.1, max_iter=1000*

### Neural Network (MLP)
```
               precision    recall  f1-score   support

    airplane       0.40      0.34      0.37      1000
  automobile       0.42      0.43      0.42      1000
        bird       0.29      0.23      0.26      1000
         cat       0.23      0.23      0.23      1000
        deer       0.28      0.25      0.26      1000
         dog       0.31      0.37      0.34      1000
        frog       0.31      0.36      0.33      1000
       horse       0.43      0.33      0.37      1000
        ship       0.40      0.50      0.44      1000
       truck       0.43      0.48      0.45      1000

    accuracy                           0.35     10000
   macro avg       0.35      0.35      0.35     10000
weighted avg       0.35      0.35      0.35     10000
```
This model was run with the parameters: *random_state=129, hidden_layer_sizes=(30,), activation="logistic", early_stopping=True, tol=0.2, max_iter=1000*

### Remark on the Results
Both models perform above chance level (10%), but the neural network is slightly better performing in overall macro avg (0.35 versus 0.32). Both classifiers are best at classes   ```ship``` and ```truck```. 

## Author
All code is made by Mina Almasi.
- github user: @MinaAlmasi
- student no: 202005465, AUID: au675000
- mail: mina.almasi@post.au.dk 

## References
Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images.