#!/bin/bash

# activate virtual environment (works only if setup.sh has been run only !)
source ./env/bin/activate

# preprocess data 
echo -e "\n [INFO:] Preprocessing CIFAR10 data ..." # user msg 
python src/preprocess_data.py

# run logistic regression
echo -e "\n [INFO:] Running classification pipeline with LR ..." # user msg 
python src/classify_LR.py

# run MLP
echo -e "\n [INFO:] Running classification pipeline with MLP ..." # user msg 
python src/classify_MLP.py

# deactivate virtual environment
deactivate

# celebratory user msg ! 
echo -e "\n [INFO:] Classifications complete!"