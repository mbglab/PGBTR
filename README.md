# PGBTR
PGBTR：A powerful and general method for inferring bacterial transcriptional regulatory networks

### PGBTR ###

>Requirement

- Pytorch 1.9.1
- Cuda 11.1 (You can choose the appropriate version of your own GPU)
- scikit-learn 0.24.2
- joblib 1.3.2
- numpy 1.24.4
- pandas  1.4.0
- tqdm 4.66.1

>Usage

#Step1. Convert gene expression data to PDGD matrix
Command lines : python Preprocessing.py
Note : You can use other data instead of the test data, and choose the appropriate number of parallel processes for your own computer.

#Step2. Training and predicting
Command lines : python PGBTR.py

>Evaluation
#The Preprocessing_evaluation.py and PGBTR_evaluation.py are used for evaluation of PGBTR in this study.
