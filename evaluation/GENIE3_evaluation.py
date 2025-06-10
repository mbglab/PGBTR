#!user/bin/env python3

# You should install all the dependency package before running this code.


import os
import sys
import pandas as pd
import numpy as np
from arboreto.algo import grnboost2, genie3
from arboreto.utils import load_tf_names
from distributed import Client, LocalCluster
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# You can change the path here to evaluate other datasets.
inFile='../data/Network3_Ecoli/net3_expression_data.tsv'
inDF = pd.read_csv(inFile, sep = '\t', index_col = 0, header = 0)
client = Client(processes = False)

for i in range(1,11):
    network = genie3(inDF.to_numpy(), client_or_address = client, gene_names = inDF.columns)
    network.to_csv(f'infer_GENIE3_net{i}_expression_data.tsv', index = False, sep = '\t')
