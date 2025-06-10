#!user/bin/env python3

# You should install PROTIA from https://github.com/AntoinePassemiers/PORTIA-Manuscript before running this code.

import portia as pr
import pandas as pd


# You can change the path to evaluate other dataset.
gold_standard = pd.read_csv('../data/Bsubtilis/gold_standard.csv')
tf_idx = list(gold_standard.TF_id)
expressionmatrix= pd.read_csv('../data/Bsubtilis/log_tpm.csv', index_col = 0)

dataset = pt.GeneExpressionDataset()
for i in range(expressionmatrix.shape[0]):
    dataset.add(pt.Experiment(list(expressionmatrix.index)[i], list(expressionmatrix.iloc[i,])))

M_bar = pt.run(dataset, tf_idx=tf_idx, method='fast')
