#!user/bin/env python3

# Convert raw gene expression data into Probability distribution and graph distances(PDGD) matrix.


import os
import pandas as pd
import numpy as np

from numpy import *
from tqdm import tqdm
from sklearn.cluster import KMeans
from joblib import Parallel, delayed


def calculate_distances(ExpressionMatrix_clustered, tf, tg, clusters, pbar):
    """
    Calculate the distance of the point of tf and tg expressions.
    """
    tf_pos = ExpressionMatrix_clustered.loc[tf, ].to_list()
    tg_pos = ExpressionMatrix_clustered.loc[tg, ].to_list()
    Dist_pos = []

    for i in range(clusters):
        for j in range(i + 1, clusters):
            Dist_pos.append(((tf_pos[i] - tf_pos[j]) ** 2 + (tg_pos[i] - tg_pos[j]) ** 2) ** 0.5)
    pbar.update(1)
    
    return [tf, tg] + Dist_pos

def graph_distances(ExpressionMatrix, gold_standard, clusters):
    """
    Calculate the Euclidean graph distance of tf and tg expressions.
    """
    # kmeans clustering    
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    kmeans.fit(ExpressionMatrix.T)
    ExpressionMatrix_clustered = pd.DataFrame(kmeans.cluster_centers_.T, index= ExpressionMatrix.index)
    # parallel computing
    gold_standard.set_index([0,1],inplace=True)
    total_iterations = gold_standard.shape[0]
    pbar = tqdm(total=total_iterations)
    results = Parallel(n_jobs=n_jobs, backend='threading')(delayed(calculate_distances)(ExpressionMatrix_clustered, tf, tg, clusters, pbar) for (tf,tg) in gold_standard.index)
    
    GDmatrix = pd.DataFrame(results, columns=['TF_id', 'TG_id'] + list(range(1, int(clusters*(clusters-1)*0.5)+1)))
    GDmatrix.fillna(0, inplace=True)
    #GDmatrix.to_csv(data_dir + 'GDmatrix_' + dataset_name + '.csv', index = False)
    
    return GDmatrix

def generate_PDGD(network, ExpressionMatrix, ExpressionRank, GDmatrix, save_dir, bins, num):
    """
    Convert raw gene expression data into Probability distribution and graph distance(PDGD) matrix.
    """
    x = []
    x_rank = []
    y = []
    z = []
    e = []
    gene_pair_label = network.copy()
    # shuffle randomly
    gene_pair_label = gene_pair_label.sample(frac=1, random_state=20230729)

    for j in range(gene_pair_label.shape[0]):
        x_gene_id, y_gene_id, label = gene_pair_label.iloc[j,0], gene_pair_label.iloc[j,1], gene_pair_label.iloc[j,2]

        y.append(label)
        z.append(x_gene_id + '\t' + y_gene_id + '\t' + str(label))
        E_list= list(GDmatrix.loc[(x_gene_id, y_gene_id),])
        # convert expression data matrix to histogram2d
        x_tf = ExpressionMatrix.loc[x_gene_id,]
        x_gene = ExpressionMatrix.loc[y_gene_id,]
        H_T = np.histogram2d(x_tf, x_gene, bins=bins)
        H = H_T[0]
        HT = H
        # convert expression data ranked matrix to histogram2d
        x_tf_rank = ExpressionRank.loc[x_gene_id,]
        x_gene_rank = ExpressionRank.loc[y_gene_id,]
        H_T_rank = np.histogram2d(x_tf_rank, x_gene_rank, bins=bins)
        H_rank = H_T_rank[0]
        HT_rank = H_rank

        # Euclidean distance matrix            
        E_matrix = np.zeros((bins,bins))
        
        n = 0
        for i in range(bins):
            for j in range(bins):
                E_matrix[j,i] = E_list[n]
                n += 1

        x.append(HT)
        x_rank.append(HT_rank)
        e.append(E_matrix*10)
    # concatenate three two-dimensional matrices into a three-dimensional matrix
    if (len(x)>0):
        xx = np.concatenate([array(x)[:, :, :,newaxis], array(x_rank)[:, :, :,newaxis], array(e)[:, :, :,newaxis]], axis=3)
    else:
        xx = np.array(x)
    # save results
    save(save_dir+'/xdata' + str(num) + '.npy', xx)
    save(save_dir+'/ydata' + str(num) + '.npy', array(y))
    save(save_dir+'/zdata' + str(num) + '.npy', array(z))

n_jobs = 64
bins = 32
data_dir = 'data/test_data/'

ExpressionMatrix = pd.read_csv(data_dir + 'ExpressionMatrix.csv', index_col = 0)
ExpressionRank = ExpressionMatrix.rank(axis=0, method='min', ascending=False)

train_data = pd.read_csv(data_dir +'train_data.txt', sep='\t', header=None)
test_data = pd.read_csv(data_dir +'test_data.txt', sep='\t', header=None)
gold_standard = pd.concat([train_data,test_data], ignore_index = True)
GDmatrix = graph_distances(ExpressionMatrix, gold_standard, 50)        
#GDmatrix.to_csv(data_dir + 'GDmatrix.csv', index=False)
#GDmatrix = pd.read_csv(data_dir + 'GDmatrix.csv')
GDmatrix.set_index(['TF_id', 'TG_id'], inplace = True)

allgenes_info = pd.read_csv(data_dir + 'allgenes_info.csv', index_col=0)

save_dir = data_dir + 'PDGD/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
generate_PDGD(train_data, ExpressionMatrix, ExpressionRank, GDmatrix, save_dir, bins, 'train')
generate_PDGD(test_data, ExpressionMatrix, ExpressionRank, GDmatrix, save_dir, bins, 'test')



        
