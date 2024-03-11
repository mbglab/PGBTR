#!user/bin/env python3

# Convert raw gene expression data into Probability distribution and graph distances(PDGD) matrix.
# Usage: python Preprocessing_evaluation.py --dataset-name Dream5_Ecoli --bins 32

import argparse
import os
import pandas as pd
import numpy as np

from numpy import *
import random
from tqdm import tqdm
from sklearn.cluster import KMeans
from joblib import Parallel, delayed


parser = argparse.ArgumentParser(description='Some parameter selections for data preprocessing')
parser.add_argument('--dataset-name', default='Dream5_Ecoli', help='The dataset you want to process.')
parser.add_argument('--bins', type=int, default=32, help='Size of matrixs.')

args = parser.parse_args()

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
    TF_list = sorted(list(set(gold_standard.iloc[:,0])))
    TG_list = list(ExpressionMatrix.index)
    total_iterations = len(TF_list) * len(TG_list)
    pbar = tqdm(total=total_iterations)
    results = Parallel(n_jobs=64, backend='threading')(delayed(calculate_distances)(ExpressionMatrix_clustered, tf, tg, clusters, pbar) for tf in TF_list for tg in TG_list)
    
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

# NEPDF
def generate_NEPDF(network, ExpressionMatrix, save_dir, bins, num):
    """
    Convert raw gene expression data into normalized empirical probability distribution function (NEPDF)  matrix.
    """
    x = []
    y = []
    z = []

    gene_pair_label = network.copy()
    # shuffle randomly
    gene_pair_label = gene_pair_label.sample(frac=1, random_state=20230729)

    for j in range(gene_pair_label.shape[0]):
        x_gene_id, y_gene_id, label = gene_pair_label.iloc[j,0], gene_pair_label.iloc[j,1], gene_pair_label.iloc[j,2]

        y.append(label)
        z.append(x_gene_id + '\t' + y_gene_id + '\t' + str(label))
        # convert expression data matrix to histogram2d
        x_tf = ExpressionMatrix.loc[x_gene_id,]
        x_gene = ExpressionMatrix.loc[y_gene_id,]
        H_T = np.histogram2d(x_tf, x_gene, bins=bins)
        H = H_T[0]
        HT = H
        x.append(HT)

    if (len(x)>0):
        xx = array(x)[:, :, :,newaxis]
    else:
        xx = array(x)
    # save results
    save(save_dir+'/xdata' + str(num) + '.npy', xx)
    save(save_dir+'/ydata' + str(num) + '.npy', array(y))
    save(save_dir+'/zdata' + str(num) + '.npy', array(z))


bins = args.bins
dataset_name = args.dataset_name

if bins == 16:
    clusters = 24

elif bins == 32:
    clusters = 50

elif bins == 8:
    clusters = 12


if dataset_name == 'Dream5_Ecoli':
    data_dir = 'data/Dream5/Network3_Ecoli/'
    ExpressionMatrix = pd.read_csv(data_dir + 'net3_expression_data.tsv', sep = '\t').T
    ExpressionRank = ExpressionMatrix.rank(axis=0, method='min', ascending=False)
    
    gold_standard = pd.read_csv(data_dir + 'net3_GoldStandard.tsv', sep = '\t', header = None)
    gold_standard.columns = ['TF_id', 'TG_id', 'Index']
    
    GDmatrix = graph_distances(ExpressionMatrix, gold_standard, clusters)
    GDmatrix.to_csv(data_dir + f'GDmatrix_bin{bins}.csv', index=False)    
    GDmatrix.set_index(['TF_id', 'TG_id'], inplace = True)
    
    # random negative sampling
    for iter_num in range(1, 11):
        random.seed(iter_num)
        final_net = gold_standard[gold_standard.Index == 1]
        
        for tf in set(final_net.TF_id):
            link_num = final_net[final_net.TF_id == tf].shape[0]
            neg_net = gold_standard[gold_standard.Index == 0]
            neg_net = neg_net[neg_net.TF_id == tf]
            final_net = pd.concat([final_net, neg_net.sample(link_num)], ignore_index=True)
        
        save_dir = data_dir + f'PDGD/{iter_num}'        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        generate_PDGD(final_net, ExpressionMatrix, ExpressionRank, GDmatrix, save_dir, bins, iter_num)

        
elif dataset_name == 'Dream5_Silico':
    data_dir = 'data/Dream5/Network1_Silico/'
    ExpressionMatrix = pd.read_csv(data_dir + 'net1_expression_data.tsv', sep = '\t').T
    ExpressionRank = ExpressionMatrix.rank(axis=0, method='min', ascending=False)
    
    gold_standard = pd.read_csv(data_dir + 'net1_GoldStandard.tsv', sep = '\t', header = None)
    gold_standard.columns = ['TF_id', 'TG_id', 'Index']
    
    GDmatrix = graph_distances(ExpressionMatrix, gold_standard, clusters)        
    GDmatrix.to_csv(data_dir + f'GDmatrix_bin{bins}.csv', index=False)
    GDmatrix.set_index(['TF_id', 'TG_id'], inplace = True)
    
    # random negative sampling
    for iter_num in range(1, 11):
        random.seed(iter_num)
        final_net = gold_standard[gold_standard.Index == 1]
        
        for tf in set(final_net.TF_id):
            link_num = final_net[final_net.TF_id == tf].shape[0]
            neg_net = gold_standard[gold_standard.Index == 0]
            neg_net = neg_net[neg_net.TF_id == tf]
            final_net = pd.concat([final_net, neg_net.sample(link_num)], ignore_index=True)
        
        save_dir = data_dir + f'PDGD/{iter_num}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        generate_PDGD(final_net, ExpressionMatrix, ExpressionRank, GDmatrix, save_dir, bins, iter_num)
        

elif dataset_name == 'RegulonDB_Ecoli':
    data_dir = 'data/RegulonDB_Ecoli/'
    ExpressionMatrix = pd.read_csv(data_dir + 'ExpressionMatrix.csv', index_col = 0)
    ExpressionRank = ExpressionMatrix.rank(axis=0, method='min', ascending=False)
    
    gold_standard = pd.read_csv(data_dir + 'gold_standard.csv')
    GDmatrix = graph_distances(ExpressionMatrix, gold_standard, clusters)
    GDmatrix.to_csv(data_dir + f'GDmatrix_bin{bins}.csv', index=False)
    #GDmatrix = pd.read_csv(data_dir + f'GDmatrix_bin{bins}.csv')
    GDmatrix.set_index(['TF_id', 'TG_id'], inplace = True)
    TRN_imodulon = pd.read_csv(data_dir + 'TRN_imodulon.csv')
    allgenes_info = pd.read_csv(data_dir + 'allgenes_info.csv', index_col=0)
    
    # random negative sampling
    all_genes = [x for x in list(ExpressionMatrix.index) if x in list(allgenes_info.index)]
    for iter_num in range(1, 11):
        random.seed(iter_num+42)
        final_net = gold_standard[gold_standard.Index == 1]

        for tf in set(final_net.TF_id):
            tmp1 = TRN_imodulon[TRN_imodulon.regulator == tf]
            link_num = final_net[final_net.TF_id == tf].shape[0]
            pos_list = list(final_net[final_net.TF_id == tf]['TG_id'])
            # Negative sampling
            neg_list = [x for x in all_genes if x not in pos_list and  x not in list(tmp1.gene_id) and  x != tf]
            neg_net = pd.DataFrame({'TF_id': [tf]*link_num, 'TG_id': random.sample(neg_list, link_num), 'Index':[0]*link_num})
            final_net = pd.concat([final_net, neg_net], ignore_index=True)
        
        save_dir = data_dir + f'PDGD/{iter_num}'
   
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        generate_PDGD(final_net, ExpressionMatrix, ExpressionRank, GDmatrix, save_dir, bins, iter_num)
        #generate_NEPDF(final_net, ExpressionMatrix, save_dir_NEPDF, bins, iter_num)

        
elif dataset_name == 'Subtiwiki_Bsubtilis':
    data_dir = 'data/Subtiwiki_Bsubtilis/'
    ExpressionMatrix = pd.read_csv(data_dir + 'ExpressionMatrix.csv', index_col = 0)
    ExpressionRank = ExpressionMatrix.rank(axis=0, method='min', ascending=False)    
    
    gold_standard = pd.read_csv(data_dir + 'gold_standard.csv')
    GDmatrix = graph_distances(ExpressionMatrix, gold_standard, clusters)
    GDmatrix.to_csv(data_dir + f'GDmatrix_bin{bins}.csv', index=False)
    #GDmatrix = pd.read_csv(data_dir + f'GDmatrix_bin{bins}.csv')
    GDmatrix.set_index(['TF_id', 'TG_id'], inplace = True)
    regulations = pd.read_csv(data_dir + 'regulations.csv')
    allgenes_info = pd.read_csv(data_dir + 'allgenes_info.csv', index_col=0)

    # random negative sampling
    for iter_num in range(1, 11):
        random.seed(iter_num+42)
        final_net = gold_standard[gold_standard.Index == 1]
        
        for tf in set(final_net.TF_id):
            tmp1 = regulations[regulations.regulator_locus == tf]
            link_num = final_net[final_net.TF_id == tf].shape[0]
            pos_list = list(final_net[final_net.TF_id == tf]['TG_id'])
            # Negative sampling
            neg_list = [x for x in list(ExpressionMatrix.index) if x not in pos_list and x not in list(tmp1.gene_locus) and  x != tf]
            neg_net = pd.DataFrame({'TF_id': [tf]*link_num, 'TG_id': random.sample(neg_list, link_num), 'Index':[0]*link_num})
            final_net = pd.concat([final_net, neg_net], ignore_index=True)
        
        save_dir = data_dir + f'PDGD/{iter_num}'
        #save_dir_NEPDF = data_dir + f'NEPDF/{bins}x{bins}/balanced/{iter_num}'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
                
        generate_PDGD(final_net, ExpressionMatrix, ExpressionRank, GDmatrix, save_dir, bins, iter_num)
        #generate_NEPDF(final_net, ExpressionMatrix, save_dir_NEPDF, bins, iter_num)
        
