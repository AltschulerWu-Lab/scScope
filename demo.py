'''The goal of this demo is to show how to identify cell subpopulations based on latent
representations of gene expression learned by scScope.'''
import scscope as DeepImpute
import pandas as pd
import phenograph
import pickle
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# For this demo we normalize data using scanpy which is not a required package for scScope.
# To install, use: pip install scanpy
import scanpy.api as sc


def RUN_MAIN():

    # 1. Load gene expression matrix of simulated data
    # gene expression with simulated dropouts
    counts_drop = pd.read_csv('counts_1.csv', header=0, index_col=0)
    # ground trouth subpopulation assignment
    cellinfo = pd.read_csv('cellinfo_1.csv', header=0, index_col=0)

    group = cellinfo.Group
    label_ground_truth = []
    for g in group:
        g = int(g.split('Group')[1])
        label_ground_truth.append(g)

    # 2. Normalize gene expression based on scanpy (normalize each cell to have same library size)
    # matrix of cells x genes
    gene_expression = sc.AnnData(counts_drop.values)
    # normalize each cell to have same count number
    sc.pp.normalize_per_cell(gene_expression)
    # update datastructure to use normalized data
    gene_expression = gene_expression.X

    latent_dim = 50

    # 3. scScope learning
    if gene_expression.shape[0] >= 100000:
        DI_model = DeepImpute.train(
            gene_expression, latent_dim, T=2, batch_size=512, max_epoch=10, num_gpus=4)
    else:
        DI_model = DeepImpute.train(
            gene_expression, latent_dim, T=2, batch_size=64, max_epoch=300, num_gpus=4)

    # 4. latent representations and imputed expressions
    latent_code, imputed_val, _ = DeepImpute.predict(
        gene_expression, DI_model)

    # 5. graph clustering
    if latent_code.shape[0] <= 10000:
        label, _, _ = phenograph.cluster(latent_code)
    else:
        label = DeepImpute.scalable_cluster(latent_code)

    # evaluate
    ARI = adjusted_rand_score(label, label_ground_truth)
    print(ARI)

    X_embedded = TSNE(n_components=2).fit_transform(latent_code)

    # visualization of the subpopulation using tSNE
    plt.figure()
    for i in range(5):
        idx = np.nonzero(label == i)[0]
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1])
    plt.show()


if __name__ == '__main__':
    RUN_MAIN()
