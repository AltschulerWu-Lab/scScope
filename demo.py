
import scScope as DeepImpute
import phenograph

import pickle
from sklearn.metrics.cluster import adjusted_rand_score


def RUN_MAIN():

    # load gene expression matrix here
    fid = open('data.pickle', 'rb')
    all_data = pickle.load(fid)
    gene_expression = all_data['Y']
    label_ground_truth = all_data['cluster_ids']
	
    print(gene_expression.shape)

    latent_dim = 50

    # scScope learning
    if gene_expression.shape[0] >= 100000:
        DI_model = DeepImpute.train(
            gene_expression, latent_dim, T=2, batch_size=512, max_epoch=10,num_gpus=4)
    else:
        DI_model = DeepImpute.train(
            gene_expression, latent_dim, T=2, batch_size=64, max_epoch=100,num_gpus=4)

    # latent representations and imputed expressions
    latent_code, imputed_val = DeepImpute.predict(
        gene_expression, DI_model)

    # graph clustering
    if latent_code.shape[0] <= 10000:
        label, _, _ = phenograph.cluster(latent_code)
    else:
        label = DeepImpute.scalable_cluster(latent_code)

    # evaluate the 
    ARI = adjusted_rand_score(label, label_ground_truth)
    print(ARI)


if __name__ == '__main__':
    RUN_MAIN()
