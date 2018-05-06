# scScope

scScope is a deep-learning based approach that can accurately and rapidly identify cell-type composition and transcriptional state from noisy single-cell gene-expression profiles containing dropout events and scale to millions of cells. 

> Citation: 
> Massive single-cell RNA-seq analysis and imputation via deep learning. Yue Deng\*, Feng Bao\*, Qionghai Dai, Lani F. Wu#, Steven J. Altschuler#


Here is an scScope implementation based Python and Tensorflow.

## Packages Requirements
- Python >= 3.6
- TensorFlow-GPU >= 1.4.1
- Numpy >= 1.13.0
- Scikit-learn >= 0.18.1
- PhenoGraph >= 1.5.1

## Usage

- <b>Import package</b>:
Switch to the scScope directory and import scScope by 

```python
import scScope as DeepImpute
```

- <b>Train the scScope deep learning model</b>:

```python
DI_model = DeepImpute.train(all_data, 
           					latent_dim, 
           					use_mask=True,
							batch_size=64,
							max_epoch=100,
							epoch_per_check=100,
							T=2，
							learning_rate=0.001,
							beta1=0.05,
							num_gpus=4)
```
where
```
    Parameters:

      all_data:         gene expression matrix in shape of n*m where n is the number of cells and m is the number of genes.
      latent_dim:       the feature dimension outputted by scScope.
      batch_size:       number of cells used in each training iteration.
      max_epoch:        maxial epoch used in training.
      epoch_per_check:  step of displying cuttent loss.
      T:                number of recurrent structures used in deep learning framework.
      use_mask:         if True, use only genes that have non-zero exprssion in at least one cell to calculate losses.
      learning_rate:    step length in gradient descending algorithm.
      beta1:            the beta1 parameter in AdamOptimizer
      num_gpus:         number fo gpus used for training in parallel.


    Output:

      DI_model: a dataframe of scScope outputs with keys:

        'latent_code_session':  	tensorflow session used in training.
        'test_input':           	todo
        'Imputated_output':     	imputed gene expressions.
        'latent_code':          	scScope features.
        'correcting_layer_output':  correcting layer learning by scScope.
```

- <b>Prediction</b>:

```python
latent_code, imputed_val = DeepImpute.prerdict(all_data, DI_model)
```
where
```
Input parameters:
	test_data:      gene expression matrix need to make prediction.
	model:          pre-trained scScope model.

Output:
	latent_code:    scScope features for inputted gene expressions.
	imputed_val:    gene expressions with imputations.
```

- <b>Clustering</b>:

1. On datasets with moderate scale.
For dataset with less than 100,000 cells, we recommend to use PhenoGraph (https://github.com/jacoblevine/PhenoGraph) for clustering with automatically determined cluster number.
```python
import phenograph
label, _,  _ = phenograph.cluster(latent_code)
```
where `latent_code` is the input feature.

2. On datasets with massive scale.
When datasets invlove more than 100,000 cells, we designed a scalable clustering method to fast and accurate capture the subpopulation structure. 
```python
label = scalable_clustering.scalable_cluster(latent_code,
                     						kmeans_num=500,
                     						cluster_num=400,
                     						display_step=50,
                     						phenograh_neighbor=30
                     						)
```
where
```
Parameters:

    latent_code:    n*m matrix of gene expression levels or representations of gene expression. n is cell size and m is gene or representation size.
    kmeans_num:     the number of independent K-means clusterings used. This is also the subset number.
    cluster_num:    cluster number for each K-means clustering. This is also the "n_clusters" in KMeans function in sklearn package.
    display_step:   displaying the process of K-means clustering.
    phenograh_neighbor: "k" parameter in PhenoGraph package.
```

## Demonstration

We provide a file (see `demon.py`) for the demonstration of applying scScope to a simulated data set with ground truth classes for feature learning and clustering. The demonstration was tested on a computer with Ubuntu 14.4, 4 Nvidia GeForce GTX Titan GPUs and CUDA Version 8.0.61. You can directly run the demonstration by 

> `python demon.py`

To run your own data with default parameters, you can simply replace the `gene_expression` with your expression data (matrix in n*m shape where n is the number of cells and m is the number of genes) and delete the evaluation part (`adjusted_rand_score`)

## Additional Information

- How to install Tensorflow-GPU？
Please refer to https://www.tensorflow.org/install/

## Copyright

Copyright Altschuler & Wu Lab 2018. All Rights Reserved. 


