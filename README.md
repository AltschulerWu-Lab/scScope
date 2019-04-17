# scScope

scScope is a deep-learning based approach that can accurately and rapidly identify cell-type composition and transcriptional state from noisy single-cell gene-expression profiles containing dropout events and scale to millions of cells. This work was published on *Nature Methods*, [full-text access](https://protect2.fireeye.com/url?k=d37bc159-8f3bf467-d37be644-0cc47ad9c120-011385b0eacba99e&u=https://rdcu.be/brEON)

> Citation: 
> Scalable analysis of cell-type composition from single-cell transcriptomics using deep recurrent learning. Yue Deng\*, Feng Bao\*, Qionghai Dai, Lani F. Wu#, Steven J. Altschuler#, Nature Methods, 2019, DOI: https://doi.org/10.1038/s41592-019-0353-7


This readme file is an introduction to the scScope software package, implemented with Python and Tensorflow. 

## Packages Requirements
- Python >= 3.6
- TensorFlow-GPU >= 1.4.1
- Numpy >= 1.13.0
- Scikit-learn >= 0.18.1
- PhenoGraph >= 1.5.1

## Install
### Step 1. Install TensorFlow-GPU or TensorFlow
  General instructions can be found at https://www.tensorflow.org/install/
  If you want to run scScope with GPUs: 
  1) Install GPU driver;
  2) Install CUDA toolkit;
  3) Install cuDNN sdk;
  4) Install the tensorflow-gpu package:
  ```terminal
  pip install tensorflow-gpu
  ```
   If you want to run scScope on a traditional CPU: 
    1) Install the tensorflow package:
  ```terminal
  pip install tensorflow
  ```
### Step 2. Install scScope
```terminal
pip install scScope
```
or download the package and 
```terminal
python setup.py install
```
Usually, it takes only several seconds to finish installation. If you want to run scScope on CPU, use "pip install scScope-cpu" to install scScope CPU version.

## Usage

### Import the scScope package:
In scScope directory, import scScope by 

```python
import scscope as DeepImpute
```

### Train the scScope deep learning model:

```python
DI_model = DeepImpute.train(
          train_data_set,
          latent_code_dim,
          use_mask=True,
          batch_size=64,
          max_epoch=100,
          epoch_per_check=100,
          T=2,
          exp_batch_idx_input=[],
          encoder_layers=[],
          decoder_layers=[],
          learning_rate=0.0001,
          beta1=0.05,
          num_gpus=1)
```
where
```
    Parameters:

      train_data_set:       gene expression matrix in shape of n * m where n is the number of cells and m is the number of genes.
      latent_code_dim:      the feature dimension outputted by scScope.
      batch_size:           number of cells used in each training iteration.
      max_epoch:            maximal epoch used in training.
      epoch_per_check:      step to display current loss.
      T:                    depth of recurrence used in deep learning framework.
      use_mask:             flag indicating whether to use only non-zero entries in calculating losses.
      learning_rate:        step length in gradient descending algorithm.
      beta1:                the beta1 parameter in AdamOptimizer.
      num_gpus:             number of gpus used for training in parallel.
      exp_batch_idx_input:  (optional) n * batch_num matrix in one-hot format, if provided, experimental batch ids are used for batch correction.
      encoder_layers:       the network structure for encoder layers of the autoencoder. for instance [64,128] means adding two layers with 64 and 128 nodes between the input and hidden features
      decoder_layers:       the network structure for decoder layers of the autoencoder. for instance [64,128] means adding two layers with 64 and 128 nodes between the hidden feature and the output layer



    Output:

      model: a dataframe of scScope outputs with keys:
            'latent_code_session':      tensorflow session used in training.
            'test_input':               tensorflow dataholder for test data.
            'test_exp_batch_idx':       tensorflow dataholder for experimental batch label.
            'imputated_output':         imputed gene expressions.
            'latent_code':              latent features by scScope.
            'removed_batch_effect':     correcting layer learning by scScope.

```
If run on CPU, remove the "num_gpus" parameter.
### Make predictions:

```python
latent_code, imputed_val = DeepImpute.prerdict(test_data, model, batch_effect=[])
```
where
```
    Parameter:
        test_data:      input gene expression matrix
        model:          pre-trained scScope model.

    Output:
        latent_fea:             scScope features output
        output_val:             gene expressions with imputations.
        predicted_batch_effect: batch effects inferenced by scScope, if experimental batches exist.

```

### Clustering analysis:

1. On datasets with moderate scale.
For dataset with less than 100,000 cells, we recommend to use PhenoGraph (https://github.com/jacoblevine/PhenoGraph) for clustering with automatically determined cluster number.
```python
import phenograph
label, _,  _ = phenograph.cluster(latent_code)
```
where `latent_code` is the input feature.

2. On datasets with large scale.
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

### Large-scale dataset beyond memory size:

In some cases dataset can be extremely large or the memory size in the computer/server is limited, it is impossible to load all cells at one time. scScope offers an option that allows flexible memory allocation. In this mode, large dataset is split into several small batches. In each training epoch, small batch files are sequentially loaded into memory for parameter updating.

To enable this mode, three steps are needed:

#### Step 1: split large data file into several small `*.npy` files

In this step, users need to build a new directory for small files at first. e.g.
```terminal
mkdir ./small_batches
```

In `./small_batches`, users can store the large expression file into several small files. Usually, it is not easy to split a large file due to memory limitations. However, the python package `pandas` offers a `chunk read` mode that enables sequential file reads. Here we provid sample code for how to split large files: 
```python
import pandas as pd
import numpy as np

cell_per_batch = 1000 # batch size

reader = pd.read_csv('gene_expression.csv', sep=',', header=0, index_col=0, chunksize=cell_per_batch)
file_count = 0
    for chunk in reader:
      np.save('./small_batches/batch_' + str(file_count) + '.npy', chunk.values)
      file_count += 1
```
Here, we assume there is a large file `gene_expression.csv`. We set the batch size (`cell_per_batch`) as 1000 and only load 1000 cells at a time from the large file. Batch data are saved in `batch_*.npy` file in directory `./small_batches`.

We note that users need to select proper parameters in `pd.read_csv`, see http://pandas.pydata.org/pandas-docs/stable/io.html.

After splitting files, the gene expression matrix should be in `cell X gene` format. Small files should be named as `batch_0.npy`, `batch_1.npy`, `batch_2.npy`, ... exactly. In each file, cell numbers and gene numbers should be consistent.


#### Step 2: running `scscope_large`
Batch file directory, file number, cell number and gene number should be provided in this mode for pre-allocation of GPU memory.

```python
from scscope.large_dataset import large_scale_processing as DeepImpute

train_data_path = './small_batches'
gene_num = 500
cell_num = 1000
file_num = 4
latent_dim = 50

DI_model = DeepImpute.train_large(train_data_path, 
                                  file_num, 
                                  cell_num, 
                                  gene_num,
                                  latent_dim)

DeepImpute.prerdict_large(train_data_path,
                          file_num,
                          DI_model)
```
After training and predicting, latent features and imputed sequences are stored under the same directory. For the data file `batch_i.npy`, the corresponding latent features and imputed gene expressions are stored in files `feature_i.npy` and `imputation_i.npy`, respectively, in the same directory.

## Demonstration

We provide a file (see `demo.py`) for the demonstration of applying scScope to a simulated data set with ground truth classes for feature learning and clustering. The demonstration was tested on a computer with Ubuntu 14.4, 4 Nvidia GeForce GTX Titan GPUs and CUDA Version 8.0.61. You can directly run the demonstration by 

> `python demo.py`

Here, the demonstration dataset is generated by Splatter (https://github.com/Oshlack/splatter-paper), where true cell identities are already known for evaluation. Usually, the scScope learning can be finished in seconds. But it will take 2 minute to visualize the results using tSNE.

To run your own data with default parameters, you can simply replace the `gene_expression` with your expression data (matrix in n*m shape where n is the number of cells and m is the number of genes) and delete the evaluation part (`adjusted_rand_score`)

## Other single-cell RNA-seq packages

There many software packages available for performing single cell sequence analysis. We encourage readers to investigate these other resources:

- MAGIC: https://github.com/KrishnaswamyLab/MAGIC
- ZINB-WaVE: https://github.com/drisso/zinbwave
- SIMLR: https://github.com/bowang87/SIMLR_PY
- DCA: https://github.com/theislab/dca
- scVI: https://github.com/YosefLab/scVI
- PhenoGraph: https://github.com/jacoblevine/PhenoGraph
- Seurat: https://satijalab.org/seurat/
- Splatter: https://github.com/Oshlack/splatter-paper

## Copyright

  Altschuler & Wu Lab 2018.
  Software provided as is under Apache License 2.0. 


