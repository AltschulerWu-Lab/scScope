scScope
=======

scScope is a deep-learning based approach that can accurately and
rapidly identify cell-type composition and transcriptional state from
noisy single-cell gene-expression profiles containing dropout events and
scale to millions of cells.

   Citation: Massive single-cell RNA-seq analysis and imputation via
   deep learning. Yue Deng*, Feng Bao*, Qionghai Dai, Lani F. Wu#,
   Steven J. Altschuler#

This readme file is an introduction to the scScope software package,
implemented with Python and Tensorflow.

Packages Requirements
---------------------

-  Python >= 3.6
-  TensorFlow-GPU >= 1.4.1
-  Numpy >= 1.13.0
-  Scikit-learn >= 0.18.1
-  PhenoGraph >= 1.5.1

Install
-------

Step 1. Install TensorFlow-GPU or TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

General instructions can be found at https://www.tensorflow.org/install/
If you want to run scScope with GPUs: 1) Install GPU driver; 2) Install
CUDA toolkit; 3) Install cuDNN sdk; 4) Install the tensorflow-gpu
package: ``terminal   pip install tensorflow-gpu`` If you want to run
scScope on a traditional CPU: 1) Install the tensorflow package:
``terminal   pip install tensorflow`` ### Step 2. Install scScope

.. code:: terminal

   pip install scscope

or download the package and

.. code:: terminal

   python setup.py install

Usage
-----

Import the scScope package:
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In scScope directory, import scScope by

.. code:: python

   import scscope as DeepImpute

Train the scScope deep learning model:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

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

Make predictions:
~~~~~~~~~~~~~~~~~

.. code:: python

   latent_code, imputed_val = DeepImpute.prerdict(test_data, model, batch_effect=[])

Clustering analysis:
~~~~~~~~~~~~~~~~~~~~

1. On datasets with moderate scale. For dataset with less than 100,000
   cells, we recommend to use PhenoGraph
   (https://github.com/jacoblevine/PhenoGraph) for clustering with
   automatically determined cluster number.

.. code:: python

   import phenograph
   label, _,  _ = phenograph.cluster(latent_code)

where ``latent_code`` is the input feature.

2. On datasets with large scale. When datasets invlove more than 100,000
   cells, we designed a scalable clustering method to fast and accurate
   capture the subpopulation structure.

.. code:: python

   label = scalable_clustering.scalable_cluster(latent_code,
                                               kmeans_num=500,
                                               cluster_num=400,
                                               display_step=50,
                                               phenograh_neighbor=30
                                               )

Copyright
---------

Altschuler & Wu Lab 2018. Software provided as is under Apache License
2.0.