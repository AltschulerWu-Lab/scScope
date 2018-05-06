from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import time
import ops


def train(Train_data,
          latent_code_dim,
          use_mask=True,
          batch_size=64,
          max_epoch=100,
          epoch_per_check=100,
          T=2,
          learning_rate=0.001,
          beta1=0.05,
          num_gpus=4
          ):
    '''
    scScope training:

    scScope is a deep-learning based approach that can accurately and rapidly identify cell-type composition and transcriptional state from noisy single-cell gene-expression profiles containing dropout events and scale to millions of cells. 

    Parameters:

      Train_data:       gene expression matrix in shape of n*m where n is the number of cells and m is the number of genes.
      latent_code_dim:  the feature dimension outputted by scScope.
      batch_size:       number of cells used in each training iteration.
      max_epoch:        maxial epoch used in training.
      epoch_per_check:  step of displying cuttent loss.
      T:                number of recurrent structures used in deep learning framework.
      use_mask:         flag indicating whether only use non-zero entries in calculating losses.
      learning_rate:    step length in gradient descending algorithm.
      beta1:            the beta1 parameter in AdamOptimizer
      num_gpus:         number fo gpus used for training in parallel.


    Output:

      Model: a dataframe of scScope outputs with keys:
        'latent_code_session':  tensorflow session used in training.
        'test_input':           todo
        'Imputated_output':     imputed gene expressions.
        'latent_code':          scScope features.
        'correcting_layer_output':  ocorrecting layer learning by scScope.

    Copyright Altschuler & Wu Lab 2018. All Rights Reserved.
    '''

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        global_step = tf.get_variable(

            'global_step', [],

            initializer=tf.constant_initializer(0), trainable=False)

        train_data = tf.placeholder(
            tf.float32, [batch_size*num_gpus, np.shape(Train_data)[1]])

        # The mask matrix indicating which entries in the input are missing.
        val_mask = tf.placeholder(
            tf.float32, [batch_size*num_gpus, np.shape(Train_data)[1]])

        # Create an optimizer that performs gradient descent.
        opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1)

        # Calculate the gradients on models deployed on each GPU then summarized the gradients.
        tower_grads = []

        with tf.variable_scope(tf.get_variable_scope()):

            for i in range(num_gpus):

                print('Building Computational Graph on GPU-'+str(i))

                with tf.device('/gpu:%d' % (i+1)):

                    with tf.name_scope('%s_%d' % ('tower', i)) as scope:

                        itv = batch_size

                        if i == 0:

                            re_use_flag = False

                        else:

                            re_use_flag = True

                        loss = tower_loss(scope,
                                          train_data[(i) * itv:(i+1)*itv, :],
                                          val_mask[(i) * itv:(i+1)*itv, :],
                                          latent_code_dim,
                                          T,
                                          re_use_flag)

                        tf.get_variable_scope().reuse_variables()
                        grads = opt.compute_gradients(loss)

                        # Save gradients from different GPUs.
                        tower_grads.append(grads)

        # Summarize gradients from multiple GPUs.
        grads = ops.average_gradients(tower_grads)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        train_op = apply_gradient_op

        init = tf.global_variables_initializer()

        # Configuration of GPU.
        config_ = tf.ConfigProto()

        config_.gpu_options.allow_growth = True

        config_.allow_soft_placement = True

        sess = tf.Session(config=config_)

        sess.run(init)

        if use_mask:
            mask = 1*(Train_data > 0)
        else:
            mask = np.ones(np.shape(Train_data))

        total_sample_list = list(range(0, np.shape(Train_data)[0]))

        reconstruction_error = []

        start = time.time()
        for step in range(1, max_epoch+1):

            total_data_size = np.shape(Train_data)[0]

            total_cnt = total_data_size/(batch_size*num_gpus)

            for itr_cnt in range(int(total_cnt)):

                sel_pos = random.sample(total_sample_list, batch_size*num_gpus)

                cur_data = Train_data[sel_pos, :]

                cur_mask = mask[sel_pos, :]

                sess.run(train_op,

                         feed_dict={train_data: cur_data,

                                    val_mask: cur_mask})

            if step % epoch_per_check == 0 and step > 0:

                all_input = tf.placeholder(
                    tf.float32, [np.shape(Train_data)[0], np.shape(Train_data)[1]])

                layer_output, train_latent_code, _ = Inference(
                    all_input, latent_code_dim, T, re_use=True)

                train_code_val, layer_output_val = sess.run(
                    [train_latent_code[-1], layer_output[-1]], feed_dict={all_input: Train_data})

                recon_error = np.linalg.norm(np.multiply(mask, layer_output_val)-np.multiply(
                    mask, Train_data))/np.linalg.norm(np.multiply(mask, Train_data))
                reconstruction_error.append(recon_error)
                print("Finisheded runing epoch：" + str(step))
                print('Current reconstruction error is: '+str(recon_error))

                if len(reconstruction_error) >= 2:
                    if (abs(reconstruction_error[-1] - reconstruction_error[-2])/reconstruction_error[-2] < 1e-3) or step == max_epoch - 1:
                        sess.close()
                        break

        Model = {}

        test_data_holder = tf.placeholder(
            tf.float32, [None, np.shape(Train_data)[1]])
        test_layer_out, test_latent_code, test_correcting_layer = Inference(
            test_data_holder, latent_code_dim, T, re_use=True)

        Model['latent_code_session'] = sess
        Model['test_input'] = test_data_holder
        Model['Imputated_output'] = test_layer_out
        Model['latent_code'] = test_latent_code
        Model['correcting_layer_output'] = test_correcting_layer

        duration = time.time()-start
        print('Finish training ' + str(len(Train_data)) + ' samples after '+str(step)+' epochs. The total training time is ' +
              str(duration)+' seconds.')

        return Model


def Inference(input_d, latent_code_dim, T, re_use=False):
    '''
    The deep neural network structure of scScope

    Parameters:
        input_d:            gene expression matrix of n*m where n is the number of cell and m is the number of genes.
        latent_code_dim:    the dimension of features outputted by scScope.
        T:                  number of recurrent structures used in deep learning framework.
        re_use:             if re-use variables in training.

    Output:                 
        output_list:        outputs of decoder (y_c in the paper) in T recurrent structures.
        latent_code_list:   latent representations (h_c in the paper) in T recurrent structures.
        correcting_layer_list:  imputations to the zero-expressed genes in T recurrent structures.


    Copyright Altschuler & Wu Lab 2018. All Rights Reserved.
    '''

    input_shape = input_d.get_shape().as_list()

    input_dim = input_shape[1]

    with tf.variable_scope('ScScope_inference') as scope_all:

        if re_use == True:

            scope_all.reuse_variables()

        latent_code_list = []
        output_list = []
        correcting_layer_list = []
        with tf.variable_scope('Correction'):
            for i in range(T):
                if i == 0:
                    W_encoder = ops._variable_with_weight_decay('encoding_layer_weights' + str(i),
                                                                [input_dim,
                                                                    latent_code_dim],
                                                                stddev=0.1, wd=0)
                    b_encoder = ops._variable_on_cpu('encoding_layer_bias' + str(i), [latent_code_dim],
                                                     tf.constant_initializer(0))

                    W_decoder = ops._variable_with_weight_decay('dencoding_layer_weights' + str(i),
                                                                [latent_code_dim,
                                                                    input_dim],
                                                                stddev=0.1, wd=0)
                    b_decoder = ops._variable_on_cpu('dencoding_layer_bias' + str(i), [input_dim],
                                                     tf.constant_initializer(0))
                    latent_code = tf.nn.relu(
                        tf.matmul(input_d, W_encoder)+b_encoder)
                    output = tf.nn.relu(
                        tf.matmul(latent_code, W_decoder)+b_decoder)
                else:
                    if i == 1:
                        W_feedback = ops._variable_with_weight_decay('impute_layer_weights'+str(i), [input_dim, input_dim],
                                                                     stddev=0.1, wd=0)
                        b_feedback = ops._variable_on_cpu(
                            'impute_layer_bias'+str(i), [input_dim], tf.constant_initializer(0))

                    correcting_layer = tf.multiply(
                        1-tf.sign(input_d), tf.nn.relu(tf.matmul(output, W_feedback)+b_feedback))

                    input_vec = correcting_layer+input_d
                    latent_code = tf.nn.relu(
                        tf.matmul(input_vec, W_encoder)+b_encoder)
                    output = tf.nn.relu(
                        tf.matmul(latent_code, W_decoder)+b_decoder)
                    correcting_layer_list.append(correcting_layer)
                latent_code_list.append(latent_code)
                output_list.append(output)

        return output_list, latent_code_list, correcting_layer_list


def Cal_Loss(outpout_layer_list, val_mask, input_data):
    '''
    Loss function of scScope.

    Parameter: 
        outpout_layer_list: encoder output of T recurrent structures in scScope.
        val_mask:           flag indicating only use non-zero genes to calculate losses.
        input_data:         original gene expression matrix inputted into scScope.

    Output:

        acc_loss:           loss function value.

    Copyright Altschuler & Wu Lab 2018. All Rights Reserved.
    '''

    for i in range(len(outpout_layer_list)):
        layer_out = outpout_layer_list[i]
        if i == 0:
            reconstruct_loss = tf.reduce_mean(
                tf.norm(tf.multiply(val_mask, (layer_out-input_data))))
        else:
            reconstruct_loss = reconstruct_loss + \
                tf.reduce_mean(
                    tf.norm(tf.multiply(val_mask, (layer_out-input_data))))
    acc_loss = reconstruct_loss
    tf.add_to_collection('losses', acc_loss)
    return acc_loss


def tower_loss(scope, batch_data, val_mask, latent_code_dim, T, re_use_flag):
    '''
    Overall losses of scScope on multiple GPUs.

    Parameter: 
        scope:              tensorflow name scope
        batch_data:         cell batch for calculating the loss
        val_mask:           flag indicating only use non-zero genes to calculate losses.
        latent_code_dim:    the dimension of features outputted by scScope.
        T:                  number of recurrent structures used in deep learning framework.
        re_use_flag:        if re-use variables in training.

    Output:
        total_loss:         total loss of multiple GPUs.

    Copyright Altschuler & Wu Lab 2018. All Rights Reserved.
    '''

    layer_out, latent_code, _ = Inference(
        batch_data, latent_code_dim, T, re_use=re_use_flag)

    _ = Cal_Loss(layer_out, val_mask, batch_data)

    losses = tf.get_collection('losses', scope)

    total_loss = tf.add_n(losses, name='total_loss')

    return total_loss


def predict(test_data, model):
    '''
    Feed forward to make predication using learned scScope model.

    Parameter: 
        test_data:      gene expression matrix need to make prediction.
        model:          pre-trained scScope model.

    Output:
        latent_code:    scScope features for inputted gene expressions.
        imputation:     gene expressions with imputations.

    Copyright Altschuler & Wu Lab 2018. All Rights Reserved.
    '''

    sess = model['latent_code_session']
    test_place_holder = model['test_input']
    output = model['Imputated_output']
    latent_code = model['latent_code']
    i = len(latent_code)-1
    latent_code_val, output_val = sess.run([latent_code[i], output[i]], feed_dict={
                                           test_place_holder: test_data})

    latent_code = np.squeeze(latent_code_val)
    imputation = np.squeeze(output_val)

    return latent_code, imputation


def scalable_cluster(latent_code,
                     kmeans_num=500,
                     cluster_num=400,
                     display_step=50,
                     phenograh_neighbor=30
                     ):
    '''
    Scalable  cluster:
    To leverage the power of graph clustering on analyzing these large-scale data, we designed a scalable clustering strategy by combining k-means and PhenoGraph.
    In detail, we divided cells into M (kmeans_num) groups with equal size and performed K-means (cluster_num) clustering on each group independently. The whole dataset was split to M×K clusters and we only input the cluster centroids into PhenoGraph for graph clustering. Finally, each cell was assigned to graph clusters according to the cluster labels of its nearest centroids.

    Parameters:

        latent_code:    n*m matrix of gene expression levels or representations of gene expression. n is cell size and m is gene or representation size.
        kmeans_num:     the number of independent K-means clusterings used. This is also the subset number.
        cluster_num:    cluster number for each K-means clustering. This is also the "n_clusters" in KMeans function in sklearn package.
        display_step:   displaying the process of K-means clustering.
        phenograh_neighbor: "k" parameter in PhenoGraph package.

    Output:

        Cluster labels for input cells.


    Copyright Altschuler & Wu Lab 2018. All Rights Reserved.
    '''

    print('Scalable clustering:')
    print('Use %d subsets of cells for initially clustering...' % kmeans_num)

    stamp = np.floor(np.linspace(0, latent_code.shape[0], kmeans_num + 1))
    stamp = stamp.astype(int)

    cluster_ceter = np.zeros([kmeans_num * cluster_num, latent_code.shape[1]])
    mapping_sample_kmeans = np.zeros(latent_code.shape[0])

    for i in range(kmeans_num):

        low_bound = stamp[i]
        upp_bound = stamp[i + 1]
        sample_range = np.arange(low_bound, upp_bound)
        select_sample = latent_code[sample_range, :]

        kmeans = KMeans(n_clusters=cluster_num,
                        random_state=0).fit(select_sample)
        label = kmeans.labels_

        for j in range(cluster_num):
            cluster_sample_idx = np.nonzero(label == j)[0]
            cluster_sample = select_sample[cluster_sample_idx, :]
            cluster_ceter[i * cluster_num + j,
                          :] = np.mean(cluster_sample, axis=0)
            mapping_sample_kmeans[sample_range[cluster_sample_idx]
                                  ] = i * cluster_num + j

        if i % display_step == 0:
            print('\tK-means clustering for %d subset.' % i)

    print('Finish intially clustering by K-means.')
    print('Start PhenoGraph clustering...\n')

    label_pheno, graph, Q = phenograph.cluster(
        cluster_ceter, k=phenograh_neighbor, n_jobs=1)

    label = np.zeros(latent_code.shape[0])
    for i in range(label_pheno.max() + 1):
        center_index = np.nonzero(label_pheno == i)[0]
        for j in center_index:
            sample_index = np.nonzero(mapping_sample_kmeans == j)[
                0]  # samples belong to this center
            label[sample_index] = i
    print('Finish density down-sampling clustering.')

    return label
