# scScope is a deep-learning based approach designed to identify cell-type composition from large-scale scRNA-seq profiles.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import time
from .ops import average_gradients, _variable_with_weight_decay, _variable_on_cpu
import phenograph
from sklearn.cluster import KMeans


def train_large(train_data_path,
                file_num,
                cell_size,
                gene_size,
                latent_code_dim,
                exp_batch_idx=0,
                use_mask=True,
                batch_size=64,
                max_epoch=100,
                epoch_per_check=100,
                T=2,
                encoder_layers=[],
                decoder_layers=[],
                learning_rate=0.0001,
                beta1=0.05,
                num_gpus=1
                ):
    '''
    scScope training:
	  This function is used to train the scScope model on gene expression data

    Parameters:

      train_data_path:      File path of multiple small gene expression files. Each file is a cell_size * gene_size matrix stored in *.npy format.
                                                    Files are named in "batch_0.npy", "batch_1.npy", ...
      file_num:             Number of gene expression files in "train_data_path".
      cell_size:            Cell numbers in each expression file. All files should include the same number of cells.
      gene_size:            Gene numbers in each expression file. All files should include the same number of genes.
      exp_batch_idx:        Number of experimental batches in the sequencing. if exp_batch_idx = 0, no batch information need to provide.
                                                    Otherwise, experimental batch labels are stored in "exp_batch_label_0.npy", "exp_batch_label_1.npy", ..., corresponding to each data batch file.
                                                    In each file, experimental batch labels are stored in an n * batch_num matrix in one-hot format. Experimental batch labels and data batch files
                                                    are in the same directory.
      latent_code_dim:      The feature dimension outputted by scScope.
      batch_size:           Number of cells used in each training iteration.
      max_epoch:            Maximal epoch used in training.
      epoch_per_check:      Step to display current loss.
      T:                    Depth of recurrence used in deep learning framework.
      use_mask:             Flag indicating whether to use only non-zero entries in calculating losses.
      learning_rate:        Step length in gradient descending algorithm.
      beta1:                The beta1 parameter in AdamOptimizer.
      num_gpus:             Number of gpus used for training in parallel.


    Output:

      model: a dataframe of scScope outputs with keys:
                    'latent_code_session':      tensorflow session used in training.
                    'test_input':               tensorflow dataholder for test data.
                    'test_exp_batch_idx':       tensorflow dataholder for experimental batch label.
                    'imputated_output':         imputed gene expressions.
                    'latent_code':              latent features by scScope.
                    'removed_batch_effect':     correcting layer learning by scScope.

    Altschuler & Wu Lab 2018.
    Software provided as is under Apache License 2.0.
    '''

    batch_size = int(batch_size * num_gpus)
    learning_rate = learning_rate * num_gpus

    if exp_batch_idx == 0:
        exp_batch_idx_input = np.zeros((cell_size, 1))
        consider_exp_batch = False
    else:
        consider_exp_batch = True

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        train_data = tf.placeholder(
            tf.float32, [batch_size, gene_size])
        exp_batch_idx = tf.placeholder(tf.float32,
                                       [batch_size, exp_batch_idx])

        # Create an optimizer that performs gradient descent.
        opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1)

        # Calculate the gradients on models deployed on each GPU then summarized the gradients.
        tower_grads = []
        tower_grads2 = []

        with tf.variable_scope(tf.get_variable_scope()):

            for i in range(num_gpus):

                print('Building Computational Graph on GPU-' + str(i))

                with tf.device('/gpu:%d' % (i + 1)):

                    with tf.name_scope('%s_%d' % ('tower', i)) as scope:

                        itv = int(batch_size / num_gpus)

                        if i == 0:

                            re_use_flag = False

                        else:

                            re_use_flag = True

                        loss = tower_loss(scope,
                                          train_data[(i) *
                                                     itv:(i + 1) * itv, :],
                                          use_mask,
                                          latent_code_dim,
                                          T,
                                          encoder_layers,
                                          decoder_layers,
                                          exp_batch_idx[(
                                              i) * itv:(i + 1) * itv, :],
                                          re_use_flag)

                        tf.get_variable_scope().reuse_variables()

                        t_vars = tf.trainable_variables()

                        inference_para = [
                            var for var in t_vars if 'inference' in var.name]
                        grads = opt.compute_gradients(loss, inference_para)
                        tower_grads.append(grads)

                        if consider_exp_batch:
                            exp_batch_effect_para = [
                                var for var in t_vars if 'batch_effect_removal' in var.name]
                            grads2 = opt.compute_gradients(
                                loss, exp_batch_effect_para)
                            tower_grads2.append(grads2)

        # Summarize gradients from multiple GPUs.
        grads = average_gradients(tower_grads)
        apply_gradient_op = opt.apply_gradients(grads)
        train_op = apply_gradient_op

        if consider_exp_batch:
            grads2 = average_gradients(tower_grads2)
            apply_gradient_op2 = opt.apply_gradients(grads2)
            train_op2 = apply_gradient_op2

        init = tf.global_variables_initializer()

        # Configuration of GPU.
        config_ = tf.ConfigProto()

        config_.gpu_options.allow_growth = True

        config_.allow_soft_placement = True

        sess = tf.Session(config=config_)

        sess.run(init)

        reconstruction_error = []

        start = time.time()
        for step in range(1, max_epoch + 1):

            for file_count in range(file_num):

                train_data_real_val = np.load(
                    train_data_path + '/batch_' + str(file_count) + '.npy')
                if exp_batch_idx > 0:
                    exp_batch_idx_input = np.load(
                        train_data_path + '/exp_batch_label_' + str(file_count) + '.npy')

                total_data_size = np.shape(train_data_real_val)[0]
                total_sample_list = list(range(total_data_size))

                total_cnt = total_data_size / (batch_size)

                for itr_cnt in range(int(total_cnt)):

                    sel_pos = random.sample(total_sample_list, batch_size)

                    cur_data = train_data_real_val[sel_pos, :]
                    cur_exp_batch_idx = exp_batch_idx_input[sel_pos, :]

                    sess.run(train_op,
                             feed_dict={train_data: cur_data,
                                        exp_batch_idx: cur_exp_batch_idx})
                    if consider_exp_batch:
                        sess.run(train_op2,
                                 feed_dict={train_data: cur_data,
                                            exp_batch_idx: cur_exp_batch_idx})

                if step % epoch_per_check == 0 and step > 0:

                    all_input = tf.placeholder(
                        tf.float32, [np.shape(train_data_real_val)[0], np.shape(train_data_real_val)[1]])
                    exp_batch_idx_all = tf.placeholder(
                        tf.float32, [np.shape(exp_batch_idx_input)[0], np.shape(exp_batch_idx_input)[1]])

                    layer_output, train_latent_code, _ = Inference(
                        all_input, latent_code_dim, T, encoder_layers, decoder_layers, exp_batch_idx_all, re_use=True)

                    train_code_val, layer_output_val = sess.run(
                        [train_latent_code[-1], layer_output[-1]],
                        feed_dict={all_input: train_data_real_val, exp_batch_idx_all: exp_batch_idx_input})

                    mask = np.sign(train_data_real_val)
                    recon_error = np.linalg.norm(np.multiply(mask, layer_output_val) - np.multiply(
                        mask, train_data_real_val)) / np.linalg.norm(np.multiply(mask, train_data_real_val))
                    reconstruction_error.append(recon_error)
                    print("Finisheded epoch：" + str(step))
                    print('Current reconstruction error is: ' + str(recon_error))

                    if len(reconstruction_error) >= 2:
                        if (abs(reconstruction_error[-1] - reconstruction_error[-2]) / reconstruction_error[-2] < 1e-3):
                            break

        model = {}

        test_data_holder = tf.placeholder(
            tf.float32, [None, gene_size])
        test_exp_batch_idx = tf.placeholder(
            tf.float32, [None, exp_batch_idx])

        test_layer_out, test_latent_code, removed_batch_effect = Inference(
            test_data_holder, latent_code_dim, T, encoder_layers, decoder_layers, test_exp_batch_idx, re_use=True)

        model['latent_code_session'] = sess
        model['test_input'] = test_data_holder
        model['test_exp_batch_idx'] = test_exp_batch_idx
        model['imputated_output'] = test_layer_out
        model['latent_code'] = test_latent_code
        model['removed_batch_effect'] = removed_batch_effect

        duration = time.time() - start
        print('Finish training ' + str(len(train_data)) + ' samples after ' + str(
            step) + ' epochs. The total training time is ' +
            str(duration) + ' seconds.')

        return model


def predict_large(train_data_path,
                  file_num,
                  model,
                  exp_batch_idx=0):
    '''
    Output the latent feature and imputed sequence for large scale dataset after training the model.

    Parameters:
            train_data_path:    The same data path as in "train_large()".
            file_num:           Number of data files in train_data_path.
            exp_batch_idx:      Number of experimental batches in sequencing. If exp_batch_idx=0, the function is run without batch correction.
            model:              The pre-trained model by "train_large()".

    Output:
            Latent features and imputed genes for each data file.
            For data file "batch_i.npy",  corresponding latent features and imputed gene expressions are stored in
            "feature_i.npy" and "imputation_i.npy" files respectively in the same directory.

    Altschuler & Wu Lab 2018.
    Software provided as is under Apache License 2.0.
    '''
    for file_count in range(file_num):

        train_data = np.load(
            train_data_path + '/batch_' + str(file_count) + '.npy')
        if exp_batch_idx > 0:
            batch_effect = np.load(
                train_data_path + '/exp_batch_label_' + str(file_count) + '.npy')
        else:
            batch_effect = []
        latent_fea, output_val, predicted_batch_effect = predict(
            train_data, model, batch_effect=batch_effect)
        np.save(train_data_path + '/feature_' +
                str(file_count) + '.npy', latent_fea)
        np.save(train_data_path + '/imputation_' +
                str(file_count) + '.npy', output_val)


def predict(test_data, model, batch_effect=[]):
    '''
    Make predications using the learned scScope model.
	
    Parameter:
            test_data:      gene expression matrix need to make prediction.
            model:          pre-trained scScope model.

    Output:
            latent_fea:             scScope features for inputted gene expressions.
            output_val:             gene expressions with imputations.
            predicted_batch_effect: batch effects inferenced by scScope, if experimental batches exist.

    Altschuler & Wu Lab 2018.
    Software provided as is under Apache License 2.0.
    '''

    sess = model['latent_code_session']
    test_data_holder = model['test_input']
    test_exp_batch_idx_holder = model['test_exp_batch_idx']
    output = model['imputated_output']
    latent_code = model['latent_code']
    removed_batch_effect = model['removed_batch_effect']
    if len(batch_effect) == 0:
        batch_effect_idx = np.zeros((np.shape(test_data)[0], 1))
    else:
        batch_effect_idx = batch_effect

    for i in range(len(latent_code)):

        latent_code_val, output_val, predicted_batch_effect = sess.run(
            [latent_code[i], output[i], removed_batch_effect], feed_dict={
                test_data_holder: test_data, test_exp_batch_idx_holder: batch_effect_idx})
        if i == 0:
            latent_fea = latent_code_val
            output_total = output_val
        else:
            latent_fea = np.concatenate([latent_fea, latent_code_val], 1)
            output_total = output_total + output_val

    output_val = output_total / len(latent_code)
    return latent_fea, output_val, predicted_batch_effect


def Inference(input_d, latent_code_dim, T, encoder_layers, decoder_layer, exp_batch_idx=[], re_use=False):
    '''
    The deep neural network structure of scScope

    Parameters:
			input_d:            gene expression matrix of dim n * m; n = number of cells, m = number of genes.
            latent_code_dim:    the dimension of features outputted by scScope.
            T:                  number of recurrent structures used in deep learning framework.
            encoder_layers:
            decoder_layer:
            exp_batch_idx:      if provided, experimental batch labels are stored in an n * batch_num matrix in one-hot format.
            re_use:             if re-use variables in training.

    Output:
            output_list:        outputs of decoder (y_c in the paper) in T recurrent structures.
            latent_code_list:   latent representations (h_c in the paper) in T recurrent structures.
            batch_effect_removal_layer:  experimental batch effects inferred by scScope.


    Altschuler & Wu Lab 2018.
    Software provided as is under Apache License 2.0.
    '''

    input_shape = input_d.get_shape().as_list()

    input_dim = input_shape[1]

    with tf.variable_scope('scScope') as scope_all:

        if re_use == True:
            scope_all.reuse_variables()

        latent_code_list = []
        output_list = []
        exp_batch_id_shape = exp_batch_idx.get_shape().as_list()
        exp_batch_dim = exp_batch_id_shape[1]
        with tf.variable_scope('batch_effect_removal'):
            batch_effect_para_weight = _variable_with_weight_decay('batch_effect_weight',
                                                                   [exp_batch_dim,
                                                                    input_dim],
                                                                   stddev=0, wd=0)

            batch_effect_removal_layer = tf.matmul(
                exp_batch_idx, batch_effect_para_weight)

        with tf.variable_scope('inference'):
            for i in range(T):
                if i == 0:
                    encoder_layer_list_W = []
                    encoder_layer_list_b = []
                    if len(encoder_layers) > 0:
                        for l in range(len(encoder_layers)):
                            if l == 0:
                                encoder_layer_list_W.append(_variable_with_weight_decay('encoder_layer' + str(l),
                                                                                        [input_dim,
                                                                                         encoder_layers[l]],
                                                                                        stddev=0.1, wd=0))
                                encoder_layer_list_b.append(
                                    _variable_on_cpu('encoder_layer_bias' + str(l), [encoder_layers[l]],
                                                     tf.constant_initializer(0)))
                            else:
                                encoder_layer_list_W.append(_variable_with_weight_decay('encoder_layer' + str(l),
                                                                                        [encoder_layers[l - 1],
                                                                                         encoder_layers[l]],
                                                                                        stddev=0.1, wd=0))
                                encoder_layer_list_b.append(
                                    _variable_on_cpu('encoder_layer_bias' + str(l), [encoder_layers[l]],
                                                     tf.constant_initializer(0)))
                        latent_code_layer_input_dim = encoder_layers[-1]

                    else:
                        latent_code_layer_input_dim = input_dim

                    W_fea = _variable_with_weight_decay('latent_layer_weights',
                                                        [latent_code_layer_input_dim,
                                                         latent_code_dim],
                                                        stddev=0.1, wd=0)
                    b_fea = _variable_on_cpu('latent_layer_bias', [latent_code_dim],
                                             tf.constant_initializer(0))

                    decoder_layer_list_W = []
                    decoder_layer_list_b = []
                    if len(decoder_layer) > 0:
                        for l in range(len(decoder_layer)):
                            if l == 0:
                                decoder_layer_list_W.append(_variable_with_weight_decay('dencoder_layer' + str(l),
                                                                                        [latent_code_dim,
                                                                                         decoder_layer[l]],
                                                                                        stddev=0.1, wd=0))
                                decoder_layer_list_b.append(
                                    _variable_on_cpu('decoder_layer_bias' + str(l), [decoder_layer[l]],
                                                     tf.constant_initializer(0)))
                            else:
                                decoder_layer_list_W.append(_variable_with_weight_decay('dencoder_layer' + str(l),
                                                                                        [decoder_layer[l - 1],
                                                                                         decoder_layer[l]],
                                                                                        stddev=0.1, wd=0))
                                decoder_layer_list_b.append(
                                    _variable_on_cpu('decoder_layer_bias' + str(l), [decoder_layer[l]],
                                                     tf.constant_initializer(0)))
                        decoder_last_layer_dim = decoder_layer[-1]

                    else:
                        decoder_last_layer_dim = latent_code_dim

                    W_recon = _variable_with_weight_decay('reconstruction_layer_weights',
                                                          [decoder_last_layer_dim,
                                                           input_dim],
                                                          stddev=0.1, wd=0)
                    b_recon = _variable_on_cpu('reconstruction_layer_bias', [input_dim],
                                               tf.constant_initializer(0))
                    input_vec = tf.nn.relu(
                        input_d - batch_effect_removal_layer)
                else:

                    if i == 1:
                        W_feedback_1 = _variable_with_weight_decay('impute_layer_weights',
                                                                   [input_dim, 64],
                                                                   stddev=0.1, wd=0)
                        b_feedback_1 = _variable_on_cpu(
                            'impute_layer_bias', [64], tf.constant_initializer(0))

                        W_feedback_2 = _variable_with_weight_decay('impute_layer_weights2',
                                                                   [64, input_dim],
                                                                   stddev=0.1, wd=0)
                        b_feedback_2 = _variable_on_cpu(
                            'impute_layer_bias2', [input_dim], tf.constant_initializer(0))

                    intermediate_layer = tf.nn.relu(
                        tf.matmul(output, W_feedback_1) + b_feedback_1)
                    imputation_layer = tf.multiply(
                        1 - tf.sign(input_d), (tf.matmul(intermediate_layer, W_feedback_2) + b_feedback_2))

                    input_vec = tf.nn.relu(
                        imputation_layer + input_d - batch_effect_removal_layer)

                intermedate_encoder_layer_list = []
                if len(encoder_layer_list_W) > 0:
                    for i in range(len(encoder_layer_list_W)):
                        if i == 0:
                            intermedate_encoder_layer_list.append(tf.nn.relu(
                                tf.matmul(input_vec, encoder_layer_list_W[i]) + encoder_layer_list_b[i]))
                        else:
                            intermedate_encoder_layer_list.append(tf.nn.relu(tf.matmul(
                                intermedate_encoder_layer_list[-1], encoder_layer_list_W[i]) + encoder_layer_list_b[i]))

                    intermedate_encoder_layer = intermedate_encoder_layer_list[-1]
                else:
                    intermedate_encoder_layer = input_vec

                latent_code = tf.nn.relu(
                    tf.matmul(intermedate_encoder_layer, W_fea) + b_fea)

                inter_decoder_layer_list = []

                if len(decoder_layer_list_W) > 0:
                    for i in range(len(decoder_layer_list_W)):
                        if i == 0:
                            inter_decoder_layer_list.append(tf.nn.relu(
                                tf.matmul(latent_code, decoder_layer_list_W[i]) + decoder_layer_list_b[i]))
                        else:
                            inter_decoder_layer_list.append(tf.nn.relu(tf.matmul(
                                inter_decoder_layer_list[-1], decoder_layer_list_W[i]) + decoder_layer_list_b[i]))
                    inter_decoder_layer = inter_decoder_layer_list[-1]
                else:
                    inter_decoder_layer = latent_code

                output = tf.nn.relu(
                    tf.matmul(inter_decoder_layer, W_recon) + b_recon)
                latent_code_list.append(latent_code)
                output_list.append(output)

        return output_list, latent_code_list, batch_effect_removal_layer


def tower_loss(scope, batch_data, use_mask, latent_code_dim, T, encoder_layers, decoder_layers, exp_batch_id,
               re_use_flag):
    '''
    Overall losses of scScope on multiple GPUs.

    Parameter:
            scope:              tensorflow name scope
            batch_data:         cell batch for calculating the loss
            use_mask:           flag indicating only use non-zero genes to calculate losses.
            latent_code_dim:    the dimension of features outputted by scScope.
            T:                  number of recurrent structures used in deep learning framework.
            encoder_layers:
            decoder_layers:
            exp_batch_id:
            re_use_flag:        if re-use variables in training.

    Output:
            total_loss:         total loss of multiple GPUs.

    Altschuler & Wu Lab 2018.
    Software provided as is under Apache License 2.0.
    '''

    layer_out, latent_code, batch_effect_removal_layer = Inference(
        batch_data, latent_code_dim, T, encoder_layers, decoder_layers, exp_batch_id, re_use=re_use_flag)

    _ = Cal_Loss(layer_out, batch_data, use_mask, batch_effect_removal_layer)

    losses = tf.get_collection('losses', scope)

    total_loss = tf.add_n(losses, name='total_loss')

    return total_loss


def Cal_Loss(outpout_layer_list, input_data, use_mask, removed_exp_batch_effect):
    '''
    Loss function of scScope.

    Parameter:
            outpout_layer_list:     encoder output of T recurrent structures in scScope.
            input_data:             original gene expression matrix inputted into scScope.
            use_mask:               flag indicating only use non-zero genes to calculate losses.
            removed_exp_batch_effect:

    Output:

            acc_loss:               loss function value.

    Altschuler & Wu Lab 2018.
    Software provided as is under Apache License 2.0.
    '''

    input_data_corrected = input_data - removed_exp_batch_effect

    if use_mask:
        val_mask = tf.sign(input_data_corrected)
    else:
        val_mask = tf.sign(input_data_corrected + 1)

    for i in range(len(outpout_layer_list)):
        layer_out = outpout_layer_list[i]
        if i == 0:
            reconstruct_loss = tf.reduce_mean(
                tf.norm(tf.multiply(val_mask, (layer_out - input_data_corrected))))
        else:
            reconstruct_loss = reconstruct_loss + \
                tf.reduce_mean(
                    tf.norm(tf.multiply(val_mask, (layer_out - input_data_corrected))))
    acc_loss = reconstruct_loss
    tf.add_to_collection('losses', acc_loss)
    return acc_loss


def scalable_cluster(latent_code,
                     kmeans_num=500,
                     cluster_num=400,
                     display_step=50,
                     phenograh_neighbor=30
                     ):
    '''
    Scalable  cluster:
    To perform graph clustering on large-scale data, we designed a scalable clustering strategy by combining k-means and PhenoGraph.
    Briefly, we divide cells into M (kmeans_num) groups of equal size and perform K-means (cluster_num) clustering on each group independently. 
	The whole dataset is split to M×K clusters and we only input the cluster centroids into PhenoGraph for graph clustering. 
	Finally, each cell is assigned to graph clusters according to the cluster labels of its nearest centroids.

    Parameters:

        latent_code:    n*m matrix; n = number of cells, m = dimension of feature representation.
        kmeans_num:     number of independent K-means clusterings used. This is also the subset number.
        cluster_num:    cluster number for each K-means clustering. This is also the "n_clusters" in KMeans function in sklearn package.
        display_step:   displaying the process of K-means clustering.
        phenograh_neighbor: "k" parameter in PhenoGraph package.
		
    Output:

            label:          Cluster labels for input cells.


    Altschuler & Wu Lab 2018.
    Software provided as is under Apache License 2.0.
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
