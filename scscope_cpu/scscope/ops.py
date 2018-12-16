import tensorflow as tf



def _variable_with_weight_decay(name, shape, stddev, wd):
    """
    Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.

    A weight decay is added only if one is specified.

    Args:

      name: name of the variable

      shape: list of ints

      stddev: standard deviation of a truncated Gaussian

      wd: add L2Loss weight decay multiplied by this float. If None, weight

          decay is not added for this Variable.

    Returns:

      Variable Tensor

    Altschuler & Wu Lab 2018. 
    Software provided as is under Apache License 2.0.
    """

    dtype = tf.float32

    var = _variable_on_cpu(

        name,

        shape,

        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))

    if wd != 0:

        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')

        tf.add_to_collection('losses', weight_decay)

    return var


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:

      name: name of the variable

      shape: list of ints

      initializer: initializer for Variable

    Returns:

      Variable Tensor

    Altschuler & Wu Lab 2018. 
    Software provided as is under Apache License 2.0.
    """

    with tf.device('/cpu:0'):

        dtype = tf.float32

        var = tf.get_variable(
            name, shape, initializer=initializer, dtype=dtype)

    return var


def average_gradients(tower_grads):
    """ Summarize the gradient calculated by each GPU.

    Altschuler & Wu Lab 2018. 
    Software provided as is under Apache License 2.0.
    """

    average_grads = []

    for grad_and_vars in zip(*tower_grads):

        # Note that each grad_and_vars looks like the following:

        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))

        grads = []

        for g, var1 in grad_and_vars:

            # Add 0 dimension to the gradients to represent the tower.

            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.

            grads.append(expanded_g)

        # Average over the 'tower' dimension.

        grad = tf.concat(axis=0, values=grads)

        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]

        grad_and_var = (grad, v)

        average_grads.append(grad_and_var)

    return average_grads
