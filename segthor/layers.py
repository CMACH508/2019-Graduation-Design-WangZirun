import tensorflow as tf

def conv_layers(x, scope_name, input_channel, output_channel):
  with tf.variable_scope(scope_name):
    # first conv
    #with tf.device('/cpu:0'):
    weights = tf.get_variable(  # _variable_on_cpu
        name='weights_1',
        shape=[3, 3, input_channel, output_channel],
        initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases_1', shape=[output_channel], initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(x, weights, [1, 1, 1, 1], padding='SAME')
    activate = tf.nn.relu(tf.nn.bias_add(conv, biases))

    # second conv
    with tf.device('/cpu:0'):
        weights = tf.get_variable(  # _variable_on_cpu
            name='weights_2',
            shape=[3, 3, output_channel, output_channel],
            initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name='biases_2', shape=[output_channel], initializer=tf.constant_initializer(0.0))
    conv1 = tf.nn.conv2d(conv, weights, [1, 1, 1, 1], padding='SAME')
    activate = tf.nn.relu(tf.nn.bias_add(conv1, biases), name='feature_map')

    return activate
  
def deconv(x, scope_name, input_channel, output_channel):
  with tf.variable_scope(scope_name):
    #with tf.device('/cpu:0'):
    kernel = tf.get_variable(
        name='upconv_filter', 
        shape=[2, 2, output_channel, input_channel],
        initializer=tf.contrib.layers.xavier_initializer())
    feature_shape = tf.shape(x)
    output_shape = tf.stack([feature_shape[0], feature_shape[1] * 2, feature_shape[2] * 2, tf.cast(feature_shape[3] / 2, tf.int32)])
    upconv = tf.nn.conv2d_transpose(value=x, filter=kernel, output_shape=output_shape, strides=[1, 2, 2, 1])
    
    return upconv