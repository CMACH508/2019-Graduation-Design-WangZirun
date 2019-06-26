import tensorflow as tf
import math
import numpy as np

import layers
import read

FIRST_OUTPUT_CHANNEL = 64
NUM_EXTRACTING_LAYER = 5
NUM_CLASSES = read.NUM_CLASSES
NUM_INPUT_CHANNEL = 1

def inference(image):
  #with tf.variable_scope('downsampling'):
  input_x = image
  activates = []
  input_channel = NUM_INPUT_CHANNEL
  output_channel = FIRST_OUTPUT_CHANNEL
  for lyr in range(1, NUM_EXTRACTING_LAYER):
    scope_name = 'conv' + str(lyr)

    activate = layers.conv_layers(input_x, scope_name, input_channel, output_channel)
    # if (lyr == NUM_EXTRACTING_LAYER - 1):
    #   activate = tf.nn.dropout(activate, keep_prob = tf.constant(0.5, dtype=tf.float32))

    activates.append(activate)

    input_x = tf.nn.max_pool(activate, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    input_channel = output_channel
    output_channel = output_channel * 2
  
  scope_name = 'conv' + str(NUM_EXTRACTING_LAYER)
  activate = layers.conv_layers(input_x, scope_name, input_channel, output_channel)
  # activate = tf.nn.dropout(activate, keep_prob = tf.constant(0.5, dtype=tf.float32))
  
  #with tf.variable_scope('upsampling'):
  input_channel = output_channel
  input_x = activate
  for lyr in range(NUM_EXTRACTING_LAYER - 1, 0, -1):
    scope_name = 'deconv' + str(lyr)
    output_channel = int(input_channel / 2)

    #deconv
    upconv = layers.deconv(input_x, scope_name, input_channel, output_channel)

    #skip connection
    contracted_feature = activates[lyr - 1]
    # current_shape = tf.shape(upconv)
    # feature_shape = tf.shape(contracted_feature)

    # current_height = current_shape[1]
    # current_width = current_shape[2]

    # height_to_crop = (feature_shape[1] - current_height) # / 2
    # width_to_crop = (feature_shape[2] - current_width) # / 2

    # cropped_feature = tf.slice(
    #     contracted_feature, 
    #     begin=[0, height_to_crop, width_to_crop, 0],
    #     size=[-1, current_height, current_width, -1])
    
    concat_feature = tf.concat([contracted_feature, upconv], axis=3)

    # conv
    # same channel num as previous
    input_x = layers.conv_layers(concat_feature, scope_name, input_channel, output_channel)
    input_channel = output_channel

  # 1x1 conv
  with tf.device('/cpu:0'):
    weights_1x1 = tf.get_variable(
        name='weight_1x1', 
        shape=[1, 1, input_channel, NUM_CLASSES], 
        initializer=tf.contrib.layers.xavier_initializer())
    biases_1x1 = tf.get_variable(name='biases_1x1', shape=[NUM_CLASSES], initializer=tf.constant_initializer(0.0))
  output_seg = tf.nn.bias_add(tf.nn.conv2d(input_x, weights_1x1, [1, 1, 1, 1], padding='SAME'), biases_1x1)

  return output_seg

def loss(logits, labels, class_weights=None):
  # TODO
  # Add weight maps

  tf.summary.image('ground_truth', tf.cast(labels, tf.float32))

  flatten_logits = tf.reshape(logits, [-1, NUM_CLASSES])
  flatten_labels = tf.reshape(labels, [-1])
  
  inference_seg = tf.reshape(tf.argmax(flatten_logits, axis=1), tf.shape(labels))
  tf.summary.image('inference_segmentation', tf.cast(inference_seg, tf.float32))

  if (class_weights is not None):
    class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
    flatten_one_hot = tf.one_hot(flatten_labels, depth=2)
    weight_map = tf.multiply(flatten_one_hot, class_weights)
    weight_map = tf.reduce_sum(weight_map, axis=1)

    loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flatten_logits,
                                                          labels=flatten_one_hot)
    weighted_loss = tf.multiply(loss_map, weight_map)
    
    loss = tf.reduce_mean(weighted_loss, name='loss')
    return loss

  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flatten_logits, labels=flatten_labels)
  # total_loss = tf.reduce_mean(cross_entropy, name='loss')
  total_loss = tf.reduce_mean(cross_entropy, name='loss')
  return total_loss * 1000
