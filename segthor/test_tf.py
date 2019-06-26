import tensorflow as tf
import numpy as np
from PIL import Image

import unet
import read

result_path = './result/nib'
BATCH_SIZE = 8

test_batch = read.load_data_npy("/cmach-data/segthor/Val/")
y_val = read.load_label_npy("/cmach-data/segthor/Val/")
print(test_batch.shape)

X = tf.placeholder(tf.float32, [BATCH_SIZE, 512, 512, 1], name='images')

logits = tf.reshape(unet.inference(X), [-1, 5])
softmax = tf.nn.softmax(logits)

with tf.Session() as sess:
  saver = tf.train.Saver()
  model_file = tf.train.latest_checkpoint('model/')
  saver.restore(sess, model_file)

  output = []

  n_batches = int(test_batch.shape[0] / BATCH_SIZE)
  for b in range(n_batches):
    softmax_batch = sess.run(softmax, feed_dict={X:test_batch[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]})
    if (b % 50 == 0):
      print(b)
    # softmax_batch = np.argmax(softmax_batch, axis = 1)
    # softmax_batch = np.reshape([BATCH_SIZE, 512, 512])
    if (b == 0):
      output = softmax_batch
    else:
      output = np.concatenate((output, softmax_batch), axis = 0)
  
  print(output.shape)
  output = np.argmax(output, axis = 1)
  output = np.reshape(output, [-1, 512, 512])
  # img_out = output * 255 / 4
  # img_out = img_out.astype(np.uint8)
  # print(img_out.shape)
  # for i in range(300):
  #   img = Image.fromarray(img_out[i])
  #   img.save('./result/tf/' + str(i) + ".png")
  
  # read.npy2nii(output, result_path)

  y_val = np.argmax(y_val, axis = 3)
  y_val = y_val[:len(output)]
  read.calc_dice(output, y_val)
  



    
    

