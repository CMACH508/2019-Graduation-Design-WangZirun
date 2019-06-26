import tensorflow as tf
import numpy as np
from PIL import Image

import unet
import data

result_path = './result'
BATCH_SIZE = 10

test_batch = data.ReadTestData()
X = tf.placeholder(tf.float32, [BATCH_SIZE, data.INPUT_HEIGHT, data.INPUT_WIDTH, 1], name='images')

logits = tf.reshape(unet.inference(X), [-1, data.NUM_CLASSES])
softmax = tf.nn.softmax(logits)

with tf.Session() as sess:
  saver = tf.train.Saver()
  model_file = tf.train.latest_checkpoint('model/')
  saver.restore(sess,model_file)

  nums = []
  for i in range(30):
    nums.append(str(i))
  nums.sort()

  n_batches = int(test_batch.shape[0] / BATCH_SIZE)
  for b in range(n_batches):

    softmax_batch = sess.run(softmax, feed_dict={X:test_batch[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]})
    print(softmax_batch.shape)
    result_batch = []
    for sm in softmax_batch:
      result_batch.append(np.argmax(sm))
    result_batch = np.array(np.reshape(result_batch, [BATCH_SIZE, data.INPUT_HEIGHT, data.INPUT_WIDTH]))

    for i in range(b*BATCH_SIZE, (b+1)*BATCH_SIZE):
      print(nums[i])
      res = (result_batch[i - b * BATCH_SIZE] * 255).astype(np.uint8)
      img = Image.fromarray(res)
      img_path = result_path + '/' +nums[i] + '.png'
      img.save(img_path)


