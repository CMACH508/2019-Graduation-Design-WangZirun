import tensorflow as tf
import time

import read
import unet

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 500.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.5  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

BATCH_SIZE = 8
NUM_EPOCHS = 100
class_weights = [0.6, 0.4]

print("Start loading...")
x_train = read.load_data_npy("/cmach-data/segthor/Train/")
y_train = read.load_label_npy("/cmach-data/segthor/Train/")
print("Finish loading.")

X = tf.placeholder(tf.float32, [BATCH_SIZE, 512, 512, 1], name='images')
Y = tf.placeholder(tf.int32, [BATCH_SIZE, 512, 512, 1], name='labels')

logits = unet.inference(X)

loss = unet.loss(logits, Y)#, class_weights=class_weights)
tf.summary.scalar('total_loss', loss)

global_step = tf.train.get_or_create_global_step()
# lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
#                                 global_step,
#                                 NUM_EPOCHS_PER_DECAY,
#                                 LEARNING_RATE_DECAY_FACTOR,
#                                 staircase=True)
# lr = tf.train.piecewise_constant(global_step,[500,700,1200,1600,2000,2500,3200,4000],[0.1,0.05,0.02,0.01,0.004,0.001,0.0005,0.0001,0.00005])
# tf.summary.scalar('learning_rate', lr)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(loss, global_step=global_step)

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

merge_summary = tf.summary.merge_all()

saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
  if tf.gfile.Exists('./graph'):
    tf.gfile.DeleteRecursively('./graph')
  writer=tf.summary.FileWriter('./graph',sess.graph)

  if tf.gfile.Exists('./model'):
    tf.gfile.DeleteRecursively('./model')
  
  start_time = time.time()
  sess.run(tf.global_variables_initializer())
  n_batches = int(len(y_train) / BATCH_SIZE)
  for i in range(NUM_EPOCHS):
    total_loss = 0
    # data_gen = read.TrainDataGenerator(BATCH_SIZE, 'isbi/train', 'data', 'label')
    for j in range(n_batches):
      print(j)
      x_batch, y_batch = read.next_batch(x_train, y_train, BATCH_SIZE)
      _, loss_batch = sess.run([optimizer, loss], feed_dict={X:x_batch, Y:y_batch})
      total_loss += loss_batch
      # writer.add_summary(ms, i)
    print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))

    # if ((i + 1) % 400 == 0):
    #   saver.save(sess, './model/unet.ckpt', global_step=i+1)
  
  print('Total time: {0} seconds'.format(time.time() - start_time))

  print('Optimization Finished!')

  # TODO
  # save the model DONE
  saver.save(sess, './model/unet.ckpt', global_step=global_step)

  writer.close()