from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from PIL import Image
import os

NUM_CLASSES = 2
OUTPUT_HEIGHT = 512
OUTPUT_WIDTH = 512
INPUT_WIDTH = 512
INPUT_HEIGHT = 512

def TrainDataGenerator(batch_size, train_filepath, data_folder, label_folder):
  data_gen_args = dict(rotation_range=0.2,
                      width_shift_range=0.05,
                      height_shift_range=0.05,
                      shear_range=0.05,
                      zoom_range=0.05,
                      horizontal_flip=True,
                      fill_mode='reflect')

  data_dg = ImageDataGenerator(**data_gen_args)
  label_dg = ImageDataGenerator(**data_gen_args)
  
  data_generator = data_dg.flow_from_directory(
    train_filepath,
    classes = [data_folder],
    class_mode = None,
    color_mode = 'grayscale',
    target_size = (INPUT_HEIGHT, INPUT_WIDTH),
    batch_size = batch_size,
    #save_to_dir = 'augmented',
    #save_prefix = 'd',
    seed = 1)

  label_generator = label_dg.flow_from_directory(
    train_filepath,
    classes = [label_folder],
    class_mode = None,
    color_mode = 'grayscale',
    target_size = (INPUT_HEIGHT, INPUT_WIDTH),
    batch_size = batch_size,
    # save_to_dir = 'augmented',
    # save_prefix = 'l',
    seed = 1)

  train_generator = zip(data_generator, label_generator)
  for (d,l) in train_generator:
    if (np.max(d) > 1):
      d = d / 255
      l = l / 255
      l[l > 0.5] = 1
      l[l < 0.5] = 0
    yield (d,l)

def ReadTestData():
  test_path = 'isbi/test'
  imgs = os.listdir(test_path)
  imgs.sort()
  print(imgs)
  test_batch = []
  for img_name in imgs:
    img = Image.open(os.path.join(test_path, img_name))
    img = np.asarray(img).reshape([OUTPUT_HEIGHT, OUTPUT_WIDTH, 1])
    test_batch.append(img)
  test_batch = np.array(test_batch)
  print(test_batch.shape)
  if (test_batch.max() > 1):
    test_batch = test_batch / 255
  # tiff_array = test_batch.astype(np.uint16).reshape([512,512,30])
  # im = Image.fromarray(tiff_array, mode='I;16')
  # im.save(r'a16.tiff')
  return test_batch



if __name__=='__main__':
  # ReadTestData()
  gen = TrainDataGenerator(100, 'isbi/train', 'data', 'label')
  cnt = 0
  for d,l in gen:
    print(d.shape)
    cnt += 1
    break
    
  print(cnt)