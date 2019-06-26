from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import load_model

import numpy as np
from PIL import Image

import read

# x_test = read.load_test_data("/cmach-data/segthor/dataset/Test/", 41, 60)
# x_test = read.load_data_npy("/cmach-data/segthor/Test/")
# x_val = read.load_data_npy("/cmach-data/segthor/Val/")
# y_val = read.load_label_npy("/cmach-data/segthor/Val/")
x_val, y_val = read.load_data("/cmach-data/segthor/dataset/Train/", 36, 40)
print(x_val.shape)

model = load_model('adam100morex8.hdf5')

predictions = model.predict(x_val, batch_size = 1)
print(predictions.shape)
predictions = np.argmax(predictions, axis = 3)
print(np.unique(predictions))
y_val = np.argmax(y_val, axis = 3)

# correct = 0
# cnt = 0
# for i in range(len(predictions)):
#   for j in range(len(predictions[i])):
#     for k in range(len(predictions[i][j])):
#       cnt += 1
#       if predictions[i][j][k] == y_val[i][j][k]:
#         correct += 1
# print(correct, cnt, correct / cnt)

img_out = predictions * 255 / 4
img_out = img_out.astype(np.uint8)
print(img_out.shape)
for i in range(300):
  img = Image.fromarray(img_out[i])
  img.save('./result/res/' + str(i) + ".png")

# read.npy2nii(predictions, './result/nibres')

read.calc_dice(predictions, y_val)