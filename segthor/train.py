from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import load_model

import read

import models

# BACKBONE = 'resnet34'
# preprocess_input = get_preprocessing(BACKBONE)

# load your data
# x_train, y_train = read.load_data("/cmach-data/segthor/dataset/Train/", 1, 35)
x_train = read.load_data_npy("/cmach-data/segthor/Train/")
y_train = read.load_label_npy("/cmach-data/segthor/Train/")
x_val = read.load_data_npy("/cmach-data/segthor/Val/")
y_val = read.load_label_npy("/cmach-data/segthor/Val/")
# x_test = read.load_test_data("/cmach-data/segthor/test/", 41, 60)
print(len(x_train), len(x_val))

# preprocess input
# x_train = preprocess_input(x_train)
# x_val = preprocess_input(x_val)

# define model
# model = Unet(BACKBONE, input_shape=(None, None, 1), classes=5, encoder_weights=None)
# model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

# model = load_model('unet_membrane.hdf5')

model = models.unet()
model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

model_checkpoint = ModelCheckpoint('ordjcd50.hdf5', monitor='loss',verbose=1, save_best_only=True)

# fit model
model.fit(
    x=x_train,
    y=y_train,
    batch_size=6,
    epochs=50,
    validation_data=(x_val, y_val),
    callbacks=[model_checkpoint]
)

# predictions = model.predict_classes(x_test)
# print(predictions.shape)
# read.save_result("./result/", predictions)

