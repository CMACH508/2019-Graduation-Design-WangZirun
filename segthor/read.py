import nibabel as nib
import numpy as np
import json
import os

import SimpleITK as sitk

from PIL import Image

# nii_path = "/cmach-data/segthor/train"
NUM_CLASSES = 5

def load_data(nii_path, start, end):
  X_train = []
  Y_train = []
  for i in range(start, end + 1):
    print(i)
    ns = str(i).zfill(2)
    data_path = nii_path + "Patient_" + ns + "/Patient_" + ns + ".nii"
    label_path = nii_path + "Patient_" + ns + "/GT.nii"
    if (i == start):
      X_train = nib.load(data_path).get_data().transpose((2, 0, 1))
      Y_train = nib.load(label_path).get_data().transpose((2, 0, 1))
    else:
      X_train = np.concatenate((X_train, nib.load(data_path).get_data().transpose((2, 0, 1))), axis = 0)
      Y_train = np.concatenate((Y_train, nib.load(label_path).get_data().transpose((2, 0, 1))), axis = 0)
  
  X_train = np.reshape(X_train, [-1, 512, 512, 1])
  Y = np.zeros([Y_train.shape[0], 512, 512, 5])
  print("???")
  for i in range(5):
    Y[Y_train == i, i] = 1
  return X_train, Y

def load_test_data(nii_path, start, end):
  X_train = []
  for i in range(start, end + 1):
    ns = str(i).zfill(2)
    data_path = nii_path + "Patient_" + ns + ".nii"
    #label_path = nii_path + "Patient_" + ns + "/GT.nii"
    if (i == start):
      X_train = nib.load(data_path).get_data().transpose((2, 0, 1))
      #Y_train = nib.load(label_path).get_data().transpose((2, 0, 1))
    else:
      X_train = np.concatenate((X_train, nib.load(data_path).get_data().transpose((2, 0, 1))), axis = 0)
      #Y_train = np.concatenate((Y_train, nib.load(label_path).get_data().transpose((2, 0, 1))), axis = 0)
  
  X_train = np.reshape(X_train, [-1, 512, 512, 1])
  return X_train

def load_data_npy(npy_path):
  X_train = np.load(npy_path + "img.npy").transpose((2, 0, 1))
  # Y_train = np.load(npy_path + "gt.npy").transpose((2, 0, 1))
  # # Y = np.zeros([Y_train.shape[0], 512, 512, 5])
  X_train = np.reshape(X_train, [-1, 512, 512, 1])
  # for i in range(5):
  #   Y[Y_train == i, i] = 1
  # Y_train = np.reshape(Y_train, [-1, 512, 512, 1])
  return X_train #, Y_train
  # print(X_train.shape, Y_train.shape)

def load_label_npy(npy_path):
  Y_train = np.load(npy_path + "gt.npy").transpose((2, 0, 1))
  Y = np.zeros([Y_train.shape[0], 512, 512, 5])
  for i in range(5):
    Y[Y_train == i, i] = 1
  # Y_train = np.reshape(Y_train, [-1, 512, 512, 1])
  return Y#_train

def save_result(save_path, predictions):
  for i in range(300):
    label_array = predictions[i] * 255 / 4
    img = Image.fromarray(label_array)
    img.save(save_path + str(i).zfill(2) + ".png")

def next_batch(X, Y, batch_size):
  perm = np.arange(len(X))
  np.random.shuffle(perm)
  return X[perm[:batch_size]], Y[perm[:batch_size]]

  # X_val = []
  # Y_val = []
  # for i in range(36, 41):
  #   ns = str(i)
  #   data_path = nii_path + "Patient_" + ns + "/Patient_" + ns + ".nii"
  #   label_path = nii_path + "Patient_" + ns + "/GT.nii"
  #   if (i == 36):
  #     X_val = nib.load(data_path).get_data().transpose((2, 0, 1))
  #     Y_val = nib.load(label_path).get_data().transpose((2, 0, 1))
  #   else:
  #     X_val = np.concatenate((X_val, nib.load(data_path).get_data().transpose((2, 0, 1))), axis = 0)
  #     Y_val = np.concatenate((Y_val, nib.load(label_path).get_data().transpose((2, 0, 1))), axis = 0)
 
  
  # return X_train, Y_train, X_val, Y_val

# if __name__ == "__main__":
#   X_train, Y_train = load_data("/cmach-data/segthor/train/")
#   print(X_train.shape)
#   print(np.unique(Y_train[150]))

def npy2nii(input, save_dir):
  """
  for transfer input to .nii
  :param input: list which length is same with the eval dataset for segthor
  :return:
  """
  SEGTHOR_JSON_PATH = '/cmach-data/segthor/testset.json'
  f = open(SEGTHOR_JSON_PATH, 'r')
  SEGTHOR_INFO = json.load(f)
  f.close()

  for i in range(41, 61):
    patient_name = "Patient_" + str(i)
    length = SEGTHOR_INFO[patient_name]['length']
    affine = SEGTHOR_INFO[patient_name]['affine']
    img = input[:length]
    input = input[length:]
    # img = np.concatenate(img, axis=2)
    img = np.transpose(img, (1, 2, 0))
    print(img.shape)
    pred = nib.Nifti1Image(img, np.array(affine))
    nib.save(pred, os.path.join(save_dir, patient_name+'.nii'))
  
def calc_dice(pred, label):
  for i in range(1, 5):
    pr = (pred == i).astype(np.uint8)
    lbl = (label == i).astype(np.uint8)
    img_pred = sitk.GetImageFromArray(pr, isVector=False)
    img_label = sitk.GetImageFromArray(lbl, isVector=False)
    diceCalc = sitk.LabelOverlapMeasuresImageFilter()
    diceCalc.Execute(img_label, img_pred)
    diceCoef = diceCalc.GetDiceCoefficient()
    print(diceCoef)
