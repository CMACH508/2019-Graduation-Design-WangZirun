import os
import json
import numpy as np
import nibabel as nib



def numpy2nifty(input, patient_names, save_dir):
    """
    for transfer input to .nii
    :param input: list which length is same with the eval dataset for segthor
    :return:
    """
    SEGTHOR_JSON_PATH = '/home/guoyuze/Lmser-S/Data/segthor/trainset.json'
    f = open(SEGTHOR_JSON_PATH, 'r')
    SEGTHOR_INFO = json.load(f)
    f.close()

    for patient_name in patient_names:
        length = SEGTHOR_INFO[patient_name]['length']
        affine = SEGTHOR_INFO[patient_name]['affine']
        img = input[:length]
        input = input[length:]
        img = np.concatenate(img, axis=2)
        pred = nib.Nifti1Image(img, np.array(affine))
        nib.save(pred, os.path.join(save_dir, patient_name+'.nii'))


