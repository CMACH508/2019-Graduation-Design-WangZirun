import numpy as np
import cv2
from PIL import Image

for i in range(30):
  im = Image.open('./result_adam1500dropout1/' + str(i) + '.png')
  img = np.asarray(im)

  kernel_ed = cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))
  img = cv2.erode(img,kernel_ed,2)
  img = cv2.dilate(img,kernel_ed,2)

  imgsave = Image.fromarray(img)
  imgsave.save('./result_adam1500dropout1/adam1500dropout_postprc/' + str(i) + '.png')