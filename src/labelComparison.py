import os
import sys
import numpy as np
from PIL import Image
from matplotlib import cm


SOURCE_PATH = '/content/drive/MyDrive/Personal/OMSCS/DL/FP/data/raw/LabeledApproved_full'
TARGET_PATH = '/content/drive/MyDrive/Personal/OMSCS/DL/FP/predictions'

#CAR - (64,0,128)
#PERSON - (0,128,192) or (64, 64, 0) or (192,128,64)
#BIKE - (0,128,192) or (192, 0, 192)
#TRAIN - (192,64,128)
SOURCE_OBJECT_COLOR = (192,128,64)

#CAR - (128, 128, 128)
#PERSON - (192, 128, 128)
#BIKE - (64,128,128)
#TRAIN - (128,192,0)
TARGET_OBJECT_COLOR = (128,192,0)


def getIou():
  sourceList = os.listdir(SOURCE_PATH)
  targetList = os.listdir(TARGET_PATH)

  mIou = 0.0
  validImgs = 0

  for index in range(len(sourceList)):
    imgName = sourceList[index]
    assert os.path.isfile(f"{SOURCE_PATH}/{sourceList[index]}")

    try:
      gtLabel = Image.open(f"{SOURCE_PATH}/{sourceList[index]}")
      predLabel = Image.open(f"{TARGET_PATH}/{imgName}")
    except:
      continue

    # print(sourceList[index])

    gtLabel = np.array(gtLabel)
    predLabel = np.array(predLabel)

    gtLabel = (gtLabel == SOURCE_OBJECT_COLOR)[:,:,0]
    predLabel = (predLabel == TARGET_OBJECT_COLOR)[:,:,0]
    check = Image.fromarray(np.uint8(cm.gist_earth(gtLabel)*255))
    check.save('out.png')

    # print(gtLabel.shape)
    # print(predLabel.shape)

    if np.count_nonzero(gtLabel):
      validImgs += 1

    # intersection = gtLabel and predLabel
    intersection = np.count_nonzero(np.logical_and(gtLabel, predLabel))
    union = np.count_nonzero(np.logical_or(gtLabel, predLabel))
    # union = gtLabel or predLabel

    iou = intersection/(union+np.finfo(float).eps)
    # print(iou)
    mIou += iou

    # break

  
  mIou = mIou/validImgs
  print("Mean IOU: ", mIou)

getIou()

