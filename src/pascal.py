#torch imports
import torch
import torch.utils.data as data
import torchvision.transforms as transform

# general imports
from PIL import Image
import numpy as np
import os
import random
from tqdm import tqdm


# Original Pascal data doesn't have color coding of the labels
# This small source creates Color map encoding for PASCAL dataset labels https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
def color_map(N=256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])
    return cmap

class PascalVOCLoader(data.Dataset):
    
    def __init__(self, datatype="train", augmentation=True, cropSize=300):
    
      #Replace with the local VOCdevkit path after downloading the data
      self.augmentation = augmentation
      self.cropSize = cropSize
      self.orgSize = 520
      self.datatype = datatype
      self.rootPath = '/content/drive/MyDrive/Personal/OMSCS/DL/FP/VOCdevkit/VOC2012'
      
      imagePath = os.path.join(self.rootPath, 'JPEGImages')
      segPath = os.path.join(self.rootPath, 'SegmentationClass')
      self.classes = 21

      filenamesDir = os.path.join(self.rootPath, 'ImageSets/Segmentation')
      if datatype == 'train':
          fileNames = os.path.join(filenamesDir, 'trainval.txt')
      elif datatype == 'val':
          fileNames = os.path.join(filenamesDir, 'val.txt')
      elif datatype == 'test':
          fileNames = os.path.join(filenamesDir, 'test.txt')
      
      self.images, self.segLabels = self.loadFiles(fileNames, segPath, imagePath)

    def loadFiles(self, fileNames, segPath, imagePath):
      images = []
      segLabels = []
      with open(os.path.join(fileNames), "r") as lines:
          for line in tqdm(lines):
            imageName = os.path.join(imagePath, line.rstrip('\n')+".jpg")
            images.append(imageName)
            segLabelName = os.path.join(segPath, line.rstrip('\n')+".png")
            segLabels.append(segLabelName)

      return images, segLabels

    def __len__(self):
      return len(self.images)

    def __getitem__(self, index):
      image = Image.open(self.images[index]).convert('RGB')
      segLabel = Image.open(self.segLabels[index])

      image, segLabel = self.preprocess(image, segLabel, augment=self.augmentation)
      if self.datatype == 'train':
        image, segLabel = self.preprocess(image, segLabel, augment=False)
      elif self.datatype == 'val' or self.datatype == 'test':
        image, segLabel = self.preprocess(image, segLabel, augment=False)
            
        image = transform.ToTensor()(image)
        segLabel = self.oneHotEncoding(segLabel)
        return image, segLabel

    def preprocess(self, image, segLabel, augment=True):
      wdt, ht = image.size
      if augment:
          # scale the image randomly between 50% to 200%
          scale = random.randint(int(self.orgSize*0.5), int(self.orgSize*2.0))
          if ht > wdt:
              wScale = int((wdt*scale)/(ht+np.finfo(float).eps))
              hScale = scale
          else:
              hScale = int((ht*scale)/(wdt+np.finfo(float).eps))
              wScale = scale
          image = image.resize((wScale, hScale), Image.BILINEAR)
          segLabel = segLabel.resize((wScale, hScale), Image.NEAREST)

          cropSize = self.cropSize
          # random crop crop_size
          wdt, ht = image.size
          startX = random.randint(0, wdt - cropSize)
          startY = random.randint(0, ht - cropSize)
          image = image.crop((startX, startY, startX+cropSize, startY+cropSize))
          segLabel = segLabel.crop((startX, startY, startX+cropSize, startY+cropSize))

          # randomly flip image by a prob of 0.5
          if random.random() < 0.5:
              segLabel = segLabel.transpose(Image.FLIP_LEFT_RIGHT)
              image = image.transpose(Image.FLIP_LEFT_RIGHT)

          segLabel = self.removeBorder(segLabel)

      return image, segLabel

        
    def removeBorder(self, segLabel):
      # Function to remove the object borders from the labels
      segLabel = np.array(segLabel).astype('int32')
      segLabel[segLabel == 255] = 0
      return torch.from_numpy(segLabel).long()

    def oneHotEncoding(self, segLabel):
      return transform.ToTensor()(segLabel)
      one_hot_labels = np.zeros((self.classes, segLabel.size[1], segLabel.size[0]))
      segLabelnp=np.array(segLabel)
      
      for i in range(self.classes):
          label = i*(i == segLabel)
          one_hot_labels[i, :, :] = label
          
      # 3 channel image with one hot encoding for 32 categories
      one_hot_labels = torch.from_numpy(np.array(one_hot_labels)).long()
      return one_hot_labels
    

