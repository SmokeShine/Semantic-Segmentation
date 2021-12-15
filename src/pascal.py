import os
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm

import torch
from base import BaseDataset

#Color map encoding for PASCA dataset https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
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
    
    def __init__(self, datatype="train", augmentation=True, cropSize=500):

        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle','bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train', 'tv/monitor', 'ambigious']
    
        #Replace with the local VOCdevkit path after downloading the data
        self.augmentation = augmentation
        self.cropSize = cropSize
        self.orgSize = 520
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

    def loadFiles(fileNames, segPath, imagePath):
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
        if self.mode == 'train':
            image, segLabel = self.preprocess(image, segLabel, augment=False)
        elif self.mode == 'val' or self.mode == 'test':
            image, segLabel = self.preprocess(image, segLabel, augment=False)
            
        image = torch.from_numpy(image).long()
        segLabel = self.oneHotEncoding(segLabel)
        return image, segLabel

    def preprocess(self, image, segLabel, augment=True):
        w, h = image.size
        if augment:
            # randomly flip image by a prob of 0.5
            if random.random() < 0.5:
                segLabel = segLabel.transpose(Image.FLIP_LEFT_RIGHT)
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            cropSize = self.cropSize
            
            # scale the image randomly between 50% to 200%
            scale = random.randint(int(self.orgSize*0.5), int(self.orgSize*2.0))
            if h > w:
                wScale = int((w*scale)/(h+0.5))
                hScale = scale
            else:
                hScale = int((h*scale)/(w+0.5))
                wScale = scale
            image = image.resize((wScale, hScale), Image.BILINEAR)
            segLabel = segLabel.resize((wScale, hScale), Image.NEAREST)

            # # pad crop
            # if short_size < cropSize:
            #     padh = cropSize - oh if oh < cropSize else 0
            #     padw = cropSize - ow if ow < cropSize else 0
            #     image = ImageOps.expand(image, border=(0, 0, padw, padh), fill=0)
            #     segLabel = ImageOps.expand(segLabel, border=(0, 0, padw, padh), fill=0)

            # random crop crop_size
            w, h = image.size
            x1 = random.randint(0, w - cropSize)
            y1 = random.randint(0, h - cropSize)
            image = image.crop((x1, y1, x1+cropSize, y1+cropSize))
            segLabel = segLabel.crop((x1, y1, x1+cropSize, y1+cropSize))

            segLabel = self.removeBorder(segLabel)

        return image, segLabel

        
    def removeBorder(self, segLabel):
        segLabel = np.array(segLabel).astype('int32')
        segLabel[segLabel == 255] = 0
        return torch.from_numpy(segLabel).long()

    def oneHotEncoding(self, segLabel):
        one_hot_labels = np.zeros((self.NUM_CLASS, segLabel.size()[1], segLabel.size()[0]))
        segLabelnp=np.array(segLabel)
       
        for i in range(self.NUM_CLASS):
            label = i*(i == segLabel)
            one_hot_labels[i, :, :] = label
            
        # 3 channel image with one hot encoding for 32 categories
        one_hot_labels = torch.from_numpy(np.array(one_hot_labels)).long()
        return one_hot_labels
    

