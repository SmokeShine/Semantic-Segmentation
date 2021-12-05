#!/usr/bin/env python

import argparse
import collections
import logging
import logging.config
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from utils import train, evaluate
import mymodels
import torchvision.transforms as transforms
from PIL import Image

logging.config.fileConfig("logging.ini")
logger = logging.getLogger()
torch.manual_seed(1)


def data_preprocessing():
    pass


def load_data():
    pass


class SequenceWithLabelDataset(Dataset):
    def __init__(self, images_file, labels_file, num_categories,pixel_classes):
        self.images_file = images_file
        self.labels_file = labels_file
        self.num_categories = num_categories
        self.list_of_images = os.listdir(images_file)
        self.list_of_labels = os.listdir(labels_file)
        self.pixel_classes=pixel_classes
        self.convert_tensor = transforms.ToTensor()
        if len(self.list_of_images) != len(self.list_of_labels):
            raise ValueError("Count of Segmentation Images do not match with that of labels")

    def __len__(self):
        return len(self.list_of_images)

    def __getitem__(self, index):
        # https://stackoverflow.com/questions/50981714/multi-label-multi-class-image-classifier-convnet-with-pytorch
        # https://stackoverflow.com/questions/56123419/how-to-cover-a-label-list-under-the-multi-label-classification-context-into-one

        # https://www.projectpro.io/recipes/convert-image-tensor-pytorch
        # Get image path
        img = Image.open(f"{self.images_file}/{self.list_of_images[index]}")
        image_raw = self.convert_tensor(img)
        # Attach Label
        # remove png and add L
        temp = self.list_of_images[index]
        img_lbl = Image.open(f"{self.labels_file}/{temp[:-4]}_L.png")
        
        one_hot_labels = np.zeros((img_lbl.size[1], img_lbl.size[0],self.num_categories))
        img_lbl_np=np.array(img_lbl)
        for i in range(self.num_categories):
            # import pdb;pdb.set_trace()
            # possible bug area
            label = np.nanmin(self.pixel_classes[i] == img_lbl_np,axis=2)
            one_hot_labels[:, :, i] = label
        # 3 channel image with one hot encoding for 32 categories
        one_hot_labels = self.convert_tensor(one_hot_labels)
        
        return (image_raw, one_hot_labels)


def new_run(train_model, images_file, labels_file, pixel_classes):
    logger.info("Generating DataLoader")
    train_dataset = SequenceWithLabelDataset(
        images_file, labels_file, num_categories=len(pixel_classes),pixel_classes=pixel_classes)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    logger.info("Training 1st Model")
    # List of Model to train
    train_model(train_loader, model_name='SegNet')


def train_model(loader, model_name):
    if model_name == 'Vanilla_SegNet':
        model = mymodels.Vanilla_SegNet(NUM_OUTPUT_CLASSES)
        save_file = 'Vanilla_SegNet.pth'
    elif model_name == 'SegNet':
        model = mymodels.SegNet(NUM_OUTPUT_CLASSES)
        save_file = 'SegNet.pth'
    else:
        sys.exit("Model Not Available")
    logger.info(model)
    optimizer = optim.SGD(model.parameters(),
                          lr=LEARNING_RATE, momentum=SGD_MOMENTUM)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    for epoch in range(NUM_EPOCHS):
        logger.info(f"Epoch {epoch}")
        train_loss, train_package = train(
            logger, model, device, loader, criterion, optimizer, epoch)
        
    logger.info(f"Training Finished for {model_name}")


def load_pickle(seqs_file, labels_file):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Final Group Project - Semantic Segmentation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", action='store_true',
                        default=False, help="Use GPU for training")
    parser.add_argument("--train", action='store_true',
                        default=False, help="Train Model")
    parser.add_argument("--batch_size", nargs='?', type=int,
                        default=1, help="Batch size for training the model")
    parser.add_argument("--num_workers", nargs='?', type=int,
                        default=5, help="Number of Available CPUs")
    parser.add_argument("--num_epochs", nargs='?', type=int,
                        default=10, help="Number of Epochs for training the model")
    parser.add_argument("--num_output_classes", nargs='?', type=int,
                        default=32, help="Number of output class for semantic segmentation")
    parser.add_argument("--learning_rate", nargs='?', type=float,
                        default=0.01, help="Learning Rate for the optimizer")
    parser.add_argument("--sgd_momentum", nargs='?', type=float,
                        default=0.5, help="Momentum for the SGD Optimizer")
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logger.info(args)
    global BATCH_SIZE, USE_CUDA,\
        NUM_EPOCHS, NUM_WORKERS,\
        LEARNING_RATE, SGD_MOMENTUM,\
        device
    __train__ = args.train
    BATCH_SIZE = args.batch_size
    USE_CUDA = args.gpu
    NUM_WORKERS = args.num_workers
    NUM_EPOCHS = args.num_epochs
    NUM_OUTPUT_CLASSES = args.num_output_classes
    LEARNING_RATE = args.learning_rate
    SGD_MOMENTUM = args.sgd_momentum
    device = torch.device(
        "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    images_file = '../data/raw/701_StillsRaw_full'
    labels_file = '../data/raw/LabeledApproved_full'
    pixel_classes = pd.read_csv(
        '../data/raw/classes.txt', header=None, usecols=[0, 1, 2], delim_whitespace=True).values
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if __train__:
        logger.info("Training Segnet")
        new_run(train_model, images_file, labels_file, pixel_classes)
