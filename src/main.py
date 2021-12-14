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
from plots import plot_loss_curve
from utils import train, evaluate, save_checkpoint, get_color_transforms, get_resize_transforms
import random
import mymodels
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Subset
from torchvision.utils import save_image
from tqdm import tqdm

logging.config.fileConfig("logging.ini")
logger = logging.getLogger()
torch.manual_seed(1)
random.seed(1)

def data_preprocessing():
    pass


def split_data(split_type):
    return 


class SequenceWithLabelDataset(Dataset):
    def __init__(self, images_file, labels_file, num_categories,pixel_classes,transform=None):
        self.images_file = images_file
        self.labels_file = labels_file
        self.num_categories = num_categories
        self.list_of_images = os.listdir(images_file)
        self.list_of_labels = os.listdir(labels_file)
        self.pixel_classes=pixel_classes
        self.transform=transform
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
        if self.transform:
            img=self.transform(img)
        image_raw = self.convert_tensor(img)
        # Attach Label
        # remove png and add L
        temp = self.list_of_images[index]
        img_lbl = Image.open(f"{self.labels_file}/{temp[:-4]}_L.png")
        
        one_hot_labels = np.zeros((img_lbl.size[1], img_lbl.size[0],self.num_categories))
        img_lbl_np=np.array(img_lbl)
        for i in range(self.num_categories):
            # possible bug area
            label = np.nanmin(self.pixel_classes[i] == img_lbl_np,axis=2)
            one_hot_labels[:, :, i] = label
        # 3 channel image with one hot encoding for 32 categories
        one_hot_labels = self.convert_tensor(one_hot_labels)
        
        return (image_raw, one_hot_labels,temp)

def predict_model(best_model,images_file, labels_file, pixel_classes):
    logger.info("Generating DataLoader For Prediction")
    images_dataset = SequenceWithLabelDataset(
        images_file, labels_file, num_categories=len(pixel_classes),pixel_classes=pixel_classes)
    pred_loader = DataLoader(
        dataset=images_dataset, batch_size=1,shuffle=False,num_workers=0)
    # No need to calculate grad as it is forward pass only
    best_model.eval()
    with torch.no_grad():
        for counter,(input,target,img_name) in tqdm(enumerate(pred_loader)):
            # Model is in GPU
            input=input.to(DEVICE)
            target=target.to(DEVICE)
            # which pixel belongs to which object, etc. 
            # assign a class to each pixel of the image. 
            output=best_model(input)
            # Output is 32 classes and we need to collapse back to 1
            # import pdb;pdb.set_trace()
            expected_width=output.shape[2]
            expected_height=output.shape[3]
            temp_image=torch.zeros((3,expected_width,expected_height))
            logging.info(f"Image is {img_name}")
            torch_pixel_classes=torch.from_numpy(pixel_classes)
            for i in range(expected_width):
                for j in range(expected_height):
                    temp_image[:,i,j]=torch_pixel_classes[torch.argmax(output[0,:,i,j])]
            # import pdb;pdb.set_trace()
            # https://discuss.pytorch.org/t/convert-float-image-array-to-int-in-pil-via-image-fromarray/82167/4
            temp_image=temp_image.permute(1,2,0).numpy().astype(np.uint8)
            save_image(input,f'./predictions/actual_{counter}.png')
            transforms.ToPILImage()(temp_image).save(f'./predictions/pred_{counter}_{img_name}.png')
            break

def train_model(images_file, labels_file, pixel_classes, model_name = 'SegNet'):

    logger.info("Generating DataLoader")
    train_images_dataset = SequenceWithLabelDataset(
        images_file, labels_file, num_categories=len(pixel_classes),pixel_classes=pixel_classes, transform= TRANSFORM)
    valid_images_dataset = SequenceWithLabelDataset(
        images_file, labels_file, num_categories=len(pixel_classes),pixel_classes=pixel_classes)
    split=int(SPLIT_PERCENTAGE*len(train_images_dataset))
    train_dataset=Subset(train_images_dataset,range(split,len(train_images_dataset)))
    valid_dataset=Subset(valid_images_dataset,range(split))
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(
        dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    logger.info("Training 1st Model")

    if model_name == 'Vanilla_SegNet':
        model = mymodels.Vanilla_SegNet(NUM_OUTPUT_CLASSES)
        save_file = 'Vanilla_SegNet.pth'
    elif model_name == 'SegNet':
        model = mymodels.SegNet(NUM_OUTPUT_CLASSES)
        save_file = 'SegNet.pth'
    else:
        sys.exit("Model Not Available")

    # Initialized Optimizer before load_state_dict
    # Bug Fix - Checkpointing loading for gpu
    model.to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=SGD_MOMENTUM)
    if MODEL_PATH != None:
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"Loaded Checkpoint from {MODEL_PATH}")

    logger.info(model)
    early_stopping_counter=0
    
    criterion = nn.CrossEntropyLoss()
    criterion.to(DEVICE)
    train_loss_history=[]
    valid_loss_history=[]
    best_validation_loss=float("inf")
    for epoch in range(1, NUM_EPOCHS + 1):
        logger.info(f"Epoch {epoch}")
        train_loss = train(
            logger, model, DEVICE, train_loader, criterion, optimizer, epoch)
        logger.info(f"Average Loss for epoch {epoch} is {train_loss}")
        train_loss_history.append(train_loss)
        valid_loss = evaluate(logger, model, DEVICE, valid_loader, criterion,optimizer)
        valid_loss_history.append(valid_loss)
        is_best=best_validation_loss>valid_loss
        logger.info(f"Current Epoch loss is better : {is_best} \t Old Loss: {best_validation_loss}  vs New loss:{valid_loss}")
        if epoch % EPOCH_SAVE_CHECKPOINT == 0:
            logger.info(f"Saving Checkpoint for {model_name} at epoch {epoch}")
            save_checkpoint(logger, model, optimizer, save_file + "_" + str(epoch) + ".tar")
        # From BD4H Piazza
        # https://piazza.com/class/ki87klxs9yite?cid=397_f2
        if is_best:
            early_stopping_counter=0
            logger.info(f"New Best Identified: \t Old Loss: {best_validation_loss}  vs New loss:\t{valid_loss} ")
            best_validation_loss=valid_loss
            torch.save(model,'./best_model.pth',_use_new_zipfile_serialization=False)
        else:
            logger.info("Loss didnot improve")
            early_stopping_counter+=1
        if early_stopping_counter >= PATIENCE:
            break
    
    # final checkpoint saved
    save_checkpoint(logger, model, optimizer, save_file + ".tar")
    # Loading Best Model
    best_model=torch.load("./best_model.pth")
    
    # plot loss curves
    logger.info(f"Plotting Charts")
    logger.info(f"Train Losses:{train_loss_history}")
    logger.info(f"Validation Losses:{valid_loss_history}")
    plot_loss_curve(logger,model_name, train_loss_history, valid_loss_history,"Loss Curve", f"{PLOT_OUTPUT_PATH}loss_curves.jpg")
    logger.info(f"Training Finished for {model_name}")


def load_pickle(seqs_file, labels_file):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Final Group Project - Semantic Segmentation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", action='store_true',
                        default=True, help="Use GPU for training")
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
    parser.add_argument("--plot_output_path", default='./Plots_', 
                        help="Output path for Plot")
    parser.add_argument("--model_path", help="Model Path to resume training")
    parser.add_argument("--epoch_save_checkpoint", nargs='?', type=int,
                        default=5, help="Epochs after which to save model checkpoint")
    parser.add_argument("--split_percentage", nargs='?', type=float,
                        default=0.1, help="Train and Validation data split")
    parser.add_argument("--patience", nargs='?', type=int,
                        default=3, help="Early stopping epoch count")
    parser.add_argument("--transforms", default="sharpness,contrast,equalize,crop,hflip" , 
                        help="Transforms to be applied to input dataset. \
                        options (posterize,sharpness,contrast,equalize,crop,hflip). \
                        comma-separated list of transforms.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logger.info(args)
    global BATCH_SIZE, USE_CUDA,\
        NUM_EPOCHS, NUM_WORKERS,\
        LEARNING_RATE, SGD_MOMENTUM,\
        DEVICE,SPLIT_PERCENTAGE,\
        TRANSFORM,PATIENCE
    __train__ = args.train
    BATCH_SIZE = args.batch_size
    USE_CUDA = args.gpu
    NUM_WORKERS = args.num_workers
    NUM_EPOCHS = args.num_epochs
    NUM_OUTPUT_CLASSES = args.num_output_classes
    LEARNING_RATE = args.learning_rate
    SGD_MOMENTUM = args.sgd_momentum
    PLOT_OUTPUT_PATH = args.plot_output_path
    EPOCH_SAVE_CHECKPOINT = args.epoch_save_checkpoint
    MODEL_PATH = args.model_path
    SPLIT_PERCENTAGE=args.split_percentage
    PATIENCE=args.patience
    TRANSFORM = transforms.Compose(get_color_transforms(logger, str(args.transforms)))
    DEVICE = torch.device(
        "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    images_file = '../data/raw/701_StillsRaw_full'
    labels_file = '../data/raw/LabeledApproved_full'
    
    pixel_classes = pd.read_csv(
        '../data/raw/classes.txt', header=None, usecols=[0, 1, 2], delim_whitespace=True).values
    if DEVICE.type == "cuda":
        logger.info("Settings for Cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if __train__:
        logger.info("Training Segnet")
        train_model(images_file, labels_file, pixel_classes)
    else:
        best_model=torch.load("./best_model.pth")
        # Predict on New Images
        predict_model(best_model,images_file, labels_file, pixel_classes)
        logger.info("Prediction Step Complete")