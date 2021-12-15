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
from utils import train, evaluate, save_checkpoint
import mymodels
import torchvision.transforms as transform
from PIL import Image
from torch.utils.data import Subset
from pascal_ import PascalVOCLoader, color_map
from tqdm import tqdm
from torchvision.utils import save_image

logging.config.fileConfig("src/logging.ini")
logger = logging.getLogger()
torch.manual_seed(1)


def data_preprocessing():
    pass


def split_data(split_type):
    return 


def predict_model(best_model,images_file, labels_file, pixel_classes):
    logger.info("Generating DataLoader For Prediction")
    testset = PascalVOCLoader(datatype="test", augmentation=False, cropSize=300)
    # dataloader
    kwargs = {'num_workers': NUM_WORKERS, 'pin_memory': True}
    test_loader = DataLoader(testset, batch_size=1,
                                       drop_last=True, shuffle=True, **kwargs)
    cmap = color_map()
    print(enumerate(test_loader))
    # No need to calculate grad as it is forward pass only
    best_model.eval()
    with torch.no_grad():
        for counter,(input,target) in tqdm(enumerate(test_loader)):
            print("predicting..", counter)
            # Model is in GPU
            input=input.to(DEVICE)
            # target=target.to(DEVICE)
            # which pixel belongs to which object, etc. 
            # assign a class to each pixel of the image. 
            output=best_model(input)
            # Output is 32 classes and we need to collapse back to 1
            # import pdb;pdb.set_trace()
            expected_width=output.shape[2]
            expected_height=output.shape[3]
            temp_image=torch.zeros((3,expected_width,expected_height))
            squeezed_output=output[0]
            torch_pixel_classes=torch.from_numpy(cmap)
            for i in range(expected_width):
                for j in range(expected_height):
                    temp_image[:,i,j]=torch_pixel_classes[torch.argmax(output[0,:,i,j])]
            # import pdb;pdb.set_trace()
            # https://discuss.pytorch.org/t/convert-float-image-array-to-int-in-pil-via-image-fromarray/82167/4
            temp_image=temp_image.permute(1,2,0).numpy().astype(np.uint8)
            save_image(input,f'./predictions/actual_{counter}.png')
            transforms.ToPILImage()(temp_image).save(f'./predictions/pred_{counter}.png')
            # break

def train_model(images_file, labels_file, pixel_classes, model_name = 'SegNet'):

    logger.info("Generating DataLoader")   
    trainset = PascalVOCLoader(datatype="train", augmentation=True, cropSize=300)
    testset = PascalVOCLoader(datatype="val", augmentation=False, cropSize=300)
    # dataloader
    kwargs = {'num_workers': NUM_WORKERS, 'pin_memory': True}
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE,
                                       drop_last=True, shuffle=True, **kwargs)
    valid_loader = DataLoader(testset, batch_size=BATCH_SIZE,
                                     drop_last=False, shuffle=False, **kwargs)
        
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

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=SGD_MOMENTUM)
    if MODEL_PATH != None:
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint['model'])
        model.to(DEVICE)
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"Loaded Checkpoint from {MODEL_PATH}")

    logger.info(model)
    early_stopping_counter=0
    model.to(DEVICE)
    # optimizer.to(DEVICE)
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
                        default=16, help="Batch size for training the model")
    parser.add_argument("--num_workers", nargs='?', type=int,
                        default=5, help="Number of Available CPUs")
    parser.add_argument("--num_epochs", nargs='?', type=int,
                        default=2, help="Number of Epochs for training the model")
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
                        default=4, help="Early stopping epoch count")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logger.info(args)
    global BATCH_SIZE, USE_CUDA,\
        NUM_EPOCHS, NUM_WORKERS,\
        LEARNING_RATE, SGD_MOMENTUM,\
        DEVICE,SPLIT_PERCENTAGE,PATIENCE
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
    DEVICE = torch.device(
        "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    images_file = '../data/raw/701_StillsRaw_full'
    labels_file = '../data/raw/LabeledApproved_full'
    
    # pixel_classes = pd.read_csv(
    #     '../data/raw/classes.txt', header=None, usecols=[0, 1, 2], delim_whitespace=True).values
    pixel_classes = None
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
        # Predict on New Images
        predict_model(best_model,images_file, labels_file, pixel_classes)
        logger.info("Prediction Step Complete")