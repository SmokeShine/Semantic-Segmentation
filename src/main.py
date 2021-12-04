#!/usr/bin/env python

import argparse
import collections
import logging
import logging.config
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from utils import train, evaluate
import mymodels

logging.config.fileConfig("logging.ini")
logger = logging.getLogger()
torch.manual_seed(1)

def data_preprocessing():
    pass

def load_data():
    pass

def generated_dataset_splits(split_name):
    pass

def new_run(train_model, seqs_file, labels_file):
    logger.info("Training 1st Model")
    train_model(seqs_file, labels_file,model_name='Vanilla_Segnet'


def train_model(seqs_file,labels_file,model_name):
    if model_name == 'Vanilla_SegNet':
        model = mymodels.Vanilla_SegNet()
        save_file = 'Vanilla_SegNet.pth'
    logger.info(model)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,momentum=SGD_MOMENTUM)
    train_dataset=None
    test_dataset=None
    valid_dataset=None
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    for epoch in range(NUM_EPOCHS):
        logger.info(f"Epoch {epoch}")
        train_loss, train_package = train(
            logger, model, device, train_loader, criterion, optimizer, epoch)
        valid_loss, valid_package = train(
            logger, model, device, valid_loader, criterion, optimizer, epoch)

    test_loss, test_package, test_results = evaluate(
        logger, model, device, test_loader, criterion)

    return None

def load_pickle(seqs_file,labels_file):
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
                        default=10, help="Number of output class for semantic segmentation")
    parser.add_argument("--learning_rate", nargs='?', type=float,
                        default=0.01, help="Learning Rate for the optimizer")
    parser.add_argument("--sgd_momentum", nargs='?', type=float,
                        default=0.5, help="Momentum for the SGD Optimizer")
    return parser.parse_args()

if __name__=='__main__':
    args=parse_args()
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
    device=torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if __train__:
        logger.info("Training Segnet")
        new_run(train_model, seqs_file, labels_file)