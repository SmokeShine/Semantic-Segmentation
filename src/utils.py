import os
import time
import logging
import logging.config
import numpy as np
# from sklearn.metrics import f1_score,roc_auc_score,accuracy_score
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


# Boiler Plate Code From BD4H and DL Class for recording metrics
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(logger,model,device,data_loader,criterion,optimizer,epoch,print_freq=50):
    
    model.train()
    losses = AverageMeter()
    for i,(input,target) in enumerate(data_loader):
        input=input.to(device)
        target=target.to(device)
        optimizer.zero_grad()
        output=model(input)
                
        # RuntimeError: "argmax_cuda" not implemented for 'Bool'
        # Causing issue in aws
        loss=criterion(output,torch.argmax(target,axis=1))
        assert not np.isnan(loss.item()),"Model diverged with loss = NaN"
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), target.size(0))
        if i%print_freq==0:
            logger.info(f"Epoch: {epoch} \t iteration: {i} \t Training Loss Current:{losses.val:.4f} Average:({losses.avg:.4f})")    

    return losses.avg

def evaluate(logger,model,device,data_loader,criterion,optimizer,print_freq=10):
    losses = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i,(input,target) in enumerate(data_loader):
            input=input.to(device)
            target=target.to(device)
            optimizer.zero_grad()
            output=model(input)
            loss=criterion(output,torch.argmax(target,axis=1))
            losses.update(loss.item(), target.size(0))
            if i%print_freq==0:
                logger.info(f"Validation Loss Current:{losses.val:.4f} Average:({losses.avg:.4f})")
    return losses.avg

def save_checkpoint(logger, model, optimizer, path):
    state = {'model': model.state_dict, 'optimizer': optimizer.state_dict}
    torch.save(state, path)
    logger.info(f"checkpoint saved at {path}")
        
# From Abhinav's commit
def get_resize_transforms(logger, transformsStr):
    tranformsList = []
    transformsAdded = ""
    tranforms = transformsStr.split(",")
    if 'crop' in [x.lower() in tranforms]:
        tranformsList.append(transforms.RandomCrop(224))
        transformsAdded += 'crop,'
    if 'hflip' in [x.lower() in tranforms]:
        tranformsList.append(transforms.RandomHorizontalFlip(p=0.5))
        transformsAdded += 'hflip,'

    logger.info(f"resize transforms added - {transformsAdded}")  
    return tranformsList

def get_color_transforms(logger, transformsStr):
    tranformsList = []
    transformsAdded = ""
    tranforms = [x.lower() for x in transformsStr.split(",")]
    if 'posterize' in tranforms:
        tranformsList.append(transforms.RandomPosterize(bits=2))
        transformsAdded += 'posterize,'
    if 'sharpness' in tranforms:
        tranformsList.append(transforms.RandomAdjustSharpness(sharpness_factor=2))
        transformsAdded += 'sharpness,'
    if 'contrast' in tranforms:
        tranformsList.append(transforms.RandomAutocontrast())
        transformsAdded += 'contrast,'
    if 'equalize' in tranforms:
        tranformsList.append(transforms.RandomEqualize())
        transformsAdded += 'equalize,'

    logger.info(f"color transforms added - {transformsAdded}")  
    return tranformsList