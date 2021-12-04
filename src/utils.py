import os
import time
import logging
import logging.config
import numpy as np
from sklearn.metrics import f1_score,roc_auc_score,accuracy_score
import torch

def train(logger,model,device,data_loader,criterion,optimizer,epoch,print_freq=10):
    
    model.train()
    for i,(input,target) in enumerate(data_loader):
        input=input.to(device)
        target=target.to(device)
        optimizer.zero_grad()
        output=model(output)
        loss=criterion(output,target)
        assert not np.isnan(loss.item()),"Model diverged with loss = NaN"
        loss.backward()
        optimizer.step()

def evaluate(logger,model,device,data_loader,criterion,optimizer,epoch,print_freq=10):
    
    model.eval()
    with torch.no_grad():
        for i,(input,target) in enumerate(data_loader):
            input=input.to(device)
            target=target.to(device)
            optimizer.zero_grad()
            output=model(output)
            loss=criterion(output,target)
        