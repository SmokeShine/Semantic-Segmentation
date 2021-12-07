import os
import time
import logging
import logging.config
import numpy as np
# from sklearn.metrics import f1_score,roc_auc_score,accuracy_score
import torch
import matplotlib.pyplot as plt

def train(logger,model,device,data_loader,criterion,optimizer,epoch,print_freq=10):
    
    model.train()
    total_loss = 0.0
    for i,(input,target) in enumerate(data_loader):
        input=input.to(device)
        target=target.to(device)
        optimizer.zero_grad()
        output=model(input)
                
        # RuntimeError: "argmax_cuda" not implemented for 'Bool'
        loss=criterion(output,torch.argmax(target,axis=1))
        assert not np.isnan(loss.item()),"Model diverged with loss = NaN"
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        logger.info(f"Epoch: {epoch} \t iteration: {i} \t loss:{loss.item()/data_loader.batch_size}")    

    avg_loss = loss/(i*data_loader.batch_size)
    return avg_loss

# Not added to code till now
def evaluate(logger,model,device,data_loader,criterion,optimizer,epoch,print_freq=10):
    
    model.eval()
    with torch.no_grad():
        for i,(input,target) in enumerate(data_loader):
            input=input.to(device)
            target=target.to(device)
            optimizer.zero_grad()
            output=model(input)
            loss=criterion(output,target)

def save_checkpoint(logger, model, optimizer, path):
    state = {'model': model.state_dict, 'optimizer': optimizer.state_dict}
    torch.save(state, path)
    logger.info(f"checkpoint saved at {path}")
        