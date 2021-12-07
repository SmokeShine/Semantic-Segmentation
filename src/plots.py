import os
import time
import logging
import logging.config
import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_loss_curve(logger, train_loss_history, title, output_path):
    
    plt.plot(train_loss_history, label= "train")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("title")
    plt.legend(loc="upper right")
    plt.savefig(output_path)
    logger.info(f"loss curve saved at {output_path}.")
