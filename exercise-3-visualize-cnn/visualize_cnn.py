# -*- coding: utf-8 -*-
"""
Created on 

@author: ziaul
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum


if __name__=='__main__':
    
    # Visualize CNN
    