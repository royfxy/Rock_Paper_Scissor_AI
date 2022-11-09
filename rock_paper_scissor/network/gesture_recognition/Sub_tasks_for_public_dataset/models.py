# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:49:15 2022

@author: Administrator
"""
import numpy as np
import pandas as pd 
import os
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset,random_split
from torchvision import transforms
import copy
from tqdm import tqdm
from PIL import Image
import random
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torchvision.models import resnet18


class MLP_Hand(nn.Module):
    
    """MLP encoder module."""
    def __init__(self, n_in, n_hid, n_out):
        super(MLP_Hand, self).__init__()
        self.dropout_0 = nn.Dropout(p = 0.2)
        self.dropout_1 = nn.Dropout(p = 0.4)
        self.fc1 = nn.Linear(n_in, n_hid, bias = True)
        self.dropout_2 = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(n_hid, 64, bias = True)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(64,n_out, bias = True)
        self.softmax = nn.Softmax(dim=1)
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_1(x)

        x = self.fc2(x)
        x = self.dropout_2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)

        return x
    

