# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:12:08 2022

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
import cv2
import mediapipe as mp
import time
from helperfunctions import *
import re

dirname = os.path.dirname(__file__)
data_dir = os.path.join(dirname+'\data\hand_new')
data_files = os.listdir(data_dir)

mpHands = mp.solutions.hands

hands = mpHands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7,
)

class HandDataset(Dataset):    
    def __init__(self,file_list,dir,hands = None):
        
        self.file_list = file_list
        self.hands = hands
        self.dir = dir    
        
    # #dataset length
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
    
    #load an one of images
    def __getitem__(self,idx):
        #
        
        img_name = self.file_list[idx]
        img_path_combine = os.path.join(self.dir, self.file_list[idx])
        # print(idx)
        # print(img_name)
        # print(img_path_combine)
        # img = Image.open(img_path_combine)
        img = cv2.flip(cv2.imread(img_path_combine), 1)
        # print(img.shape)
        
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # print(results.multi_hand_landmarks)
        pre_processed_landmark_list = None
        label = None
        # print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks is not None:
            landmark_list,landmark_list_xyz= calc_landmark_list_train(img, results.multi_hand_landmarks[0],'xyz')
            pre_processed_landmark_list = pre_process_landmark_xyz(landmark_list_xyz)
            pre_processed_landmark_list = torch.FloatTensor(pre_processed_landmark_list)
        # img_transformed = self.transform(img)
            label = img_name.split('.')[0]
            label = re.split('(\d+)',label)[0]
            # print(label)
        
            if label == 'fist':
                label = 0
            elif label == 'five':
                label = 1
            elif label == 'four':
                label = 2
            elif label == 'heratSingle':
                label = 3
            elif label == 'iloveu':
                label = 4
            elif label == 'nine':
                label = 5
            elif label == 'ok':
                label = 6
            elif label == 'one':
                label = 7
            elif label == 'pink':
                label = 8
            elif label == 'shoot':
                label = 9
            elif label == 'six':
                label = 10
            elif label == 'three':
                label = 11
            elif label == 'thumbup':
                label = 12
            elif label == 'yearh':
                label = 13

            
        return pre_processed_landmark_list , label


# hand_data_set = HandDataset(all_data_set, data_dir, hands = hands)




