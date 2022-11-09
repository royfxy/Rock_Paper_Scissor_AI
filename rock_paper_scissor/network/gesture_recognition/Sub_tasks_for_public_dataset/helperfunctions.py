# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:50:35 2022

@author: Administrator
"""

import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
import time
from tqdm import tqdm
import torch

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

from read_data import *
from helperfunctions import *
from torch.utils.data.dataloader import default_collate

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    return number, mode

def save_data_csv(number,mode,landmark_list,temp):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint_spc_xyz.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
            temp += 1
            print('pass save',temp)
    return temp

def Hand_box_coordinates(image, landmarks):
    
    imgHeight = image.shape[0]
    
    imgWeight = image.shape[1]

    landmark_list = np.empty((0, 2), int)

    for i, landmark in enumerate(landmarks.landmark):
        
        xPos = min(int(landmark.x * imgWeight), imgWeight - 1)
        yPos = min(int(landmark.y * imgHeight), imgHeight - 1)

        landmark_Pos = [np.array((xPos, yPos))]

        landmark_list = np.append(landmark_list, landmark_Pos, axis=0)

    x, y, w, h = cv.boundingRect(landmark_list)
    
    # the bounding box coordinates
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    
    imgHeight = image.shape[0]
    imgWeight = image.shape[1]

    landmark_list = []


    for i, landmark in enumerate(landmarks.landmark):
        
        xPos = min(int(landmark.x * imgWeight), imgWeight - 1)
        yPos = min(int(landmark.y * imgHeight), imgHeight - 1)
        # landmark_z = landmark.z

        landmark_list.append([xPos, yPos])

    return landmark_list

def pre_process_landmark(landmark_list):
    
    temp_landmark_list = []

    # Convert to relative coordinates
    origin_x = None
    origin_y = None
    
    for idx, landmark_point in enumerate(landmark_list):
        
        if idx == 0:
            origin_x = landmark_point[0]
            origin_y = landmark_point[1]

        xPos = landmark_list[idx][0] - origin_x
        yPos = landmark_list[idx][1] - origin_y
        
        
        temp_landmark_list.append([xPos,yPos])
        
    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def calc_landmark_list_train(image, landmarks, flag):
    
    imgHeight = image.shape[0]
    imgWeight = image.shape[1]

    landmark_list = []
    landmark_list_xyz = []
    # print(landmarks)


    for i, landmark in enumerate(landmarks.landmark):
        
        xPos = min(int(landmark.x * imgWeight), imgWeight - 1)
        yPos = min(int(landmark.y * imgHeight), imgHeight - 1)
        # landmark_z = landmark.z

        landmark_list.append([xPos, yPos])
        
    if flag=='xyz':
        
        for i, landmark in enumerate(landmarks.landmark):
            
            landmark_list_xyz.append([landmark.x, landmark.y, landmark.z])
    

    return landmark_list, landmark_list_xyz

def pre_process_landmark_xyz(landmark_list_xyz):
    
    temp_landmark_list_xyz = []

    # Convert to relative coordinates
    origin_x = None
    origin_y = None
    origin_z = None
    
    for idx, landmark_point in enumerate(landmark_list_xyz):
        
        if idx == 0:
            origin_x = landmark_point[0]
            origin_y = landmark_point[1]
            origin_z = landmark_point[2]
            
        xPos = landmark_list_xyz[idx][0] - origin_x
        yPos = landmark_list_xyz[idx][1] - origin_y
        zPos = landmark_list_xyz[idx][2] - origin_z
        
        
        temp_landmark_list_xyz.append([xPos,yPos,zPos])
        
    # Convert to a one-dimensional list
    temp_landmark_list_xyz = list(
        itertools.chain.from_iterable(temp_landmark_list_xyz))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list_xyz)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list_xyz = list(map(normalize_, temp_landmark_list_xyz))

    return temp_landmark_list_xyz




def data2cvs(dataset):
    csv_path = 'data/keypoint.csv'
    with open(csv_path, 'a', newline="") as f:
        for data, label in tqdm(dataset):
        # print(data.shape)
            if label is not None:
                writer = csv.writer(f)
                writer.writerow([label, *data])
                
def my_collate_fn(batch):

    batch = list(filter(lambda x:x[0] is not None, batch))
    if len(batch) == 0: return torch.Tensor()
    return default_collate(batch) 

def draw_bounding_box(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_info_text(image, handbox, handedness, hand_sign_text):
    cv.rectangle(image, (handbox[0], handbox[1]), (handbox[2], handbox[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (handbox[0] + 5, handbox[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    # if finger_gesture_text != "":
    #     cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
    #                cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    #     cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
    #                cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
    #                cv.LINE_AA)

    return image

def draw_info(image, fps):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)
    return image


# cap = cv.VideoCapture(0)
# mpHands = mp.solutions.hands

# Hands = mpHands.Hands()
# mpDraw = mp.solutions.drawing_utils
# handLmsStyle = mpDraw.DrawingSpec(color=(0,0,255),thickness=5)
# handConStyle = mpDraw.DrawingSpec(color=(0,255,0),thickness=10)

# pTime = 0
# cTime = 0

# while True:
    
#     ret,img = cap.read()
    
#     if ret:
#         imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        
#         result = Hands.process(imgRGB)
        
#         imgHeight = img.shape[0]
        
#         imgWidth = img.shape[1]
#         # print(result.multi_hand_landmarks)
        
#         if result.multi_hand_landmarks:
            
#             for handLms in result.multi_hand_landmarks:
#                 mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS,handLmsStyle,handConStyle)
                
#                 # for i,lm in enumerate(handLms.landmark):
                    
#                 #     xPos = int(lm.x * imgWidth)
                    
#                 #     yPos = int(lm.y * imgHeight)
                    
#                 #     cv2.putText(img,str(i),(xPos-25,yPos+5),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,255),1)
                    
#                 #     print(i, xPos, yPos)
#             for hand_landmarks, handedness in zip(result.multi_hand_landmarks,result.multi_handedness):
                
#                 brect = Hand_box_coordinates(img, hand_landmarks)
                
#                 landmark_list = calc_landmark_list(img, hand_landmarks)
                
#                 pre_processed_landmark_list = pre_process_landmark(landmark_list)
                
            
        # cTime = time.time()
        # fps = 1/(cTime - pTime)
        # pTime = cTime
        
        # cv.putText(img, f'FPS : {int(fps)}',(30,50),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
        
        # cv.imshow('img',img)
    
    # if cv.waitKey(1) == ord('q'):
        # break
# cap.release()