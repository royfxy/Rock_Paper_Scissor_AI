# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 17:48:34 2022

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
import argparse
from read_data import *
from helperfunctions import *
from torch.utils.data.dataloader import default_collate

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

dirname = os.path.dirname(__file__)
data_dir_train = os.path.join(dirname+'\data\hand_new')
data_files_train = os.listdir(data_dir_train)
data_dir_test = os.path.join(dirname+'\data\hand_new_test')
data_files_test = os.listdir(data_dir_test)

mpHands = mp.solutions.hands

hands = mpHands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)

random.shuffle(data_files_train)
random.shuffle(data_files_test)
train, validation= random_split(data_files_train, [1903, 476])
test = random_split(data_files_test, [471])[0]


hand_data_set_train = HandDataset(train, data_dir_train, hands = hands)
hand_data_set_validation = HandDataset(validation, data_dir_train, hands = hands)
hand_data_set_test = HandDataset(test, data_dir_test, hands = hands)

# envale device to GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
#
# I define a valdation process in here so we could plugin the function later
def get_validation_loss(model, Val):
    
    model.eval()
    val_loss = []
    accuracy = 0

    # In my case I used CrossEntropy loss and enable it to GPU device
    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss().to(device)
    
    for data, label in Val:
        
        if label is not None:
        
            data, label = Variable(data), Variable(label.long())
            output = model(data)
            loss = criterion(output, label)
            val_loss.append(loss.item())
            val_acc = ((output.argmax(dim=1) == label).float().mean())
            accuracy += val_acc/len(Val)
        
    return np.mean(np.array(val_loss)),accuracy


def draw_curve(x_epoch,epoch_accuracy,epoch_loss,val_acc,val_loss):
    x_size = x_epoch
    x_length = list(range(x_size))
    fig = plt.figure()
    ax0 = fig.add_subplot(221, title="Train_accuracy")
    ax1 = fig.add_subplot(222, title="Train_loss")
    ax2 = fig.add_subplot(223, title="Val_acc")
    ax3 = fig.add_subplot(224, title="Val_loss")
    ax0.plot(x_length, epoch_accuracy, 'b-', label='Train_accuracy')    
    ax1.plot(x_length, epoch_loss, 'r-', label='Train_loss')
    ax2.plot(x_length, val_acc, 'm-', label='Val_acc')
    ax3.plot(x_length, val_loss, 'g-', label='Val_loss')
    plt.tight_layout()
    plt.show(fig)


# The main function to do the train and validation process    
def main_task():
    
    batch_size = args.batch_size
    
    Train_in_NN = DataLoader(dataset=hand_data_set_train,collate_fn = my_collate_fn, batch_size=batch_size, shuffle=True, num_workers=0)
    Val_in_NN = DataLoader(dataset=hand_data_set_validation,collate_fn = my_collate_fn, batch_size=batch_size, shuffle=True, num_workers=0)
    # Dte_in_NN = DataLoader(dataset=hand_data_set_test,collate_fn = my_collate_fn, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Total 100 epochs to train the network, min_epoch is the minimum number of required epoch for training process
    # min loss is used for determine the best model durin training for if-else condition
    epoch_num = 100
    best_model = None
    min_epochs = 5
    min_val_loss = 5

    
    
    ###########################################################################
    model = MLP_Hand(63,128,14)
    
    ########################################################################### 
    
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    epoch_a=[]
    epoch_l=[]
    val_a=[]
    val_l=[]
    
    
    for epoch in tqdm(range(epoch_num),ascii=True):

        
        epoch_train_loss=[]
        epoch_accuracy = 0
        epoch_loss = 0
        
        for data,label in Train_in_NN:
            
            if label is not None:
            
                data, label = Variable(data), Variable(label.long())
    
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss.append(loss.item())
                acc = ((output.argmax(dim=1) == label).float().mean())
                epoch_accuracy += acc/len(Train_in_NN)
                epoch_loss += loss/len(Train_in_NN)
            
            
        print('\n Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy,epoch_loss))
        val_loss,val_acc = get_validation_loss(model, Val_in_NN)
        
        model.train()
        epoch_a.append(epoch_accuracy)
        epoch_l.append(epoch_loss.detach().numpy())
        val_a.append(val_acc)
        val_l.append(val_loss)
        
        if epoch + 1 > min_epochs and val_loss < min_val_loss:
            print('pass here')
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
            
            draw_curve(epoch+1,epoch_a,epoch_l,val_a,val_l)
        
        
        
        tqdm.write('Epoch {:03d} train_loss {:.5f} val_loss {:.5f} val_acc {:.5f}'.format(epoch, np.mean(np.array(epoch_train_loss)), val_loss,val_acc))
    
    # Change the name based on you decision
    torch.save(best_model.state_dict(), "D:\Bnlearn_example\ENGN8536_Hand\hand-gesture-recognition-mediapipe-main\model\MLP_hand_xyz.pkl")
    # torch.save(model, 'D:\Bnlearn_example\lab2\model\model.pth')
    # torch.save(model.state_dict(), 'resnet18.pt')
    
    
    
    # tb.close()
    

def test():

    
    batch_size = args.batch_size
    Test_in_NN = DataLoader(dataset=hand_data_set_test,collate_fn = my_collate_fn, batch_size=batch_size, shuffle=True, num_workers=0)
    
    
    # model = MLP_Hand(42,4).to(device)
    model = MLP_Hand(63,128,14)
    ##########################################################################
    model.load_state_dict(torch.load("D:\Bnlearn_example\ENGN8536_Hand\hand-gesture-recognition-mediapipe-main\model\MLP_hand_xyz.pkl"), False)
    model.eval()
    
    test_accuracy = 0
    test_loss = 0
    total = 0
    current = 0

    for data, label in Test_in_NN:
        
        if label is not None:
        
            data, label = data, label.long()
            outputs = model(data)
            acc = ((outputs.argmax(dim=1) == label).float().mean())
            test_accuracy += acc/len(Test_in_NN) 
        
    print('\n Test accuracy is : {}'.format(test_accuracy))
    print("Total number of parameters  in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
    

# 
if __name__ == '__main__':
    # main_task()
    test()  
