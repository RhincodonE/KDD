import os, gc, sys
import json, PIL, time, random
import torch
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.nn.modules.loss import _Loss
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F
import random

def search_index(elements, target):

    index = np.array([])

    for i in elements:

        temp = np.array(np.where(target == i))

        index = np.hstack((index,temp[0]))
        
    return index.astype(int)

class GrayFolder(data.Dataset):
    
    def __init__(self, name, file_path,max_size,class_num):
        self.max_size = max_size
        self.name = name
        self.name_list, self.targets = self.get_list(file_path) 
        self.image_list = self.load_img()

        classes = [i for i in range(class_num)]
        
        idx = search_index(classes,np.array(self.targets))

        target_temp=[]

        image_temp=[]


        for i in range(idx.shape[0]):
            
            target_temp.append(self.targets[idx[i]])

            image_temp.append(self.image_list[idx[i]])

        self.targets = target_temp

        self.image_list = image_temp

        self.num_img = len(self.image_list)
        self.n_classes = 10
        self.classes = np.unique(self.targets)
        
        print("Load " + str(self.num_img) + " images")

    def get_list(self, file_path):
   
        name_list, label_list = [], []
        f = open(file_path, "r")
        count = 0
        lines = f.readlines()

        random.shuffle(lines)
        
        for line in lines:
            
            img_name, iden = line.strip().split(' ')
            name_list.append(img_name)
            label_list.append(int(iden))
            count+=1

            if count == self.max_size:

                break

        return name_list, label_list

    
    def load_img(self):
        
        img_list = []

        for i, img_name in enumerate(self.name_list):
            
            if img_name.endswith(".png"):
                
                path =  img_name
                
                img = PIL.Image.open(path)
                
                img2 = np.divide(np.array(img.copy()),255.)

                img_list.append(img2)
            
        print('load_finished')
        
        return img_list


    def __getitem__(self, index):

        img = self.image_list[index]

        img = np.moveaxis(img, -1, 0)
         
        label = self.targets[index]
        
        return img, label

  
    def __len__(self):

        return self.num_img
    
if __name__=="__main__":

    GrayFolder('A','./Data_train/Mnist/test/label.txt',2000,10)
