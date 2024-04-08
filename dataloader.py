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

class GrayFolder(data.Dataset):
    
    def __init__(self, args, file_path, prob_path = None):
                
        self.args = args
        
        self.dataset=args["dataset"]["name"]
        
        self.img_type=args["dataset"]["img_type"]
        
        self.img_path = args["dataset"]["img_path"]

        self.img_list = os.listdir(self.img_path)
        
        self.name_list, self.label_list, self.pic_dict = self.get_list(file_path)

        self.targets = self.label_list

        self.classes = self.label_list
        
        self.image_list = self.load_img()
        
        self.num_img = len(self.image_list)
        
        self.n_classes = args["dataset"]["n_classes"]

        if prob_path!=None:

            self.class_list, self.prob_list = self.load_class_prob(prob_path)

            self.dic = self.load_img_idx(self.class_list,self.label_list)


        print("Load " + str(self.num_img) + " images")

    def get_list(self, file_path):
   
        name_list, label_list, pic_dict = [], [], {}
        
        f = open(file_path, "r")
        
        for line in f.readlines():
            
            img_name, iden = line.strip().split(' ')
            
            name_list.append(img_name)
            
            label_list.append(int(iden))

            pic_dict[img_name]=int(iden)

        return name_list, label_list, pic_dict
    
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

    def load_class_prob(self,file_path):

        class_list = []

        prob_list = []

        f = open(file_path, "r")
        
        for line in f.readlines():
            
            class_name, prob = line.strip().split(' ')
            
            class_list.append(int(class_name))

            prob_list.append(float(prob))
            
        return class_list, prob_list

    def load_img_idx(self, class_list, label_list):

        dic = {}

        for i in class_list:
            
            idx = [idx0 for idx0, x in enumerate(label_list) if x == i]

            dic[i] = idx

        return dic  #{class:[idx1,idx2,idx3]}

    def load_evaluation(self):

        classes =random.choices(self.class_list,weights=self.prob_list,k=1000)

        idxs=[]


        for i in classes:
                
                idx_l = self.dic[i]
                
                idxs.append(random.choice(idx_l))
                
        self.image_list = [self.image_list[i] for i in idxs]

        self.label_list = [0 for i in idxs]


    def __getitem__(self, index):

        img = self.image_list[index]

        img = np.moveaxis(img, -1, 0)
         
        label = self.label_list[index]
        
        return img, label

    def __len__(self):

        return self.num_img
