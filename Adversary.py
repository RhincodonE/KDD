'''
The adversary will use this file to update possibility of each cluster(class)
'''

import os
import time
from utils import *
from dataloader import *
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("./datasets")
import torch
import torchvision
import argparse
from torch.nn import BCELoss
from torch.autograd import grad
import torchvision.utils as tvls
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
from dataloader import *
from scipy.stats import entropy
from random import shuffle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic

#########Get Transfer set########
def attack(budget,batch_size,queryset,blackbox,t,prob_c):
'''
budget: the total budget for adversary

queryset: the adversary's dataset.

blackbox: the trained target model. There has been one contained in ./Attack/Target_model

t: threshold for entropy

prob_c: probability change for clusters
'''
    idx_set = set(range(len(queryset.image_list)))
    start_B = 0
    end_B = budget
    with tqdm(total=budget) as pbar:
        
        for t, B in enumerate(range(start_B, end_B, batch_size)):

            # Randomely choose which clusters to try. 
            
            classes =random.choices(queryset.class_list,weights=queryset.prob_list,k=batch_size)

            idxs = []

            #Randomely choose images from the chosen clusters
            for i in classes:
                
                idx_l = queryset.dic[i]
                
                idxs.append(random.choice(idx_l))

            #Input the images to the target model
                
            img_t = [queryset.name_list[i] for i in idxs]  # Image paths
            
            x_t = torch.stack([torch.unsqueeze(torch.from_numpy(queryset.image_list[i]), dim=0).to(torch.float32) for i in idxs]).to(device)
            
            y_t = blackbox(x_t).cpu()

            prob = F.softmax(y_t,dim=1).tolist()

            entropy_list = [] 

            #Compute output confidence vector entropy
            for i in prob:

                entropy_list.append(entropy(i))

            for i in range(len(entropy_list)):
                
                class_name = classes[i]
            # The probability of the cluster will increase by prob_c if the entropy is greater
            # threshold t. Otherwise, it will decrease by prob_c
                if entropy_list[i]<t:

                    queryset.prob_list[class_name] = queryset.prob_list[class_name]+prob_c
                    
                else:

                    queryset.prob_list[class_name] = queryset.prob_list[class_name]-prob_c

    #Store the updated probability in new file class_adversary.txt
            
    addr_store_original_train='./Data'+'/con/original/train/'

    with open(addr_store_original_train+'class_adversary.txt','a') as f1:

        for i in queryset.class_list:

            f1.write(str(i)+' '+str(queryset.prob_list[i])+'\n')
            

def query_set(file="./transfer.json"):

    args = load_json(json_file=file)

    file_path = args["dataset"]["train_file_path"]

    class_prob_path = args["dataset"]["class_prob_path"]

    data_set = GrayFolder(args, file_path,class_prob_path)

    return data_set

if __name__ == '__main__':

    #########Target model############
    PATH = './Attack/Target_model/Mnist_res18.tar'
    Model = torch.load(PATH)
    #########Querry Set##############
    query = query_set()
    ########Transferset##############
    attack(budget = 1000,batch_size = 10,queryset=query,blackbox=Model,t=0.7,prob_c=0.0001)
    
    
