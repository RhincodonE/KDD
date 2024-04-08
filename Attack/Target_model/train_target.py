import os
import time
import utils
import torch
import dataloader
import torchvision
from utils import *
from torch.nn import BCELoss
from torch.autograd import grad
import torchvision.utils as tvls
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from resnet import ResNet18




#define model save direction
model_addr='./Attack/attack_models/Cifar_resnet18_original.tar'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#load net
network = ResNet18()
network.to(device)

#load args file
file = "./MNIST.json"

args = load_json(json_file=file)

#load training and testing dataloader
train_file_path = args["dataset"]["train_file_path"]

test_file_path = args["dataset"]["test_file_path"]

train_batch=128

dataloader = init_dataloader(args, train_file_path, train_batch, mode="target")

test_batch=1000

test_dl= init_dataloader(args, test_file_path, test_batch, mode="target")

#load optimizer and set learning rate

optimizer = optim.SGD(network.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
#load loss function
loss_func = nn.CrossEntropyLoss() 

#define epochs number
num_epochs = 30

def train(num_epochs, cnn, loaders):
    
    cnn.train()
        
    # Train the model
    total_step = len(loaders)
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images).to(device)   # batch x
            
            b_y = Variable(labels).to(device)   # batch y

            optimizer.zero_grad()
            
            output = cnn(b_x)
            
            loss = loss_func(output, b_y)
            
            loss.backward()
                    
            optimizer.step()
             
            if i%train_batch==0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        test_intrain(network=cnn)
        
        print('learning rate: '+str(optimizer.param_groups[0]['lr']))
        
        scheduler.step()
        
    #save trained model
    torch.save({'state_dict':cnn.state_dict()}, model_addr)
    

def test_intrain(network):
    
    # Test the model
    
    #target_path = model_addr
    
    #ckp_T = torch.load(target_path)['state_dict']
    
    #utils.load_my_state_dict(network, ckp_T)
    
    #print(network.state_dict())
    network.to(device)
    network.eval()
    
    with torch.no_grad():
        
        correct = 0
        
        total = 0
        
        for images, labels in test_dl:

            label = labels.to(device)
            
            test_output = network(images.to(device))
            
            pred_y = torch.max(test_output, 1)[1].data.squeeze().to(device)
            
            accuracy = (pred_y == label).sum().item() / float(label.size(0))
            
            pass
        
    print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)
    
    
def test(network):
    
    # Test the model
    
    target_path = model_addr
    
    ckp_T = torch.load(target_path)['state_dict']
    
    utils.load_my_state_dict(network, ckp_T)
    
    print(network.state_dict())
    
    network.eval()
    
    with torch.no_grad():
        
        correct = 0
        
        total = 0
        
        for images, labels in test_dl:
            
            test_output = network(images)
            
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            
            pass
    print('Test Accuracy of the model on the test images: %.2f' % accuracy)
    

train(num_epochs, network, dataloader)
test(network)
