import os
import time
from utils import *
from dataloader import *
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
from resnet import ResNet18, ResNet34, ResNet50, ResNet101

model_addr = './Attack/Target_model/Mnist_res18.tar'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


#Load data

file = "./target.json"

args_loader = load_json(json_file=file)

train_file_path = args_loader["dataset"]["train_file_path"]

test_file_path = args_loader["dataset"]["test_file_path"]

train_batch=128

trainloader = init_dataloader(args_loader, train_file_path,train_batch)

test_batch=100

testloader = init_dataloader(args_loader, test_file_path, test_batch)
#load nt
network = ResNet18()
#state_dict = torch.load(model_addr)['state_dict']
#network.load_state_dict(state_dict,strict=False)
#network.eval()
network.to(device)


if device == 'cuda':
    network = torch.nn.DataParallel(network)
    cudnn.benchmark = True


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']



#load optimizer and set learning rate

optimizer = optim.SGD(network.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

#load loss function

criterion = nn.CrossEntropyLoss()

def train(epoch):
    print('\nEpoch: %d' % epoch)
    network.train()
    train_loss = 0
    correct = 0
    total = 0
 
    for batch_idx, (inputs,targets) in enumerate(trainloader):
        inputs = torch.unsqueeze(inputs, dim=1)
        inputs, targets = inputs.to(torch.float32).to(device), targets.long().to(device)
        optimizer.zero_grad()
        outputs = network(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
  
        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))



def test(epoch):
    global best_acc
    network.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_idx=0
    with torch.no_grad():
        for inputs, (inputs,targets) in enumerate(testloader):
            inputs = torch.unsqueeze(inputs, dim=1)
            inputs, targets = inputs.to(torch.float32).to(device), targets.long().to(device)

            outputs = network(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            batch_idx = batch_idx+1
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': network.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    

for epoch in range(1, 40):
    train(epoch)
    test(epoch)
    torch.save(network, model_addr)
    scheduler.step()
