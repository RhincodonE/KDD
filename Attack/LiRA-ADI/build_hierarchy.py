from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm
import os
import torch as ch
import pandas as pd
import scipy.stats as stats
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
import torchvision
import random
import concurrent.futures
import json
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from fastargs import Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf
from fastargs import get_current_config

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

Section('data', 'data related stuff').params(
    output_dir=Param(str, 'file to store in datasets', required=True),
    statistic_out=Param(str, 'file to store in datasets', required=True),
    gpu=Param(int, 'GPU to use', required=True),
)

Section('training', 'Hyperparameters').params(
    num_workers=Param(int, 'The number of workers', default=8),
    model_save_path=Param(str, 'model save addr', default=True),
)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class TreeNode:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.children = []
        self.file = None
        self.probability = 0  # Initialized to zero

    def add_child(self, child):
        self.children.append(child)
        self.update_probabilities()  # Update probabilities every time a child is added

    def set_file(self, file_path):
        self.file = file_path

    def update_probabilities(self):
        if self.children:
            total = len(self.children)
            for child in self.children:
                child.probability = 1 / total  # Set each child's probability

def build_tree(directory, parent=None):
    node = TreeNode(name=os.path.basename(directory), path=directory)
    if parent:
        parent.add_child(node)

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            build_tree(item_path, parent=node)
        elif os.path.isfile(item_path) and item.endswith('.beton'):
            node.set_file(item_path)

    return node


def choose_leaf_node(node):
    current_node = node
    while current_node.children:
        probabilities = [child.probability for child in current_node.children]
        current_node = random.choices(current_node.children, weights=probabilities, k=1)[0]
    return current_node

def batch_choose_leaf_nodes(root_node, batch_size):
    leaf_nodes = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Future objects for each leaf node selection
        futures = [executor.submit(choose_leaf_node, root_node) for _ in range(batch_size)]
        for future in concurrent.futures.as_completed(futures):
            leaf_nodes.append(future.result())
    return leaf_nodes

def assign_probabilities(root):
    if root.children:
        root.probability = 1
        root.update_probabilities()

def save_tree_to_file(node, file_path, level=0):
    indent = '    ' * level  # Indentation to show hierarchy level
    with open(file_path, 'a') as file:
        if node.file:
            file.write(f"{indent}{node.name} (Prob: {node.probability}, File: {node.file})\n")
        else:
            file.write(f"{indent}{node.name} (Prob: {node.probability})\n")
        for child in node.children:
            save_tree_to_file(child, file_path, level + 1)

def tree_to_dict(node):
    node_dict = {
        'name': node.name,
        'path': node.path,
        'file': node.file,
        'probability': node.probability,
        'children': [tree_to_dict(child) for child in node.children]
    }
    return node_dict

def dict_to_tree(node_dict):
    node = TreeNode(node_dict['name'], node_dict['path'])
    node.file = node_dict.get('file')
    node.probability = node_dict.get('probability', 0)
    node.children = [dict_to_tree(child) for child in node_dict['children']]
    return node

def save_tree_to_json(node, file_path):
    tree_dict = tree_to_dict(node)
    with open(file_path, 'w') as file:
        json.dump(tree_dict, file, indent=4)

def load_tree_from_json(file_path):
    with open(file_path, 'r') as file:
        tree_dict = json.load(file)
    return dict_to_tree(tree_dict)

@param('training.num_workers')
@param('data.gpu')
def make_dataloaders(leaf_nodes, image_number, num_workers, gpu):
    CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR_STD = [0.2023, 0.1994, 0.2010]
    loaders = {}
    names = []
    label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(ch.device(f'cuda:{gpu}')), Squeeze()]
    image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
    image_pipeline.extend([
        ToTensor(),
        ToDevice(ch.device(f'cuda:{gpu}'), non_blocking=True),
        ToTorchImage(),
        Convert(ch.float16),
        torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    for node in leaf_nodes:
        # Creating a loader to randomly access images
        loader = Loader(node.file, batch_size=1, num_workers=num_workers,
                        order=OrderOption.RANDOM,
                        pipelines={'image': image_pipeline, 'label': label_pipeline})

        # Load and randomly select images
        random_images = []
        for _ in tqdm(range(image_number)):
            random_images.append(next(iter(loader))[0])  # Only take the image tensor, ignoring the label

# Concatenate the list of images along the batch dimension
        random_images = ch.cat(random_images, dim=0)

# Create a custom dataset with the randomly selected images
        custom_dataset = CustomDataset(random_images)

        custom_loader = DataLoader(custom_dataset, batch_size=10, shuffle=True, num_workers=num_workers)

        loaders[node.path] = custom_loader
        names.append(node.path)

    return loaders, names

class Mul(ch.nn.Module):
    def __init__(self, weight):
       super(Mul, self).__init__()
       self.weight = weight
    def forward(self, x): return x * self.weight

class Flatten(ch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Residual(ch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return ch.nn.Sequential(
            ch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                         stride=stride, padding=padding, groups=groups, bias=False),
            ch.nn.BatchNorm2d(channels_out),
            ch.nn.ReLU(inplace=True)
    )

@param('data.gpu')
def construct_model(gpu = 0):
    num_class = 10
    model = ch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(ch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        ch.nn.MaxPool2d(2),
        Residual(ch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        ch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        ch.nn.Linear(128, num_class, bias=False),
        Mul(0.2)
    )
    model = model.to(memory_format=ch.channels_last).cuda(int(gpu))
    return model

@param('data.statistic_out')
def load_statistics(statistic_out):
    """Load statistics from a CSV file."""
    stats_df_out = pd.read_csv(statistic_out)
    out_mu = stats_df_out.loc[stats_df_out['Parameter'] == 'mu', 'Value'].values[0]
    out_std = stats_df_out.loc[stats_df_out['Parameter'] == 'std', 'Value'].values[0]
    return (out_mu, out_std)

def evaluate(model, loaders, names):
    model.eval()
    gaps = {}
    with ch.no_grad():
        for name in names:
            logit_gaps = []
            for ims in tqdm(loaders[name]):
                with autocast():
                    out = model(ims.cuda())
                    top_two_logits = ch.topk(out, 2, dim=1).values

                    # Calculate logit gap: top logit - second top logit
                    logit_gap = top_two_logits[:, 0] - top_two_logits[:, 1]
                    logit_gaps.extend(logit_gap.cpu().tolist())

            average_logit_gap = sum(logit_gaps) / len(logit_gaps)
            gaps[name] = average_logit_gap
            print(f'{name} gap: {average_logit_gap:.4f}%')

    return gaps

def calculate_lambda(gap, mu_out, sigma_out):
    # Calculate the z-score
    z = (gap - mu_out) / sigma_out

    # Calculate the cumulative probability
    cumulative_prob = stats.norm.cdf(z)

    # Calculate Lambda
    lambda_value = 1 - cumulative_prob

    return lambda_value

def load_model(model_path, device):
    model = construct_model()  # Assuming this function correctly constructs the model
    model.load_state_dict(ch.load(model_path, map_location=device))
    return model

if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='ADI')
    config.augment_argparse(parser)
    # Loads from args.config_file if provided
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    model = load_model(config['training.model_save_path'], f'cuda:{config["data.gpu"]}')

    output_file = 'tree_structure.json'
    tree_root = build_tree(config['data.output_dir'])

    assign_probabilities(tree_root)
    save_tree_to_json(tree_root, output_file)

    batch_size = 2  # Number of leaf nodes to select
    leaf_nodes = batch_choose_leaf_nodes(tree_root, batch_size)
    for leaf in leaf_nodes:
        print(f"Chosen Leaf Node: {leaf.path}, File: {leaf.file}")
    print(f"Tree structure saved to {output_file}")

    loaders, names = make_dataloaders(leaf_nodes, 1000)
    out_mu, out_std = load_statistics()

    gaps = evaluate(model, loaders, names)

    lambdas = [calculate_lambda(gap, out_mu, out_std) for gap in gaps.values()]

# Output the results
    for name, lambda_value in zip(gaps.keys(), lambdas):
        print(f"Lambda for {name}: {lambda_value}")
