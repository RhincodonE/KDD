from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset
import torchvision.transforms as transforms
import os
import torch as ch
import torchvision
from PIL import Image

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
import time
import pandas as pd
# Configuring paths and dataset-specific settings
Section('data', 'Dataset paths and settings').params(
    output_dir=Param(str, 'Path to the output directory for beton files', default='./latent')
)

# Define a transformation pipeline that includes conversion to PIL Image
standard_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to CIFAR-10 dimensions
    transforms.ToTensor(),
    transforms.ToPILImage(),  # Convert tensors to PIL Images
    transforms.Lambda(lambda x: x.convert("RGB")),  # Ensure 3 channels
])

# Dataset loading utility
def load_dataset(name):
    if name == 'CIFAR-100':
        return torchvision.datasets.CIFAR100('./tmp', train=True, download=True, transform=standard_transform), 100
    elif name == 'CINIC':
        # Assuming CINIC dataset is similar to CIFAR and available in torchvision
        return torchvision.datasets.ImageFolder('./tmp', transform=standard_transform), 10
    elif name == 'MNIST':
        return torchvision.datasets.MNIST('./tmp', train=True, download=True, transform=standard_transform), 10
    elif name == 'EMNIST':
        return torchvision.datasets.EMNIST('./tmp', split='balanced', train=True, download=True, transform=standard_transform), 47
    elif name == 'FashionMNIST':
        return torchvision.datasets.FashionMNIST('./tmp', train=True, download=True, transform=standard_transform), 10
    elif name == 'CIFAR-10':
        return torchvision.datasets.CIFAR10('./tmp', train=True, download=True, transform=standard_transform), 100
    else:
        raise ValueError(f"Unsupported dataset: {name}")

@param('data.output_dir')
def main(output_dir):
    #datasets = ['CIFAR-100', 'CINIC', 'MNIST', 'EMNIST', 'FashionMNIST']
    datasets = ['CIFAR-10', 'EMNIST']
    for dataset_name in datasets:
        dataset, num_classes = load_dataset(dataset_name)
        print(dataset_name)
        for class_id in range(num_classes):
            class_data = Subset(dataset, [i for i, (_, label) in enumerate(dataset) if label == class_id])
            path = os.path.join(output_dir, dataset_name, str(class_id))
            os.makedirs(path, exist_ok=True)

            writer = DatasetWriter(f'{path}/data.beton', {
                'image': RGBImageField(),
                'label': IntField()
            })

            writer.from_indexed_dataset(class_data)
            print(f"Finished writing {dataset_name} class {class_id}")

if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='Dataset preparation for classification')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()
