
from argparse import ArgumentParser
from typing import List
import time
import re
import logging
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import torch as ch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
import torchvision
from torchvision import datasets, transforms
from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf
import pandas as pd
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter
import os
from torch.utils.data import Subset
from scipy.stats import norm
from torch.utils.data import DataLoader
import random
Section('data', 'data related stuff').params(
    index_folder=Param(str, 'folder to store the observations', required=True),
)

def get_files_with_contents(directory):
    files_contents = {}
    for filename in os.listdir(directory):
        if filename.startswith('in_') and filename.endswith('.npy'):
            file_path = os.path.join(directory, filename)
            data = np.load(file_path)
            files_contents[filename] = set(data)  # Store data as a set for efficient look-up
    return files_contents

def find_files_with_and_without_indices(files_contents, total_indices):
    results = {}
    # Add tqdm to the outer loop for key in total_indices
    for key in tqdm(total_indices, desc="Processing keys"):
        key_files = {f for f, contents in files_contents.items() if key in contents}
        non_key_dict = {}

        # Add tqdm to the inner loop for checking each index m except key
        for m in tqdm([x for x in total_indices if x != key], desc=f"Comparing key {key}", leave=False):
            # Files where key exists but m does not
            m_files = {f for f in key_files if m not in files_contents[f]}
            if m_files:
                non_key_dict[m] = list(m_files)

        if non_key_dict:
            results[key] = non_key_dict

    return results


def flatten_results_dict(results_dict):
    rows = []
    # Add tqdm to the loop for iterating through the results dictionary
    for key, m_dict in tqdm(results_dict.items(), desc="Flattening results"):
        for m, files in m_dict.items():
            if files:  # Ensure there are files to list
                rows.append({'Key': key, 'Index': m, 'Files': ', '.join(files)})
    return rows


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    config.augment_argparse(parser)
    # Loads from args.config_file if provided
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    file_dicts = {}
    index_file_path = config["data.index_folder"]
    random_indices = random.sample(range(50000), 100)
    files_with_contents = get_files_with_contents(index_file_path)
    results_dict = find_files_with_and_without_indices(files_with_contents, random_indices)
    flat_list = flatten_results_dict(results_dict)

# Create DataFrame from the flat list
    df = pd.DataFrame(flat_list)
    print(df.head())  # Print first few rows to verify the DataFrame looks as expected

# Save the DataFrame to a CSV file
    df.to_csv('/users/home/ygu27/cifar/MIA/data/inf/results_dict.csv', index=False)
    print("Saved the results to 'results_dict.csv'")
