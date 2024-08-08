from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset

import torch as ch
import torchvision

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
import time
import pandas as pd

# Load the top 5000 indices from a CSV file

Section('data', 'arguments to give the writer').params(
    in_dataset=Param(str, 'Where to write the in dataset', required=True),
    val_dataset=Param(str, 'Where to write the val dataset', required=True),
    out_dataset=Param(str, 'Where to write the in dataset', required=True),
    out_dataset_index=Param(str, 'Where to write the in dataset', required=True),
    in_dataset_index=Param(str, 'Where to write the in dataset', required=True),
    unlearning_index=Param(str, 'Where to indicate which samples to unlearn', required=True),
    unlearning=Param(bool, 'Choose if the dataset is used to unlearning', required=True),
)




@param('data.in_dataset')
@param('data.val_dataset')
@param('data.out_dataset')
@param('data.out_dataset_index')
@param('data.in_dataset_index')
def main(in_dataset, val_dataset, out_dataset, out_dataset_index,in_dataset_index,test = True):
# Set seed for reproducibility
    np.random.seed(int(time.time()))

# Load the full CIFAR10 training dataset
    original_dataset = torchvision.datasets.CIFAR10('/users/home/ygu27/cifar/tmp_u', train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10('/users/home/ygu27/cifar/tmp_u', train=False, download=True)
    full_train_dataset = original_dataset
    indices = np.random.permutation(len(full_train_dataset))
    midpoint = len(indices) -100
    out_indices = indices[midpoint:]
    in_indices = indices[:midpoint]

    in_subset = Subset(full_train_dataset, in_indices)

    datasets = {
            'in': in_subset,
            'test': test_dataset,
            }

    for (name, ds) in datasets.items():
        if name == 'in':
            path = in_dataset
        else:
            path = val_dataset

        writer = DatasetWriter(path, {
            'image': RGBImageField(),
            'label': IntField()
        })

        writer.from_indexed_dataset(ds)

    num_finetune_files = 500
# Generate and save fine-tune index files
    for i in range(num_finetune_files):
    # Randomly select indices with a 50% probability
        fine_tune_indices = [idx for idx in out_indices if np.random.rand() < 0.5]
        remaining_out_indices = [idx for idx in out_indices if idx not in fine_tune_indices]

    # Save the fine-tune indices
        np.save(f'/users/home/ygu27/cifar/tmp_f/finetune_index_{i}.npy', fine_tune_indices)

    # Save the remaining out indices
        np.save(f'/users/home/ygu27/cifar/tmp_f/remaining_index_{i}.npy', remaining_out_indices)

        fine_tune_subset = Subset(full_train_dataset, fine_tune_indices)
        remaining_subset = Subset(full_train_dataset, remaining_out_indices)

        datasets = {
            'fine_tune': fine_tune_subset,
            'remaining': remaining_subset
            }

        for (name, ds) in datasets.items():
            if name == 'fine_tune':
                path = f'/users/home/ygu27/cifar/tmp_f/finetune_{i}.beton'
            else:
                path = f'/users/home/ygu27/cifar/tmp_f/remaining_{i}.beton'

            writer = DatasetWriter(path, {
                'image': RGBImageField(),
                'label': IntField()
            })

            writer.from_indexed_dataset(ds)

if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()
