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
    train_dataset=Param(str, 'Where to write the in dataset', required=True),
    test_dataset=Param(str, 'Where to write the in dataset', required=True),
)


@param('data.train_dataset')
@param('data.test_dataset')
def main(train_dataset,test_dataset):
# Set seed for reproducibility


# Load the full CIFAR10 training dataset
    train_set = torchvision.datasets.CIFAR10('./tmp', train=True, download=True)

    test_set = torchvision.datasets.CIFAR10('./tmp', train=False, download=True)

    datasets = {
        'train': train_set,
        'test': test_set,
        }

    for (name, ds) in datasets.items():
        if name == 'train':
            path = train_dataset
        else:
            path = test_dataset

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
