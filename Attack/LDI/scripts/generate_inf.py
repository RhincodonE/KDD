
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
from collections import defaultdict
Section('data', 'data related stuff').params(
    statistic_in=Param(str, 'folder to store the observations', required=True),
    statistic_out=Param(str, 'folder to store the observations', required=True),
    gpu=Param(int, 'folder to store the observations', required=True),
)

@param('data.gpu')
def make_dataloaders(dataset=None, batch_size=1, num_workers=3, gpu=0):

    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]

    label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(ch.device(f'cuda:{gpu}')), Squeeze()]
    image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

    image_pipeline.extend([
            ToTensor(),
            ToDevice(ch.device(f'cuda:{gpu}'), non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    loader = Loader(dataset, batch_size=batch_size, num_workers=num_workers,
                               order=OrderOption.SEQUENTIAL,
                               pipelines={'image': image_pipeline, 'label': label_pipeline})

    return loader

def extract_number_from_filename(filename):
    # Regular expression to find numbers in the filename
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None


# Initialize a three-level nested dictionary

@param('data.statistic_in')
@param('data.statistic_out')
def load_statistics(statistic_in,statistic_out):
    """Load statistics from a CSV file."""
    stats_df_in = pd.read_csv(statistic_in)
    stats_df_out = pd.read_csv(statistic_out)
    in_mu = stats_df_in.loc[stats_df_in['Parameter'] == 'mu', 'Value'].values[0]
    in_std = stats_df_in.loc[stats_df_in['Parameter'] == 'std', 'Value'].values[0]
    out_mu = stats_df_out.loc[stats_df_out['Parameter'] == 'mu', 'Value'].values[0]
    out_std = stats_df_out.loc[stats_df_out['Parameter'] == 'std', 'Value'].values[0]
    return (in_mu, in_std), (out_mu, out_std)

# Model (from KakaoBrain: https://github.com/wbaek/torchskeleton)
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
    model = model.to(memory_format=ch.channels_last).cuda(gpu).half()
    return model

def load_model(model_path, device):
    model = construct_model()  # Assuming this function correctly constructs the model
    model.load_state_dict(ch.load(model_path, map_location=device))
    return model

def pdf_normal(x, mu, sigma):
    """Calculate the normal distribution's PDF at x."""
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def classify_sample(x, in_stats, out_stats):
    """Classify a sample x based on in and out distribution statistics."""
    in_pdf = pdf_normal(x, *in_stats)  # Unpack the tuple in_stats into mu and sigma
    out_pdf = pdf_normal(x, *out_stats)

    if in_pdf > out_pdf:
        return "train_in"
    else:
        return "train_out"

def flatten_nested_dict(nested_dict):
    # Create a list to hold all flattened entries
    rows = []
    # Iterate through each level of the dictionary
    for key, index_dict in nested_dict.items():
        for index, predictions in index_dict.items():
            # Convert set of predictions to a string for easy CSV storage
            predictions_str = ', '.join(predictions)
            # Append the flattened data as a dictionary
            rows.append({'Key': key, 'Index': index, 'Predictions': predictions_str})
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
    original_dataset = torchvision.datasets.CIFAR10('/users/home/ygu27/cifar/tmp', train=True, download=True)

    writer = DatasetWriter('/users/home/ygu27/cifar/tmp/original_dataset.beton', {
            'image': RGBImageField(),
            'label': IntField()
        })

    writer.from_indexed_dataset(original_dataset)

    loader = make_dataloaders('/users/home/ygu27/cifar/tmp/original_dataset.beton')

# Load the CSV file into a DataFrame
    df = pd.read_csv('/users/home/ygu27/cifar/MIA/data/inf/results_dict.csv')

    nested_dict = defaultdict(lambda: defaultdict(set))

    stat_in,stat_out = load_statistics()

    # Iterate through each row in the DataFrame
    nested_predictions = defaultdict(lambda: defaultdict(set))
    config = {'data.gpu': 0}  # Assuming the GPU configuration is set

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing DataFrame Rows"):
        key = row['Key']
        index = row['Index']
        files = row['Files'].split(', ')

        for file in tqdm(files,total=len(files)):
            number = extract_number_from_filename(file)
            if number is not None:
                model_addr = f'/users/home/ygu27/cifar/models/model_in_{number}.pth'
                model = load_model(model_addr, f'cuda:{config["data.gpu"]}')
                if model:
                    model.eval()
                    with ch.no_grad():
                    # Adding tqdm to the loader loop with a conditional check
                        for idx, (ims, labs) in tqdm(enumerate(loader), total=len(loader), desc="Evaluating Model", leave=False):
                            if idx == key:  # Check if the index matches the key
                                out = model(ims.to(f'cuda:{config["data.gpu"]}'))
                                logit = out.softmax(dim=1)
                                values, indices = ch.topk(logit, 2, dim=1)
                                logit_gap = values[:, 0] - values[:, 1]
                                prediction = classify_sample(logit_gap.cpu().item(),stat_in,stat_out)
                                print(prediction)
                                nested_predictions[key][index].append(prediction)
                                break
            # Flatten the nested dictionary
        flat_list = flatten_nested_dict(nested_predictions)

# Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(flat_list)

# Save the DataFrame to a CSV file
        csv_file_path = '/users/home/ygu27/cifar/MIA/data/inf/nested_predictions.csv'
        df.to_csv(csv_file_path, index=False)

        print(f"Data saved to {csv_file_path}")


# Print some of the data to verify
    for key, index_dict in list(nested_predictions.items()):
        for index, predictions in index_dict.items():
            print(f"Key: {key}, Index: {index}, Predictions: {predictions}")
