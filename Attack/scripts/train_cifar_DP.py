"""
Fast training script for CIFAR-10 using FFCV.
For tutorial, see https://docs.ffcv.io/ffcv_examples/cifar10.html.

First, from the same directory, run:

    `python write_datasets.py --data.train_dataset [TRAIN_PATH] \
                              --data.val_dataset [VAL_PATH]`

to generate the FFCV-formatted versions of CIFAR.

Then, simply run this to train models with default hyperparameters:

    `python train_cifar.py --config-file default_config.yaml`

You can override arguments as follows:

    `python train_cifar.py --config-file default_config.yaml \
                           --training.lr 0.2 --training.num_workers 4 ... [etc]`

or by using a different config file.
"""
from argparse import ArgumentParser
from typing import List
import time
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
from opacus import PrivacyEngine
import torch as ch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
import torchvision
from torch.optim.lr_scheduler import LambdaLR
from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf
import torch.nn as nn
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

os.environ['TORCH_USE_CUDA_DSA'] = '1'
Section('training', 'Hyperparameters').params(
    lr=Param(float, 'The learning rate to use', required=True),
    epochs=Param(int, 'Number of epochs to run for', required=True),
    lr_peak_epoch=Param(int, 'Peak epoch for cyclic lr', required=True),
    batch_size=Param(int, 'Batch size', default=512),
    momentum=Param(float, 'Momentum for SGD', default=0.9),
    weight_decay=Param(float, 'l2 weight decay', default=5e-4),
    label_smoothing=Param(float, 'Value of label smoothing', default=0.1),
    num_workers=Param(int, 'The number of workers', default=8),
    lr_tta=Param(bool, 'Test time augmentation by averaging with horizontally flipped version', default=True),
    in_model_save_path=Param(str, 'model save addr', default=True),
)

Section('data', 'data related stuff').params(
    in_dataset=Param(str, 'file to store in datasets', required=True),
    val_dataset=Param(str, 'file to store val datasets', required=True),
    model_folder=Param(str, 'folder to store models', required=True),
    gpu=Param(int, 'GPU to use', required=True),
)

@param('data.in_dataset')
@param('data.val_dataset')
@param('training.batch_size')
@param('training.num_workers')
@param('data.gpu')
def make_dataloaders(in_dataset=None, out_dataset=None, val_dataset=None, batch_size=None, num_workers=None, gpu=0):
    paths = {
        'train_in': in_dataset,
        'train_out': out_dataset,
        'test': val_dataset
    }

    start_time = time.time()
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    loaders = {}

    for name in ['train_in', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(ch.device(f'cuda:{gpu}')), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        if name == 'train_in':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
                Cutout(4, tuple(map(int, CIFAR_MEAN))),
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice(ch.device(f'cuda:{gpu}'), non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        ordering = OrderOption.RANDOM if name == 'train_in' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name], batch_size=batch_size, num_workers=num_workers,
                               order=ordering,
                               pipelines={'image': image_pipeline, 'label': label_pipeline})
    return loaders, start_time

# Model (from KakaoBrain: https://github.com/wbaek/torchskeleton)
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def create_custom_dataloader(loader, device):
    images = []
    labels = []
    for img_batch, label_batch in tqdm(loader):
        # It's usually better to collect tensors and then move them in bulk to the desired device
        images.extend(img_batch.cpu().detach().tolist())  # Convert tensors to lists to save GPU memory
        labels.extend(label_batch.cpu().detach().tolist())
    images = ch.tensor(images)
    labels = ch.tensor(labels)
    dataset = CustomDataset(images, labels)
    new_loader = DataLoader(dataset, batch_size=loader.batch_size, shuffle=True, num_workers=loader.num_workers)
    return new_loader

def conv_gn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1, num_groups=8):
    """ Replaces BatchNorm with GroupNorm to facilitate differential privacy training. """
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                  stride=stride, padding=padding, groups=groups, bias=False),
        nn.GroupNorm(num_groups, channels_out),  # Use GroupNorm instead of BatchNorm
        nn.ReLU(inplace=True)
    )

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Mul(nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight

class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)

@param('data.gpu')
def construct_model(gpu=0):
    num_class = 10
    model = nn.Sequential(
        conv_gn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_gn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(nn.Sequential(conv_gn(128, 128), conv_gn(128, 128))),
        conv_gn(128, 256, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(2),
        Residual(nn.Sequential(conv_gn(256, 256), conv_gn(256, 256))),
        conv_gn(256, 128, kernel_size=3, stride=1, padding=0),
        nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        nn.Linear(128, num_class, bias=False),
        Mul(0.2)
    )
    model = model.to(memory_format=ch.channels_last).cuda(gpu)  # Converts the model to half precision

    return model

# Example usage to construct the model on GPU 0


@param('training.lr')
@param('training.epochs')
@param('training.momentum')
@param('training.weight_decay')
@param('training.label_smoothing')
@param('training.lr_peak_epoch')
@param('data.gpu')
def train_in(model, loaders, batch_size, lr=None, epochs=None, label_smoothing=None,
             momentum=None, weight_decay=None, lr_peak_epoch=None,
             max_grad_norm=None, gpu=0):
    device = f'cuda:{gpu}'
    model = model.to(device)
    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    iters_per_epoch = len(loaders['train_in'])
    # Cyclic LR with single triangle
    lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    scheduler = LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    # Attach the privacy engine to the optimizer

    privacy_engine = PrivacyEngine()
    model, opt, dataloader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=opt,
    data_loader=loaders['train_in'],
    max_grad_norm=1.0,
    clipping='flat',
    target_epsilon=10,
    target_delta=0.00004,
    epochs=epochs,
    )

    for epoch in range(epochs):
        model.train()
        for ims, labs in tqdm(dataloader):
            ims, labs = ims.cuda(gpu), labs.cuda(gpu)  # Ensure data is on the correct device

            opt.zero_grad(set_to_none=True)  # Reset gradients

        # Forward pass
            out = model(ims)
            loss = loss_fn(out, labs)  # Compute the loss

        # Backward pass with gradient scaling
            scaler.scale(loss).backward()

        # Update model parameters and the scaler
            scaler.step(opt)
            scaler.update()

        # Reset gradients for next iteration
            opt.zero_grad()

        # Step the learning rate scheduler
            scheduler.step()

        # Display the privacy budget spent so far
        epsilon = privacy_engine.get_epsilon(0.00004)
        print(f"Epoch: {epoch+1} - ε: {epsilon:.2f}, δ: {0.00004}")

    return loaders['train_in'], model


# Example call


@param('training.lr_tta')
@param('data.gpu')
def evaluate_in(model, loaders, gpu, lr_tta=False):
    model.eval()
    with ch.no_grad():
        for name in ['train_in', 'test']:
            total_correct, total_num = 0., 0.
            for ims, labs in tqdm(loaders[name]):

                out = model(ims.cuda(gpu).float())

                if lr_tta:
                    out += model(ims.cuda(gpu).float().flip(-1))
                total_correct += out.argmax(1).eq(labs.cuda(gpu)).sum().cpu().item()
                total_num += ims.shape[0]
            print(f'{name} accuracy: {total_correct / total_num * 100:.1f}%')

if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    config.augment_argparse(parser)
    # Loads from args.config_file if provided
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    loaders, start_time = make_dataloaders()
    gpu = config['data.gpu']
    device = f'cuda:{gpu}'
    custom_loader = create_custom_dataloader(loaders['train_in'], device)
    loaders['train_in'] = custom_loader

    model_in = construct_model()
    loaders['train_in'],model_in = train_in(model_in, loaders, max_grad_norm=1.0, batch_size=config['training.batch_size'])
    evaluate_in(model_in, loaders)
    print(f'Total time: {time.time() - start_time:.5f}')

    if not os.path.exists(config['data.model_folder']):
        os.makedirs(config['data.model_folder'])
        print(f"Created directory: {config['data.model_folder']}")
    else:
        print(f"Directory already exists: {config['data.model_folder']}")

    ch.save(model_in.state_dict(), config['training.in_model_save_path'])
