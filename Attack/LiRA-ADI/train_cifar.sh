#!/bin/bash

# Base YAML configuration file path
yaml_file="config.yaml"


python write_datasets.py --config-file $yaml_file

python train_cifar.py --config-file $yaml_file

echo "Model trained."
