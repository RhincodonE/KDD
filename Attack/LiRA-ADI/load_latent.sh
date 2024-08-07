#!/bin/bash

# Base YAML configuration file path
yaml_file="config_latent.yaml"


python write_latent_datasets.py --config-file $yaml_file


echo "Model trained."
