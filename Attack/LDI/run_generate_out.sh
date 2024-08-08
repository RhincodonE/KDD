#!/bin/bash

# Base YAML configuration file path
yaml_file="./configs/config_attack_out.yaml"
original_yaml_file="./configs/original_config_attack.yaml"

# Make a copy of the original YAML to preserve it
cp $yaml_file $original_yaml_file

python ./scripts/attack_out.py --config-file $yaml_file

cp $original_yaml_file $yaml_file

rm $original_yaml_file
echo "Attacking finished!"
