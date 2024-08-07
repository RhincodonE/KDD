#!/bin/bash

# Base YAML configuration file path
yaml_file="./config_ADI.yaml"


python build_hierarchy.py --config-file $yaml_file


echo "Finished."
