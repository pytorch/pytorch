#!/bin/bash -xe

# Allows this script to be invoked from any directory:
cd $(dirname "$0")

./generate_config_yml.py > config.yml
