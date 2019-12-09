#!/bin/bash -xe

# Allows this script to be invoked from any directory:
cd $(dirname "$0")

NEW_FILE=$(mktemp)
./generate_config_yml.py > $NEW_FILE
cp $NEW_FILE config.yml
