#!/bin/bash -xe

YAML_FILENAME=$1

# Allows this script to be invoked from any directory:
cd $(dirname "$0")

pushd ..

TEMP_FILENAME=$(mktemp)

cat $YAML_FILENAME | ./codegen_validation/normalize_yaml_fragment.py > $TEMP_FILENAME
mv $TEMP_FILENAME $YAML_FILENAME

popd
