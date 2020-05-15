#!/bin/bash -xe


YAML_FILENAME=verbatim-sources/workflows-pytorch-ge-config-tests.yml
DIFF_TOOL=meld


# Allows this script to be invoked from any directory:
cd $(dirname "$0")

pushd ..


$DIFF_TOOL $YAML_FILENAME <(./codegen_validation/normalize_yaml_fragment.py < $YAML_FILENAME)


popd
