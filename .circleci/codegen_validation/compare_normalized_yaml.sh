#!/bin/bash -xe


YAML_FILENAME=verbatim-sources/windows-build-test.yml


# Allows this script to be invoked from any directory:
cd $(dirname "$0")

pushd ..


meld $YAML_FILENAME <(./normalize_yaml_fragment.py < $YAML_FILENAME)


popd