#!/bin/bash

if [ -n "$TENSORRT_VERSION" ]; then
    python3 -m pip install --upgrade setuptools pip
    python3 -m pip install nvidia-pyindex
    python3 -m pip install nvidia-tensorrt==${TENSORRT_VERSION} --extra-index-url https://pypi.ngc.nvidia.com
fi
