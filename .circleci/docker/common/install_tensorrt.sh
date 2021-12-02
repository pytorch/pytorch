#!/bin/bash

if [ -n "$TENSORRT" ]; then
    python3 -m pip install --upgrade setuptools pip
    python3 -m pip install nvidia-pyindex
    python3 -m pip install --upgrade nvidia-tensorrt
fi
