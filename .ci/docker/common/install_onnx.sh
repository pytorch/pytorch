#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

# A bunch of custom pip dependencies for ONNX
pip_install \
  beartype==0.10.4 \
  filelock==3.9.0 \
  flatbuffers==2.0 \
  mock==5.0.1 \
  ninja==1.10.2 \
  networkx==2.0 \
  numpy==1.22.4

# ONNXRuntime should be installed before installing
# onnx-weekly. Otherwise, onnx-weekly could be
# overwritten by onnx.
pip_install \
  onnxruntime==1.15.1 \
  parameterized==0.8.1 \
  pytest-cov==4.0.0 \
  pytest-subtests==0.10.0 \
  tabulate==0.9.0 \
  transformers==4.31.0

# Using 1.15dev branch for the following not yet released features and fixes.
# - Segfault fix for shape inference.
# - Inliner to workaround ORT segfault.
pip_install onnx-weekly==1.15.0.dev20230717

# TODO: change this when onnx-script is on testPypi
pip_install onnxscript-preview==0.1.0.dev20230809 --no-deps
