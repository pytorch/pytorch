#!/bin/bash
# Script used only in CD pipeline

set -ex

mkdir -p /usr/local/mnist/

cd /usr/local/mnist

for img in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  wget -q https://ossci-datasets.s3.amazonaws.com/mnist/$img
  gzip -d $img
done
