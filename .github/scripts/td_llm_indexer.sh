#!/bin/bash

set -euxo pipefail

# Download requirements
cd llm-target-determinator
pip install -q -r requirements.txt
cd ../codellama
pip install --no-build-isolation -v -e .
pip install numpy==2.0.2

# Run indexer
cd ../llm-target-determinator

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=1 \
    indexer.py \
    --experiment-name indexer-files \
    --granularity FILE
