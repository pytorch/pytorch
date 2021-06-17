#!/bin/sh

cd "$(dirname "$0")"
cd ..

python -u launcher.py \
        --benchmark="3" \
        --data="DummyData" \
        --model="DummyModel" \
        --server="None" \
        --trainer="DdpNcclTrainer"
