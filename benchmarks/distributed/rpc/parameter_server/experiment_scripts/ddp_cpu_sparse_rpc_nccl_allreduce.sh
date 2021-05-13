#!/bin/sh

cd "$(dirname "$0")"
cd ..

python -u launcher.py \
    --benchmark="4" \
    --data="DummyData" \
    --model="DummyModelSparse" \
    --server="AverageCpuParameterServer" \
    --trainer="DdpCpuSparseRpcTrainer"
