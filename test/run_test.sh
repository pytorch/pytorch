#!/usr/bin/env bash
set -e

pushd "$(dirname "$0")"

echo "Running torch tests"
python test_torch.py

echo "Running autograd tests"
python test_autograd.py

echo "Running nn tests"
python test_nn.py

echo "Running legacy nn tests"
python test_legacy_nn.py

echo "Running multiprocessing tests"
python test_multiprocessing.py

echo "Running util tests"
python test_utils.py

echo "Running dataloader tests"
python test_dataloader.py

if which nvcc >/dev/null 2>&1
then
    echo "Running cuda tests"
    python test_cuda.py

    echo "Running NCCL tests"
    python test_nccl.py
else
    echo "nvcc not found in PATH, skipping CUDA tests"
fi

popd
