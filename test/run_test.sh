#!/usr/bin/env bash
set -e

pushd "$(dirname "$0")"

python test_torch.py
python test_autograd.py
python test_nn.py
python test_legacy_nn.py
python test_multiprocessing.py
if which nvcc >/dev/null 2>&1
then
    python test_cuda.py
else
    echo "nvcc not found in PATH, skipping CUDA tests"
fi

popd
