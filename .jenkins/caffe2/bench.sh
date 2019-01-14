#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Anywhere except $ROOT_DIR should work
cd "$INSTALL_PREFIX"

if [[ $BUILD_ENVIRONMENT == *-cuda* ]]; then
    num_gpus=$(nvidia-smi -L | wc -l)
elif [[ $BUILD_ENVIRONMENT == *-rocm* ]]; then
    num_gpus=$(rocminfo | grep 'Device Type.*GPU' | wc -l)
else
    num_gpus=0
fi

if (( $num_gpus == 0 )); then
    "$PYTHON" "$CAFFE2_PYPATH/python/examples/resnet50_trainer.py" --train_data null --batch_size 64 --epoch_size 6400 --num_epochs 2 --use_cpu
else
    "$PYTHON" "$CAFFE2_PYPATH/python/examples/resnet50_trainer.py" --train_data null --batch_size 64 --epoch_size 6400 --num_epochs 2 --num_gpus 1
    if (( $num_gpus >= 2 )); then
        "$PYTHON" "$CAFFE2_PYPATH/python/examples/resnet50_trainer.py" --train_data null --batch_size 128 --epoch_size 12800 --num_epochs 2 --num_gpus 2
    fi
    if (( $num_gpus >= 4 )); then
        "$PYTHON" "$CAFFE2_PYPATH/python/examples/resnet50_trainer.py" --train_data null --batch_size 256 --epoch_size 25600 --num_epochs 2 --num_gpus 4
    fi
fi
