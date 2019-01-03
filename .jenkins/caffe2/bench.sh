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

cmd="$PYTHON $CAFFE2_PYPATH/python/examples/resnet50_trainer.py --train_data null --batch_size 64 --epoch_size 6400 --num_epochs 2"
if (( $num_gpus == 0 )); then
    cmd="$cmd --use_cpu"
else
    cmd="$cmd --num_gpus 1"
fi

eval "$cmd"
