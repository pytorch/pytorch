#!/bin/bash

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Anywhere except $ROOT_DIR should work. This is so the python import doesn't
# get confused by any 'caffe2' directory in cwd
cd "$INSTALL_PREFIX"

if [[ $BUILD_ENVIRONMENT == *-cuda* ]]; then
    num_gpus=$(nvidia-smi -L | wc -l)
elif [[ $BUILD_ENVIRONMENT == *-rocm* ]]; then
    num_gpus=$(rocminfo | grep 'Device Type.*GPU' | wc -l)
else
    num_gpus=0
fi

caffe2_pypath="$(cd /usr && $PYTHON -c 'import os; import caffe2; print(os.path.dirname(os.path.realpath(caffe2.__file__)))')"
# Resnet50
if (( $num_gpus == 0 )); then
    "$PYTHON" "$caffe2_pypath/python/examples/imagenet_trainer.py" --train_data null --batch_size 128 --epoch_size 12800 --num_epochs 2 --use_cpu
fi
if (( $num_gpus >= 1 )); then
    "$PYTHON" "$caffe2_pypath/python/examples/imagenet_trainer.py" --train_data null --batch_size 128 --epoch_size 12800 --num_epochs 2 --num_gpus 1
    # Let's skip the fp16 bench runs for now, as it recompiles the miopen kernels and can take 10+min to run.
    # We can resume when we (1) bindmount the miopen cache folder in jenkins; (2) install the pre-compiled miopen kernel library in the docker
    # "$PYTHON" "$caffe2_pypath/python/examples/imagenet_trainer.py" --train_data null --batch_size 256 --epoch_size 25600 --num_epochs 2 --num_gpus 1 --float16_compute --dtype float16
fi
if (( $num_gpus >= 4 )); then
    "$PYTHON" "$caffe2_pypath/python/examples/imagenet_trainer.py" --train_data null --batch_size 512 --epoch_size 51200 --num_epochs 2 --num_gpus 4
fi

# ResNext
if (( $num_gpus == 0 )); then
    "$PYTHON" "$caffe2_pypath/python/examples/imagenet_trainer.py" --resnext_num_groups 32 --resnext_width_per_group 4 --num_layers 101 --train_data null --batch_size 32 --epoch_size 3200 --num_epochs 2 --use_cpu
fi
if (( $num_gpus >= 1 )); then
    "$PYTHON" "$caffe2_pypath/python/examples/imagenet_trainer.py" --resnext_num_groups 32 --resnext_width_per_group 4 --num_layers 101 --train_data null --batch_size 32 --epoch_size 3200 --num_epochs 2 --num_gpus 1
    # "$PYTHON" "$caffe2_pypath/python/examples/imagenet_trainer.py" --resnext_num_groups 32 --resnext_width_per_group 4 --num_layers 101 --train_data null --batch_size 64 --epoch_size 3200 --num_epochs 2 --num_gpus 1 --float16_compute --dtype float16
fi
if (( $num_gpus >= 4 )); then
    "$PYTHON" "$caffe2_pypath/python/examples/imagenet_trainer.py" --resnext_num_groups 32 --resnext_width_per_group 4 --num_layers 101 --train_data null --batch_size 128 --epoch_size 12800 --num_epochs 2 --num_gpus 4
fi

# Shufflenet
if (( $num_gpus == 0 )); then
    "$PYTHON" "$caffe2_pypath/python/examples/imagenet_trainer.py" --train_data null --batch_size 32 --epoch_size 3200 --num_epochs 2 --use_cpu --model shufflenet
fi
if (( $num_gpus >= 1 )); then
    "$PYTHON" "$caffe2_pypath/python/examples/imagenet_trainer.py" --train_data null --batch_size 32 --epoch_size 3200 --num_epochs 2 --num_gpus 1 --model shufflenet
fi
if (( $num_gpus >= 4 )); then
    "$PYTHON" "$caffe2_pypath/python/examples/imagenet_trainer.py" --train_data null --batch_size 128 --epoch_size 12800 --num_epochs 2 --num_gpus 4 --model shufflenet
fi
