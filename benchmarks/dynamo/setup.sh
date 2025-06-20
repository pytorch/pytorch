#!/bin/bash

# Setup the environment for running benchmarks.
# Assumptions:
#   a) a conda environment with pytorch is already activated.
#   b) run from `pytorch/benchmarks/dynamo`
#
# Usage:
#   $0 <benchmark_name> <model_name>
#
# <benchmark_name> needs to be "timm", "huggingface", or "torchbench"
# for "timm" and "huggingface", <model_name> is ignored
# for "torchbench", only models in <model_name> is installed. Support
# single model name (e.g., "alexnet"), or multiple model names (e.g.,
# "alexnet basic_gnn_gcn BERT_pytorch"), or empty (i.e., "") for installing
# all models.


BENCHMARK=$1
MODELS=$2
HARDWARE="cuda"

# Check if the first argument is provided
if [ -z "$1" ]; then
    echo "Error: No benchmark name provided."
    echo "Usage: $0 <benchmark_name> <model_name>"
    exit 1
fi
# Check if $1 is one of the allowed values
case "$1" in
    timm|huggingface|torchbench)
        echo "Valid benchmark name: $1"
        ;;
    *)
        echo "Error: Unrecognized benchmark name '$1'."
        echo "Should choose from timm, huggingface, and torchbench."
        exit 1
        ;;
esac

# Setup common dependencies
source "../../.ci/pytorch/common.sh"
pip install pandas scipy tqdm
(cd ../.. && install_torchvision) # ETA: 1 min

if [ "$1" == "timm" ] || [ "$1" == "huggingface" ]; then
    # timm and huggingface only require torchvision
    exit 0
elif [ "$1" == "torchbench" ]; then
    (cd ../.. && install_torchaudio $HARDWARE) # ETA: 1 min
    (cd ../.. && checkout_install_torchbench $MODELS) # ETA: 0.5 min per model
fi

# only dlrm needs fbgemm and torchrec
if [[ "$HARDWARE" != "cpu" && "$MODELS" == *"dlrm"* ]]; then
    # Do this after checkout_install_torchbench to ensure we clobber any
    # nightlies that torchbench may pull in
    (cd ../.. && install_torchrec_and_fbgemm) # ETA: 3 min
fi
