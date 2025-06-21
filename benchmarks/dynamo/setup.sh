#!/bin/bash

# Setup the environment for running benchmarks.
# Assumptions: run from `pytorch/benchmarks/dynamo`
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
VENV_NAME="$BENCHMARK-venv"


# Check if the first argument is provided
if [ -z "$BENCHMARK" ]; then
    echo "Error: No benchmark name provided."
    echo "Usage: $0 <benchmark_name> <model_name>"
    exit 1
fi
# Check if $BENCHMARK is one of the allowed values
case "$BENCHMARK" in
    timm|huggingface|torchbench)
        echo "Valid benchmark name: $BENCHMARK"
        ;;
    *)
        echo "Error: Unrecognized benchmark name '$BENCHMARK'."
        echo "Should choose from timm, huggingface, and torchbench."
        exit 1
        ;;
esac

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv $VENV_NAME --python 3.12 && source $VENV_NAME/bin/activate

(cd ../.. && uv pip install -r requirements.txt)
(cd ../.. && uv pip install mkl mkl-static mkl-include scipy protobuf numba \
    cython scikit-learn librosa tqdm pandas tabulate)

# Build PyTorch locally
python -m ensurepip --upgrade # install pip to uv environment
(cd ../.. && make triton)
(cd ../.. && python setup.py develop)

# Build torchvision locally. ETA: 1 min
(cd ../../../torchvision && python setup.py develop)

if [ "$BENCHMARK" == "torchbench" ]; then
    # Build torchaudio locally. ETA: 1 min
    (cd ../../../torchaudio && uv pip install -e . --no-build-isolation)

    # Build torchbenchmark locally. ETA: 20 seconds per model
    if [ "$MODELS" ]; then
        (cd ../../../torchbenchmark && python install.py --continue_on_fail models $MODELS)
    else
        (cd ../../../torchbenchmark && python install.py --continue_on_fail)
    fi
fi

# only dlrm needs fbgemm and torchrec
if [[ "$MODELS" == *"dlrm"* ]]; then
    uv pip uninstall torchrec-nightly fbgemm-gpu-nightly
    uv pip install setuptools-git-versioning scikit-build pyre-extensions
    # Do this after checkout_install_torchbench to ensure we clobber any
    # nightlies that torchbench may pull in
    (cd ../../../torchrec && uv pip install -e .)

    # current fbgemm pin is too old such that git checkout would fail.
    # So install from git instead.
    fbgemm_commit="$(cat ../../.github/ci_commit_pins/fbgemm.txt)"
    python3 -m pip install --progress-bar off --no-use-pep517 \
        "git+https://github.com/pytorch/FBGEMM.git@${fbgemm_commit}#egg=fbgemm-gpu&subdirectory=fbgemm_gpu"
fi

echo "Activate with: source $VENV_NAME/bin/activate"
