#!/bin/bash

set -ex

UNKNOWN=()

# defaults
PARALLEL=0

while [[ $# -gt 0 ]]
do
    arg="$1"
    case $arg in
        -p|--parallel)
            PARALLEL=1
            shift # past argument
            ;;
        *) # unknown option
            UNKNOWN+=("$1") # save it in an array for later
            shift # past argument
            ;;
    esac
done
set -- "${UNKNOWN[@]}" # leave UNKNOWN

pip install pytest scipy torchvision
if [[ $PARALLEL == 1 ]]; then
    pip install pytest-xdist
fi

# realpath might not be available on MacOS
script_path=$(python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")
top_dir=$(dirname $(dirname $(dirname "$script_path")))
test_paths=(
    "$top_dir/test/onnx"
)

args=()
args+=("-v")
if [[ $PARALLEL == 1 ]]; then
  args+=("-n")
  args+=("3")
fi

pytest "${args[@]}" \
  -k \
  'not (TestOperators and test_full_like) and not (TestOperators and test_zeros_like) and not (TestOperators and test_ones_like) and not (TestModels and test_super_resolution) and not (TestModels and test_vgg16) and not (TestModels and test_vgg16_bn) and not (TestModels and test_vgg19) and not (TestModels and test_vgg19_bn)' \
  "${test_paths[@]}"
