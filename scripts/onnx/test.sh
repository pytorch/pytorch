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

if [[ $PARALLEL == 1 ]]; then
    pytest -n 3 "${test_paths[@]}"
else
    pytest "${test_paths[@]}"
fi
