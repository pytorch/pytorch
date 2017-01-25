#!/usr/bin/env bash
set -e

PYCMD=${PYCMD:="python"}
COVERAGE=0
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -p|--python) PYCMD=$2; shift 2 ;;
        -c|--coverage) COVERAGE=1; shift 2 ;;
        --) shift; break ;;
        *) echo "Invalid argument: $1!" ; exit 1 ;;
    esac
done

if [[ $COVERAGE -eq 1 ]]; then
    coverage erase
    PYCMD="coverage run --parallel-mode --source torch "
    echo "coverage flag found. Setting python command to: \"$PYCMD\""
fi

pushd "$(dirname "$0")"

echo "Running torch tests"
$PYCMD test_torch.py $@

echo "Running autograd tests"
$PYCMD test_autograd.py $@

echo "Running sparse tests"
$PYCMD test_sparse.py $@

echo "Running nn tests"
$PYCMD test_nn.py $@

echo "Running legacy nn tests"
$PYCMD test_legacy_nn.py $@

echo "Running optim tests"
$PYCMD test_optim.py $@

echo "Running multiprocessing tests"
$PYCMD test_multiprocessing.py $@
MULTIPROCESSING_METHOD=spawn $PYCMD test_multiprocessing.py $@
MULTIPROCESSING_METHOD=forkserver $PYCMD test_multiprocessing.py $@

echo "Running util tests"
$PYCMD test_utils.py $@

echo "Running dataloader tests"
$PYCMD test_dataloader.py $@

echo "Running cuda tests"
$PYCMD test_cuda.py $@

echo "Running NCCL tests"
$PYCMD test_nccl.py $@

if [[ $COVERAGE -eq 1 ]]; then
    coverage combine
    coverage html
fi

popd
