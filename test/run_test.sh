#!/usr/bin/env bash
set -e

PYCMD=${PYCMD:="python"}
COVERAGE=0
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -p|--python) PYCMD=$2; shift 2 ;;
        -c|--coverage) COVERAGE=1; shift 1;;
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
$PYCMD test_potrf.py $@

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

echo "Running JIT tests"
$PYCMD test_jit.py $@

distributed_set_up() {
  export TEMP_DIR="$(mktemp -d)"
  rm -rf "$TEMP_DIR/"*
  mkdir "$TEMP_DIR/barrier"
  mkdir "$TEMP_DIR/test_dir"
}

distributed_tear_down() {
  rm -rf "$TEMP_DIR"
}

trap distributed_tear_down EXIT SIGHUP SIGINT SIGTERM

echo "Running distributed tests for the TCP backend"
distributed_set_up
BACKEND=tcp WORLD_SIZE=3 $PYCMD ./test_distributed.py
distributed_tear_down

echo "Running distributed tests for the TCP backend with file init_method"
distributed_set_up
BACKEND=tcp WORLD_SIZE=3 INIT_METHOD='file://'$TEMP_DIR'/shared_init_file' $PYCMD ./test_distributed.py
distributed_tear_down

echo "Running distributed tests for the Gloo backend"
distributed_set_up
BACKEND=gloo WORLD_SIZE=3 $PYCMD ./test_distributed.py
distributed_tear_down

echo "Running distributed tests for the Gloo backend with file init_method"
distributed_set_up
BACKEND=gloo WORLD_SIZE=3 INIT_METHOD='file://'$TEMP_DIR'/shared_init_file' $PYCMD ./test_distributed.py
distributed_tear_down

if [ -x "$(command -v mpiexec)" ]; then
  echo "Running distributed tests for the MPI backend"
  distributed_set_up
  BACKEND=mpi mpiexec -n 3 $PYCMD ./test_distributed.py
  distributed_tear_down

  echo "Running distributed tests for the MPI backend with file init_method"
  distributed_set_up
  BACKEND=mpi INIT_METHOD='file://'$TEMP_DIR'/shared_init_file' mpiexec -n 3 $PYCMD ./test_distributed.py
  distributed_tear_down
else
  echo "Skipping MPI backend tests (MPI not found)"
fi

if [[ $COVERAGE -eq 1 ]]; then
    coverage combine
    coverage html
fi

popd
