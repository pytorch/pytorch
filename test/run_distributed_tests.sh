#!/usr/bin/env bash

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
  BACKEND=mpi mpiexec -n 3 --noprefix $PYCMD ./test_distributed.py
  distributed_tear_down

  echo "Running distributed tests for the MPI backend with file init_method"
  distributed_set_up
  BACKEND=mpi INIT_METHOD='file://'$TEMP_DIR'/shared_init_file' mpiexec -n 3 --noprefix $PYCMD ./test_distributed.py
  distributed_tear_down
else
  echo "Skipping MPI backend tests (MPI not found)"
fi

echo "Running distributed tests for the Nccl backend"
distributed_set_up
BACKEND=nccl WORLD_SIZE=2 $PYCMD ./test_distributed.py
distributed_tear_down

echo "Running distributed tests for the Nccl backend with file init_method"
distributed_set_up
BACKEND=nccl WORLD_SIZE=2 INIT_METHOD='file://'$TEMP_DIR'/shared_init_file' $PYCMD ./test_distributed.py
distributed_tear_down
