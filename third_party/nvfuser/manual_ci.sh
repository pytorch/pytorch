#!/bin/bash

failed_tests=false

run_test() {
  eval "$1"
  status=$?
  if [ $status -ne 0 ];
  then
    failed_tests=true
    echo "============================================================="
    echo "= test_failed!"
    echo "= $1"
    echo "============================================================="
  fi
}

cd "$(dirname "${BASH_SOURCE[0]}")"

run_test './bin/nvfuser_tests'
run_test 'python python_tests/test_dynamo.py'
run_test 'python python_tests/test_python_frontend.py'
run_test 'PYTORCH_TEST_WITH_SLOW=1 python python_tests/test_torchscript.py'

if $failed_tests;
then
  echo "=== CI tests failed, do NOT merge your PR! ==="
  exit 1
else
  echo "=== CI tests passed, ship it! ==="
  exit 0
fi
