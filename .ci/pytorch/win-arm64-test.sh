#!/bin/bash
set -ex -o pipefail

SCRIPT_PARENT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
# shellcheck source=./common.sh
source "$SCRIPT_PARENT_DIR/common.sh"

call %PYTORCH_ROOT%\.ci\pytorch\windows\arm64\bootstrap_tests.bat

run_tests() {

    echo Running smoke_test.py...
    python %PYTORCH_ROOT%\.ci\pytorch\smoke_test\smoke_test.py --package torchonly

    echo Running test_autograd.oy, test_nn.py, test_torch.py...
    push %PYTORCH_ROOT%\test
    set CORE_TEST_LIST=test_autograd.py test_nn.py test_torch.py

    for /L %%i in (1,1,%1) do (
        for %%t in (%CORE_TEST_LIST%) do (
            echo Running test: %%t
            python %%t --verbose --save-xml --use-pytest -vvvv -rfEsxXP -p no:xdist
        )
    ) 
}

run_tests
echo "TEST PASSED"
