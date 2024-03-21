#!/bin/bash
__doc__="
This script simply runs the torch doctests via the xdoctest runner.

This must be run from the root of the torch repo, as it needs the path to the
torch source code.

This script is provided as a developer convenience. On the CI the doctests are
invoked in 'run_test.py'
"
# To simply list tests
# xdoctest -m torch --style=google list

# Reference: https://stackoverflow.com/questions/59895/bash-script-dir
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TORCH_MODPATH=$SCRIPT_DIR/../torch
echo "TORCH_MODPATH = $TORCH_MODPATH"

if [[ ! -d "$TORCH_MODPATH" ]] ; then
    echo "Could not find the path to the torch module"
else
    export XDOCTEST_GLOBAL_EXEC="from torch import nn\nimport torch.nn.functional as F\nimport torch"
    export XDOCTEST_OPTIONS="+IGNORE_WHITESPACE"
    # Note: google wont catch numpy style docstrings (a few exist) but it also wont fail
    # on things not intended to be doctests.
    export XDOCTEST_STYLE="google"
    xdoctest torch "$TORCH_MODPATH" --style="$XDOCTEST_STYLE" --global-exec "$XDOCTEST_GLOBAL_EXEC" --options="$XDOCTEST_OPTIONS"
fi
