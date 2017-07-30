#!/bin/sh
set -e

# test import
python3 -c "import torch"

# run unit tests
cd test; bash run_test.sh -p python3.5

exit 0
