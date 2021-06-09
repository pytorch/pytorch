#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

python -m torch.utils.collect_env

# test_functorch_lagging_op_db.py: Only run this locally because it checks
# the functorch lagging op db vs PyTorch's op db.
#
# test_pythonkey.py: No CI for this yet because I need to figure out
# how to actually build PyTorch with LLVM and set up a configuration
# for it.
find test \( -name test\*.py ! -name test_functorch_lagging_op_db.py ! -name test_pythonkey.py \) | xargs -I {} -n 1 bash -c "python {} || exit 255"
