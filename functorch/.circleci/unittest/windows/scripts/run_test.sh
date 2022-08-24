#!/usr/bin/env bash

set -e

export IN_CI=1
mkdir test-reports

eval "$(./conda/Scripts/conda.exe 'shell.bash' 'hook')"
conda activate ./env

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$this_dir/set_cuda_envs.sh"

python -m torch.utils.collect_env

EXIT_STATUS=0
# TODO: we should be able to acquire the following from some bash commands
# Tests currently ordered in order of runtime...
python test/test_eager_transforms.py -v || EXIT_STATUS=$?
python test/test_compile_cache.py -v || EXIT_STATUS=$?
python test/test_minifier.py -v || EXIT_STATUS=$?
python test/test_memory_efficient_fusion.py -v || EXIT_STATUS=$?
python test/test_pythonkey.py -v || EXIT_STATUS=$?
python test/test_vmap.py -v || EXIT_STATUS=$?
python test/test_ops.py -v || EXIT_STATUS=$?
exit $EXIT_STATUS
