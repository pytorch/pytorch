#!/usr/bin/env bash
set -ex

# The generic Linux job chooses to use base env, not the one setup by the image
CONDA_ENV=$(conda env list --json | jq -r ".envs | .[-1]")
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate "${CONDA_ENV}"

# Use uv to speed up lintrunner init
python3 -m pip install uv==0.1.45

CACHE_DIRECTORY="/tmp/.lintbin"
# Try to recover the cached binaries
if [[ -d "${CACHE_DIRECTORY}" ]]; then
    # It's ok to fail this as lintrunner init would download these binaries
    # again if they do not exist
    cp -r "${CACHE_DIRECTORY}" . || true
fi

# This has already been cached in the docker image
lintrunner init 2> /dev/null

# Do build steps necessary for linters
if [[ "${CLANG}" == "1" ]]; then
    python3 -m tools.linter.clang_tidy.generate_build_files
fi
python3 -m tools.generate_torch_version --is_debug=false
python3 -m tools.pyi.gen_pyi \
    --native-functions-path aten/src/ATen/native/native_functions.yaml \
    --tags-path aten/src/ATen/native/tags.yaml \
    --deprecated-functions-path "tools/autograd/deprecated.yaml"
python3 torch/utils/data/datapipes/gen_pyi.py

RC=0
# Run lintrunner on all files
if ! lintrunner --force-color --all-files --tee-json=lint.json ${ADDITIONAL_LINTRUNNER_ARGS} 2> /dev/null; then
    echo ""
    echo -e "\e[1m\e[36mYou can reproduce these results locally by using \`lintrunner -m origin/main\`. (If you don't get the same results, run \'lintrunner init\' to update your local linter)\e[0m"
    echo -e "\e[1m\e[36mSee https://github.com/pytorch/pytorch/wiki/lintrunner for setup instructions.\e[0m"
    RC=1
fi

# Use jq to massage the JSON lint output into GitHub Actions workflow commands.
jq --raw-output \
    '"::\(if .severity == "advice" or .severity == "disabled" then "warning" else .severity end) file=\(.path),line=\(.line),col=\(.char),title=\(.code) \(.name)::" + (.description | gsub("\\n"; "%0A"))' \
    lint.json || true

exit $RC
