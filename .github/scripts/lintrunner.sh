#!/usr/bin/env bash
set -x

# Use uv to speed up lintrunner init
python3 -m pip install uv==0.1.45 setuptools

CACHE_DIRECTORY="/tmp/.lintbin"
# Try to recover the cached binaries
if [[ -d "${CACHE_DIRECTORY}" ]]; then
    # It's ok to fail this as lintrunner init would download these binaries
    # again if they do not exist
    cp -r "${CACHE_DIRECTORY}" . || true
fi

# if lintrunner is not installed, install it
if ! command -v lintrunner &> /dev/null; then
    python3 -m pip install lintrunner==0.12.7
fi

# This has already been cached in the docker image
lintrunner init 2> /dev/null

python3 -m tools.linter.clang_tidy.generate_build_files
python3 -m tools.generate_torch_version --is_debug=false
python3 -m tools.pyi.gen_pyi \
    --native-functions-path aten/src/ATen/native/native_functions.yaml \
    --tags-path aten/src/ATen/native/tags.yaml \
    --deprecated-functions-path "tools/autograd/deprecated.yaml"
python3 torch/utils/data/datapipes/gen_pyi.py

# Also check generated pyi files
find torch -name '*.pyi' -exec git add --force -- "{}" +
for linter in $(lintrunner list 2>/dev/null| tail -n +2); do
  echo ""
  time lintrunner --force-color --tee-json=lint.json --take "${linter}" --all-files 2> /dev/null
done

RC=0
# Run lintrunner on all files

# Unstage temporally added pyi files
find torch -name '*.pyi' -exec git restore --staged -- "{}" +

# Use jq to massage the JSON lint output into GitHub Actions workflow commands.
jq --raw-output \
    '"::\(if .severity == "advice" or .severity == "disabled" then "warning" else .severity end) file=\(.path),line=\(.line),col=\(.char),title=\(.code) \(.name)::" + (.description | gsub("\\n"; "%0A"))' \
    lint.json || true

exit $RC
