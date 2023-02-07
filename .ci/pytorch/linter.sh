#!/bin/bash

# Do build steps necessary for linters
python3 -m tools.linter.clang_tidy.generate_build_files
python3 -m tools.generate_torch_version --is_debug=false
python3 -m tools.pyi.gen_pyi \
  --native-functions-path aten/src/ATen/native/native_functions.yaml \
  --tags-path aten/src/ATen/native/tags.yaml \
  --deprecated-functions-path "tools/autograd/deprecated.yaml"

RC=0
# Run lintrunner on all files
if ! lintrunner --force-color --all-files --tee-json=lint.json; then
  echo ""
  echo -e "\e[1m\e[36mYou can reproduce these results locally by using \`lintrunner\`.\e[0m"
  echo -e "\e[1m\e[36mSee https://github.com/pytorch/pytorch/wiki/lintrunner for setup instructions.\e[0m"
  RC=1
fi

# Store annotations
if [[ "${EVENT_NAME}" == "pull_request" ]]; then
  # Use jq to massage the JSON lint output into GitHub Actions workflow commands.
  jq --raw-output \
    '"::\(if .severity == "advice" or .severity == "disabled" then "warning" else .severity end) file=\(.path),line=\(.line),col=\(.char),title=\(.code) \(.name)::" + (.description | gsub("\\n"; "%0A"))' \
    lint.json || true
fi

exit $RC
