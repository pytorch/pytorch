#!/usr/bin/env bash
set -ex

CACHE_DIRECTORY="/tmp/.lintbin"
# Try to recover the cached binaries
if [[ -d "${CACHE_DIRECTORY}" ]]; then
    # It's ok to fail this as lintrunner init would download these binaries
    # again if they do not exist
    cp -r "${CACHE_DIRECTORY}" . || true
fi

# Do build steps necessary for linters
if [[ "${CLANG}" == "1" ]]; then
    spin regenerate-clangtidy-files
fi

spin regenerate-version
spin regenerate-type-stubs

# Also check generated pyi files
find torch -name '*.pyi' -exec git add --force -- "{}" +

RC=0
# Run lintrunner on all files
if ! spin lint -- --force-color --tee-json=lint.json ${ADDITIONAL_LINTRUNNER_ARGS} 2> /dev/null; then
    echo ""
    echo -e "\e[1m\e[36mYou can reproduce these results locally by using \`lintrunner -m origin/main\`. (If you don't get the same results, run \'lintrunner init\' to update your local linter)\e[0m"
    echo -e "\e[1m\e[36mSee https://github.com/pytorch/pytorch/wiki/lintrunner for setup instructions. To apply suggested patches automatically, use the -a flag. Before pushing another commit,\e[0m"
    echo -e "\e[1m\e[36mplease verify locally and ensure everything passes.\e[0m"
    RC=1
fi

# Unstage temporally added pyi files
find torch -name '*.pyi' -exec git restore --staged -- "{}" +

# Use jq to massage the JSON lint output into GitHub Actions workflow commands.
jq --raw-output \
    '"::\(if .severity == "advice" or .severity == "disabled" then "warning" else .severity end) file=\(.path),line=\(.line),col=\(.char),title=\(.code) \(.name)::" + (.description | gsub("\\n"; "%0A"))' \
    lint.json || true

exit $RC
