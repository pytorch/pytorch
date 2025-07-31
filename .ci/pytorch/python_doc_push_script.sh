#!/bin/bash

# This is where the local pytorch install in the docker image is located
pt_checkout="/var/lib/jenkins/workspace"

source "$pt_checkout/.ci/pytorch/common_utils.sh"

echo "python_doc_push_script.sh: Invoked with $*"

set -ex -o pipefail

# for statements like ${1:-${DOCS_INSTALL_PATH:-docs/}}
# the order of operations goes:
#   1. Check if there's an argument $1
#   2. If no argument check for environment var DOCS_INSTALL_PATH
#   3. If no environment var fall back to default 'docs/'

# NOTE: It might seem weird to gather the second argument before gathering the first argument
#       but since DOCS_INSTALL_PATH can be derived from DOCS_VERSION it's probably better to
#       try and gather it first, just so we don't potentially break people who rely on this script
# Argument 2: What version of the docs we are building.
version="${2:-${DOCS_VERSION:-main}}"
if [ -z "$version" ]; then
echo "error: python_doc_push_script.sh: version (arg2) not specified"
  exit 1
fi

# Argument 1: Where to copy the built documentation to
# (pytorch_docs/$install_path)
install_path="${1:-${DOCS_INSTALL_PATH:-${DOCS_VERSION}}}"
if [ -z "$install_path" ]; then
echo "error: python_doc_push_script.sh: install_path (arg1) not specified"
  exit 1
fi

is_main_doc=false
if [ "$version" == "main" ]; then
  is_main_doc=true
fi

# Argument 3: The branch to push to. Usually is "site"
branch="${3:-${DOCS_BRANCH:-site}}"
if [ -z "$branch" ]; then
echo "error: python_doc_push_script.sh: branch (arg3) not specified"
  exit 1
fi

echo "install_path: $install_path  version: $version"


build_docs () {
  set +e
  set -o pipefail
  make "$1" 2>&1 | tee /tmp/docs_build.txt
  code=$?
  if [ $code -ne 0 ]; then
    set +x
    echo =========================
    grep "WARNING:" /tmp/docs_build.txt
    echo =========================
    echo Docs build failed. If the failure is not clear, scan back in the log
    echo for any WARNINGS or for the line "build finished with problems"
    echo "(tried to echo the WARNINGS above the ==== line)"
    echo =========================
  fi
  set -ex -o pipefail
  return $code
}


git clone https://github.com/pytorch/docs pytorch_docs -b "$branch" --depth 1
pushd pytorch_docs

export LC_ALL=C
export PATH=/opt/conda/bin:$PATH
if [ -n "$ANACONDA_PYTHON_VERSION" ]; then
  export PATH=/opt/conda/envs/py_$ANACONDA_PYTHON_VERSION/bin:$PATH
fi

rm -rf pytorch || true

# Get all the documentation sources, put them in one place
pushd "$pt_checkout"
pushd docs

# Profile the docs build to see what is taking the longest
python -m cProfile -o docs_build.prof -m sphinx.cmd.build -b html -d build/doctrees source build/html
python -c "import pstats; p = pstats.Stats('docs_build.prof'); p.sort_stats('cumtime').print_stats(50)"

# Build the docs
if [ "$is_main_doc" = true ]; then
  build_docs html || exit $?

  make coverage
  # Now we have the coverage report, we need to make sure it is empty.
  # Count the number of lines in the file and turn that number into a variable
  # $lines. The `cut -f1 ...` is to only parse the number, not the filename
  # Skip the report header by subtracting 2: the header will be output even if
  # there are no undocumented items.
  #
  # Also: see docs/source/conf.py for "coverage_ignore*" items, which should
  # be documented then removed from there.
  lines=$(wc -l build/coverage/python.txt 2>/dev/null |cut -f1 -d' ')
  undocumented=$((lines - 2))
  if [ $undocumented -lt 0 ]; then
    echo coverage output not found
    exit 1
  elif [ $undocumented -gt 0 ]; then
    echo undocumented objects found:
    cat build/coverage/python.txt
    echo "Make sure you've updated relevant .rsts in docs/source!"
    echo "You can reproduce locally by running 'cd docs && make coverage && cat build/coverage/python.txt'"
    exit 1
  fi
else
  # skip coverage, format for stable or tags
  build_docs html-stable || exit $?
fi

# Move them into the docs repo
popd
popd
git rm -rf "$install_path" || true
mv "$pt_checkout/docs/build/html" "$install_path"

git add "$install_path" || true
git status
git config user.email "soumith+bot@pytorch.org"
git config user.name "pytorchbot"
# If there aren't changes, don't make a commit; push is no-op
git commit -m "Generate Python docs from pytorch/pytorch@${GITHUB_SHA}" || true
git status

if [[ "${WITH_PUSH:-}" == true ]]; then
  # push to a temp branch first to trigger CLA check and satisfy branch protections
  git push -u origin HEAD:pytorchbot/temp-branch-py -f
  git push -u origin HEAD^:pytorchbot/base -f
  sleep 30
  git push -u origin "${branch}"
fi

popd
