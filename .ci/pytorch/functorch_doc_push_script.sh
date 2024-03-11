#!/bin/bash

# This is where the local pytorch install in the docker image is located
pt_checkout="/var/lib/jenkins/workspace"
source "$pt_checkout/.ci/pytorch/common_utils.sh"
echo "functorch_doc_push_script.sh: Invoked with $*"

set -ex

version=${DOCS_VERSION:-nightly}
echo "version: $version"

if [[ "$BUILD_ENVIRONMENT" == linux-jammy-py3.8-gcc11* ]] ; then
  echo "Sometimes in kernel 5+ anaconda decides to use its own libstdc++ instead of the system one. As we've always been relying on the system one, let's make sure it's used."
  sudo rm "/opt/conda/envs/py_${ANACONDA_PYTHON_VERSION}/lib/libstdc++.so.6"
fi

# Build functorch docs
pushd $pt_checkout/functorch/docs
make html
popd

git clone https://github.com/pytorch/functorch -b gh-pages --depth 1 functorch_ghpages
pushd functorch_ghpages

if [ "$version" == "main" ]; then
  version=nightly
fi

git rm -rf "$version" || true
mv "$pt_checkout/functorch/docs/build/html" "$version"

git add "$version" || true
git status
git config user.email "soumith+bot@pytorch.org"
git config user.name "pytorchbot"
# If there aren't changes, don't make a commit; push is no-op
git commit -m "Generate Python docs from pytorch/pytorch@${GITHUB_SHA}" || true
git status

if [[ "${WITH_PUSH:-}" == true ]]; then
  git push -u origin gh-pages
fi

popd
