#!/bin/bash
# =================== The following code **should** be executed inside Docker container ===================

# Install dependencies
sudo apt-get -y update
sudo apt-get -y install expect-dev

# This is where the local pytorch install in the docker image is located
pt_checkout="/var/lib/jenkins/workspace"
source "$pt_checkout/.jenkins/pytorch/common_utils.sh"
echo "functorch_doc_push_script.sh: Invoked with $*"

set -ex

version=${DOCS_VERSION:-nightly}
echo "version: $version"

# Build functorch docs
pushd $pt_checkout/functorch/docs
pip -q install -r requirements.txt
make html
popd

git clone https://github.com/pytorch/functorch -b gh-pages --depth 1 functorch_ghpages
pushd functorch_ghpages

if [ $version == "master" ]; then
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
# =================== The above code **should** be executed inside Docker container ===================
