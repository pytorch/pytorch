#!/usr/bin/env bash
#
# Usage (run from root of project):
#  RELEASE_VERSION=2.1 apply-release-changes.sh
#
# RELEASE_VERSION: Version of this current release

set -eou pipefail

# Create and Check out to Release Branch
# git checkout -b "${RELEASE_BRANCH}"

# Change all GitHub Actions to reference the test-infra release branch
# as opposed to main.
echo "Applying to workflows"
for i in .github/workflows/*.yml; do
    sed -i -e s#@main#@"release/${RELEASE_VERSION}"# $i;
done

# Change all checkout step in templates to not add ref to checkout
echo "Applying to templates"
for i in .github/templates/*.yml.j2; do
    sed -i 's#common.checkout(\(.*\))#common.checkout(\1, checkout_pr_head=False)#' $i;
done
# Change conda token for test env for conda upload
sed -i 's#CONDA_PYTORCHBOT_TOKEN#CONDA_PYTORCHBOT_TOKEN_TEST#' .github/templates/upload.yml.j2

# Triton wheel
echo "Triton Changes"
sed -i -e s#-\ main#"-\ release\/${RELEASE_VERSION}"# .github/workflows/build-triton-wheel.yml

# XLA related changes
echo "XLA Changes"
sed -i -e s#--quiet#-b\ r"${RELEASE_VERSION}"# .ci/pytorch/common_utils.sh
sed -i -e s#.*#r"${RELEASE_VERSION}"# .github/ci_commit_pins/xla.txt

# Binary tests
echo "Binary tests"
sed -i 's#/nightly/#/test/#' .circleci/scripts/binary_linux_test.sh
sed -i 's#"\\${PYTORCH_CHANNEL}"#pytorch-test#' .circleci/scripts/binary_linux_test.sh

# Regenerated templates
./.github/regenerate.sh

# Optional
# git commit -m "[RELEASE-ONLY CHANGES] Branch Cut for Release {RELEASE_VERSION}"
# git push origin "${RELEASE_BRANCH}"
