#!/usr/bin/env bash
#
# Step 2 after branch cut is complete.
#
# Creates PR with release only changes.
#
# Prerequisite: Must be  successfully authenticated in aws fbossci account.
#
# Usage (run from root of project):
#  DRY_RUN=disabled ./scripts/release/apply-release-changes.sh
#
# RELEASE_VERSION: Version of this current release

set -eou pipefail

GIT_TOP_DIR=$(git rev-parse --show-toplevel)
RELEASE_VERSION=${RELEASE_VERSION:-$(cut -d'.' -f1-2 "${GIT_TOP_DIR}/version.txt")}
DRY_RUN=${DRY_RUN:-enabled}

echo "Applying to workflows"
for i in .github/workflows/*.yml; do
    sed -i -e s#@main#@"release/${RELEASE_VERSION}"# $i;
done

echo "Applying to templates"
for i in .github/templates/*.yml.j2; do
    sed -i 's#common.checkout(\(.*\))#common.checkout(\1, checkout_pr_head=False)#' $i;
    sed -i -e s#main#"release/${RELEASE_VERSION}"# $i;
done

echo "Applying to changes to linux binary builds"
for i in  ".github/workflows/_binary-build-linux.yml" ".github/workflows/_binary-test-linux.yml"; do
    sed -i "/github.event_name == 'pull_request'/d" $i;
    sed -i -e s#main#"release/${RELEASE_VERSION}"# $i;
done

sed -i -e "/generate_ci_workflows.py/i \\\t\t\t\texport RELEASE_VERSION_TAG=${RELEASE_VERSION}" .github/workflows/lint.yml

# Triton wheel
echo "Triton Changes"
sed -i -e s#-\ main#"-\ release\/${RELEASE_VERSION}"# .github/workflows/build-triton-wheel.yml

# XLA related changes
echo "XLA Changes"
sed -i -e s#--quiet#-b\ r"${RELEASE_VERSION}"# .ci/pytorch/common_utils.sh
sed -i -e s#.*#r"${RELEASE_VERSION}"# .github/ci_commit_pins/xla.txt

# Regenerate templates
export RELEASE_VERSION_TAG=${RELEASE_VERSION}
./.github/regenerate.sh

# Pin Unstable and disabled jobs and tests
UNSTABLE_VER=$(aws s3api list-object-versions --bucket ossci-metrics --prefix unstable-jobs.json --query 'Versions[?IsLatest].[VersionId]' --output text)
DISABLED_VER=$(aws s3api list-object-versions --bucket ossci-metrics --prefix disabled-jobs.json --query 'Versions[?IsLatest].[VersionId]' --output text)
SLOW_VER=$(aws s3api list-object-versions --bucket ossci-metrics --prefix slow-tests.json --query 'Versions[?IsLatest].[VersionId]' --output text)
DISABLED_TESTS_VER=$(aws s3api list-object-versions --bucket ossci-metrics --prefix disabled-tests-condensed.json --query 'Versions[?IsLatest].[VersionId]' --output text)
sed -i -e s#unstable-jobs.json#"unstable-jobs.json?versionId=${UNSTABLE_VER}"# .github/scripts/filter_test_configs.py
sed -i -e s#disabled-jobs.json#"disabled-jobs.json?versionId=${DISABLED_VER}"# .github/scripts/filter_test_configs.py
sed -i -e s#disabled-tests-condensed.json#"disabled-tests-condensed.json?versionId=${DISABLED_TESTS_VER}"# tools/stats/import_test_stats.py
# Optional
git commit -m "[RELEASE-ONLY CHANGES] Branch Cut for Release {RELEASE_VERSION}"
git push origin "${RELEASE_BRANCH}"
