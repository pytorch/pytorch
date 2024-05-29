#!/bin/bash

SCRIPT_PARENT_DIR=$(dirname "${BASH_SOURCE[0]}")

# shellcheck source=.ci/pytorch/common.sh
source "$SCRIPT_PARENT_DIR/common.sh"

cd .ci/pytorch/perf_test

echo "Running CPU perf test for PyTorch..."

pip install -q awscli

# Set multipart_threshold to be sufficiently high, so that `aws s3 cp` is not a multipart read
# More info at https://github.com/aws/aws-cli/issues/2321
aws configure set default.s3.multipart_threshold 5GB
UPSTREAM_DEFAULT_BRANCH="$(git remote show https://github.com/pytorch/pytorch.git | awk '/HEAD branch/ {print $NF}')"

if [[ "$COMMIT_SOURCE" == "$UPSTREAM_DEFAULT_BRANCH" ]]; then
    # Get current default branch commit hash
    DEFAULT_BRANCH_COMMIT_ID=$(git log --format="%H" -n 1)
    export DEFAULT_BRANCH_COMMIT_ID
fi

# Find the default branch commit to test against
git remote add upstream https://github.com/pytorch/pytorch.git
git fetch upstream
IFS=$'\n'
while IFS='' read -r commit_id; do
    if aws s3 ls s3://ossci-perf-test/pytorch/cpu_runtime/"${commit_id}".json; then
        LATEST_TESTED_COMMIT=${commit_id}
        break
    fi
done < <(git rev-list upstream/"$UPSTREAM_DEFAULT_BRANCH")
aws s3 cp s3://ossci-perf-test/pytorch/cpu_runtime/"${LATEST_TESTED_COMMIT}".json cpu_runtime.json

if [[ "$COMMIT_SOURCE" == "$UPSTREAM_DEFAULT_BRANCH" ]]; then
    # Prepare new baseline file
    cp cpu_runtime.json new_cpu_runtime.json
    python update_commit_hash.py new_cpu_runtime.json "${DEFAULT_BRANCH_COMMIT_ID}"
fi

# Include tests
# shellcheck source=./perf_test/test_cpu_speed_mini_sequence_labeler.sh
. ./test_cpu_speed_mini_sequence_labeler.sh
# shellcheck source=./perf_test/test_cpu_speed_mnist.sh
. ./test_cpu_speed_mnist.sh
# shellcheck source=./perf_test/test_cpu_speed_torch.sh
. ./test_cpu_speed_torch.sh
# shellcheck source=./perf_test/test_cpu_speed_torch_tensor.sh
. ./test_cpu_speed_torch_tensor.sh

# Run tests
export TEST_MODE="compare_with_baseline"
if [[ "$COMMIT_SOURCE" == "$UPSTREAM_DEFAULT_BRANCH" ]]; then
    export TEST_MODE="compare_and_update"
fi

# Operator tests
run_test test_cpu_speed_torch ${TEST_MODE}
run_test test_cpu_speed_torch_tensor ${TEST_MODE}

# Sample model tests
run_test test_cpu_speed_mini_sequence_labeler 20 ${TEST_MODE}
run_test test_cpu_speed_mnist 20 ${TEST_MODE}

if [[ "$COMMIT_SOURCE" == "$UPSTREAM_DEFAULT_BRANCH" ]]; then
    # This could cause race condition if we are testing the same default branch commit twice,
    # but the chance of them executing this line at the same time is low.
    aws s3 cp new_cpu_runtime.json s3://ossci-perf-test/pytorch/cpu_runtime/"${DEFAULT_BRANCH_COMMIT_ID}".json --acl public-read
fi
