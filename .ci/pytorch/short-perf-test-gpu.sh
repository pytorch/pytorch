#!/bin/bash

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

pushd .ci/pytorch/perf_test

echo "Running GPU perf test for PyTorch..."

# Trying to uninstall PyYAML can cause problem. Workaround according to:
# https://github.com/pypa/pip/issues/5247#issuecomment-415571153
pip install -q awscli --ignore-installed PyYAML

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
    if aws s3 ls s3://ossci-perf-test/pytorch/gpu_runtime/"${commit_id}".json; then
        LATEST_TESTED_COMMIT=${commit_id}
        break
    fi
done < <(git rev-list upstream/"$UPSTREAM_DEFAULT_BRANCH")
aws s3 cp s3://ossci-perf-test/pytorch/gpu_runtime/"${LATEST_TESTED_COMMIT}".json gpu_runtime.json

if [[ "$COMMIT_SOURCE" == "$UPSTREAM_DEFAULT_BRANCH" ]]; then
    # Prepare new baseline file
    cp gpu_runtime.json new_gpu_runtime.json
    python update_commit_hash.py new_gpu_runtime.json "${DEFAULT_BRANCH_COMMIT_ID}"
fi

# Include tests
# shellcheck source=./perf_test/test_gpu_speed_mnist.sh
. ./test_gpu_speed_mnist.sh
# shellcheck source=./perf_test/test_gpu_speed_word_language_model.sh
. ./test_gpu_speed_word_language_model.sh
# shellcheck source=./perf_test/test_gpu_speed_cudnn_lstm.sh
. ./test_gpu_speed_cudnn_lstm.sh
# shellcheck source=./perf_test/test_gpu_speed_lstm.sh
. ./test_gpu_speed_lstm.sh
# shellcheck source=./perf_test/test_gpu_speed_mlstm.sh
. ./test_gpu_speed_mlstm.sh

# Run tests
if [[ "$COMMIT_SOURCE" == "$UPSTREAM_DEFAULT_BRANCH" ]]; then
    run_test test_gpu_speed_mnist 20 compare_and_update
    run_test test_gpu_speed_word_language_model 20 compare_and_update
    run_test test_gpu_speed_cudnn_lstm 20 compare_and_update
    run_test test_gpu_speed_lstm 20 compare_and_update
    run_test test_gpu_speed_mlstm 20 compare_and_update
else
    run_test test_gpu_speed_mnist 20 compare_with_baseline
    run_test test_gpu_speed_word_language_model 20 compare_with_baseline
    run_test test_gpu_speed_cudnn_lstm 20 compare_with_baseline
    run_test test_gpu_speed_lstm 20 compare_with_baseline
    run_test test_gpu_speed_mlstm 20 compare_with_baseline
fi

if [[ "$COMMIT_SOURCE" == "$UPSTREAM_DEFAULT_BRANCH" ]]; then
    # This could cause race condition if we are testing the same default branch commit twice,
    # but the chance of them executing this line at the same time is low.
    aws s3 cp new_gpu_runtime.json s3://ossci-perf-test/pytorch/gpu_runtime/"${DEFAULT_BRANCH_COMMIT_ID}".json --acl public-read
fi

popd
