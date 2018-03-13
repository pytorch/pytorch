#!/bin/bash

COMPACT_JOB_NAME="short-perf-test-cpu"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

cd .jenkins/perf_test

export PATH=/opt/conda/bin:$PATH

echo "Running CPU perf test for PyTorch..."

# Get last master commit hash
export PYTORCH_COMMIT_ID=$(git log --format="%H" -n 1)

# Get baseline file from https://github.com/yf225/perf-tests
if [ -f /var/lib/jenkins/host-workspace/perf_test_numbers_cpu.json ]; then
    cp /var/lib/jenkins/host-workspace/perf_test_numbers_cpu.json perf_test_numbers_cpu.json
else
    curl https://raw.githubusercontent.com/yf225/perf-tests/master/perf_test_numbers_cpu.json -O
fi

if [[ "$GIT_COMMIT" == *origin/master* ]]; then
    # Prepare new baseline file
    cp perf_test_numbers_cpu.json new_perf_test_numbers_cpu.json
    python update_commit_hash.py new_perf_test_numbers_cpu.json ${PYTORCH_COMMIT_ID}
fi

# Include tests
. ./test_cpu_speed_mini_sequence_labeler.sh
. ./test_cpu_speed_mnist.sh

# Run tests
if [[ "$GIT_COMMIT" == *origin/master* ]]; then
    run_test test_cpu_speed_mini_sequence_labeler 20 compare_and_update
    run_test test_cpu_speed_mnist 20 compare_and_update
else
    run_test test_cpu_speed_mini_sequence_labeler 20 compare_with_baseline
    run_test test_cpu_speed_mnist 20 compare_with_baseline
fi

if [[ "$GIT_COMMIT" == *origin/master* ]]; then
    # Push new baseline file
    cp new_perf_test_numbers_cpu.json /var/lib/jenkins/host-workspace/perf_test_numbers_cpu.json
    cd /var/lib/jenkins/host-workspace
    git config --global user.email jenkins@ci.pytorch.org
    git config --global user.name Jenkins
    git add perf_test_numbers_cpu.json
    git commit -m "New CPU perf test baseline from ${PYTORCH_COMMIT_ID}"
fi
