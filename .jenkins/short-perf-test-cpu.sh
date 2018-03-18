#!/bin/bash

COMPACT_JOB_NAME="short-perf-test-cpu"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

cd .jenkins/perf_test

export PATH=/opt/conda/bin:$PATH

pip install GitPython sqlalchemy psycopg2-binary

echo "Running CPU perf test for PyTorch..."

# Get last master commit hash
export PYTORCH_COMMIT_ID=$(git log --format="%H" -n 1)

# Get baseline data from database
if [ -z ${BUILD_ID} ]; then
    python get_baseline.py --local --testtype cpu_runtime --datafile perf_test_numbers_cpu.json
else
    python get_baseline.py --username ${USERNAME} --password ${PASSWORD} --hostname ${DBHOSTNAME} --dbname ${DBNAME} --testtype cpu_runtime --datafile perf_test_numbers_cpu.json
fi

if [[ "$COMMIT_SOURCE" == *master* ]]; then
    # Prepare new baseline file
    cp perf_test_numbers_cpu.json new_perf_test_numbers_cpu.json
    python update_commit_hash.py new_perf_test_numbers_cpu.json ${PYTORCH_COMMIT_ID}
fi

# Include tests
. ./test_cpu_speed_mini_sequence_labeler.sh
. ./test_cpu_speed_mnist.sh

# Run tests
if [[ "$COMMIT_SOURCE" == *master* ]]; then
    run_test test_cpu_speed_mini_sequence_labeler 20 compare_and_update
    run_test test_cpu_speed_mnist 20 compare_and_update
else
    run_test test_cpu_speed_mini_sequence_labeler 20 compare_with_baseline
    run_test test_cpu_speed_mnist 20 compare_with_baseline
fi

# Push new baseline data to database
if [[ "$COMMIT_SOURCE" == *master* ]]; then
    if [ -z ${BUILD_ID} ]; then
        python update_baseline.py --local --testtype cpu_runtime --datafile new_perf_test_numbers_cpu.json
    else
        python update_baseline.py --username ${USERNAME} --password ${PASSWORD} --hostname ${DBHOSTNAME} --dbname ${DBNAME} --testtype cpu_runtime --datafile new_perf_test_numbers_cpu.json
    fi
fi
