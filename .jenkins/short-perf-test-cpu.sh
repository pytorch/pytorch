#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

cd .jenkins/perf_test

export PATH=/opt/conda/bin:$PATH

echo "Running CPU perf test for PyTorch..."

# Include tests
. ./test_cpu_speed_mini_sequence_labeler.sh
. ./test_cpu_speed_mnist.sh

# Run tests
run_test test_cpu_speed_mini_sequence_labeler compare_with_baseline
run_test test_cpu_speed_mnist compare_with_baseline
