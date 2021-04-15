#!/bin/sh

# configuration ids
bconfig_id=7
dconfig_id=1
mconfig_id=1

# moves to directory and runs the benchmark with the configurations selected
cd "$(dirname $(dirname "$0"))"
source ./bash_experiment_scripts/helper_functions.sh
run_benchmark_basic "$bconfig_id" "$dconfig_id" "$mconfig_id"