#!/bin/sh

# requires slurm
# configuration ids
bconfig_id=3
dconfig_id="DummyData"
mconfig_id="DummyModel"
tconfig_id="DdpNcclTrainer"
pconfig_id="None"
# moves to directory and runs the benchmark with the configurations selected
cd "$(dirname $(dirname "$0"))"
source ./bash_experiment_scripts/helper_functions.sh
run_benchmark_basic "$bconfig_id" "$dconfig_id" "$mconfig_id" "$tconfig_id" "$pconfig_id"
