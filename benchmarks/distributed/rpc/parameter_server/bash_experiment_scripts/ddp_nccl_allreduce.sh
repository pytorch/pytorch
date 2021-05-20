#!/bin/sh

# requires slurm
# configuration ids
benchmark=3
data="DummyData"
model="DummyModel"
trainer="DdpNcclTrainer"
server="None"
# moves to directory and runs the benchmark with the configurations selected
cd "$(dirname $(dirname "$0"))"
source ./bash_experiment_scripts/helper_functions.sh
run_benchmark_basic "$benchmark" "$data" "$model" "$trainer" "$server"
