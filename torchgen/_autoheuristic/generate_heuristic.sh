#!/bin/bash

if [ $# -lt 8 ]; then
    echo "Error: This script requires exactly at least 8 arguments."
    exit 1
fi

MODE=$1
GPU_DEVICE_IDS=$2
CONDA_ENV=$3
NUM_SAMPLES=$4
OUTPUT_DIR=$5
HEURISTIC_NAME=$6
BENCHMARK_SCRIPT=$7
TRAIN_SCRIPT=$8
EXTRA_TRAIN_ARGS=$9

mkdir -p ${OUTPUT_DIR}

if [ "$MODE" = "collect" ]; then
    # this will collect data for NUM_SAMPLES samples on the number of GPUs specified in GPU_DEVICE_IDS in parallel
    bash ../collect_data.sh "python ${BENCHMARK_SCRIPT}" ${GPU_DEVICE_IDS} ${NUM_SAMPLES} ${CONDA_ENV} ${OUTPUT_DIR}
elif [ "$MODE" = "generate" ]; then
    # the bash script above generates one separate txt file per GPU
    # if GPU_DEVICE_IDS=6,7, it will generate "data_6.txt", "data_7.txt" inside OUTPUT_DIR
    # these files have to be merged into a single file before we can use AutoHeuristic to learn a heuristic
    OUTPUT_FILE="${OUTPUT_DIR}/${HEURISTIC_NAME}.txt"
    INPUT_FILES=$(echo $GPU_DEVICE_IDS | tr ',' '\n' | sed "s|^|${OUTPUT_DIR}/data_|" | sed 's/$/.txt/')
    python ../merge_data.py ${OUTPUT_FILE} ${INPUT_FILES}

    # This will learn a heuristic and generate the code into torch/_inductor/autoheuristic/artifacts/_${HEURISTIC_NAME}.py
    python ${TRAIN_SCRIPT} ${OUTPUT_FILE} --heuristic-name ${HEURISTIC_NAME} ${EXTRA_TRAIN_ARGS}
else
    echo "Error: Invalid mode ${MODE}. Please use 'collect' or 'generate'."
    exit 1
fi
