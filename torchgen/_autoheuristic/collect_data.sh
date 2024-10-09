#!/bin/bash

# this script makes it easy parallize collecting data across using multiple GPUs

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "tmux is not installed. Please install it and try again."
    exit 1
fi

# Check if the correct number of arguments is provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 \"<python_command>\" <comma_separated_device_numbers> <num_samples to generate> <CONDA_ENV> <OUTPUT_DIR>"
    echo "Example: $0 \"python run.py --a b --b c\" 1,4,5,3 1000 pytorch-3.10 a100"
    exit 1
fi

PYTHON_COMMAND=$1
DEVICE_NUMBERS=$2
NUM_SAMPLES=$3
CONDA_ENV=$4
OUTPUT_DIR=$5

# Create a new tmux session
SESSION_NAME="parallel_run_$(date +%s)"
tmux new-session -d -s "$SESSION_NAME"

# Split the device numbers
IFS=',' read -ra DEVICES <<< "$DEVICE_NUMBERS"

NUM_GPUS=${#DEVICES[@]}
NUM_SAMPLES_PER_GPU=$((NUM_SAMPLES / NUM_GPUS))
echo "AutoHeuristic will collect ${NUM_SAMPLES} samples split across ${NUM_GPUS} GPUs"
echo "Each GPU will collect ${NUM_SAMPLES_PER_GPU}"

# Function to create a new pane and run the script
create_pane() {
    local device=$1
    tmux split-window -t "$SESSION_NAME"
    tmux send-keys -t "$SESSION_NAME" "conda activate ${CONDA_ENV} && $PYTHON_COMMAND --device $device -o ${OUTPUT_DIR}/data_${device}.txt --num-samples ${NUM_SAMPLES_PER_GPU}" C-m
}

# Create panes for each device number
for device in "${DEVICES[@]}"; do
    create_pane ${device}
done

# Remove the first pane (empty one)
tmux kill-pane -t "$SESSION_NAME.0"

# Arrange panes in a tiled layout
tmux select-layout -t "$SESSION_NAME" tiled

# Attach to the tmux session
tmux attach-session -t "$SESSION_NAME"
