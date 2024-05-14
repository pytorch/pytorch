#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 {experiment_name}"
  exit 1
fi

experiment_name="$1"
benchmark_script="server.py"
checkpoint_file="resnet18-f37072fd.pth"
downloaded_checkpoint=false
num_iters=10

batch_size_values=(1 32 64 128 256)
compile_values=(true false)

if [ -f $checkpoint_file ]; then
  echo "Checkpoint exists."
else
  downloaded_checkpoint=true
  echo "Downloading checkpoint..."
  wget https://download.pytorch.org/models/resnet18-f37072fd.pth
fi

for batch_size in "${batch_size_values[@]}"; do
  for compile in "${compile_values[@]}"; do
    echo "Running benchmark for batch size ${batch_size} and compile=${compile}..."
    output_file="output_${batch_size}_${compile}.csv"
    if [ -e "./results/$output_file" ]; then
      rm "./results/$output_file"
    fi
    for i in $(seq 1 $num_iters); do
      if [ "$compile" = true ]; then
        python -W ignore "$benchmark_script" --batch_size "$batch_size" --output_file "$output_file" --compile
      else
        python -W ignore "$benchmark_script" --batch_size "$batch_size" --output_file "$output_file" --no-compile
      fi
    done
    python process_metrics.py --csv "$output_file" --name "$experiment_name"
    rm "./results/$output_file"
  done
done

if [ "$downloaded_checkpoint" = true ]; then
  echo "Cleaning up checkpoint..."
  rm "$checkpoint_file"
else
  echo "No cleanup needed"
fi
