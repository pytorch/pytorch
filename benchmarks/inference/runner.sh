#!/bin/bash

batch_size_values=(1 32 64 128 256)
compile_values=(True False)

benchmark_script="server.py"
checkpoint_file="resnet18-f37072fd.pth"
downloaded_checkpoint=false

if [ -f $checkpoint_file ]; then
  echo "Checkpoint exists."
else
  downloaded_checkpoint=true
  echo "Downloading checkpoint..."
  wget https://download.pytorch.org/models/resnet18-f37072fd.pth
  echo "============================================================================="
fi

echo "Starting benchmark..."
for batch_size in "${batch_size_values[@]}"; do
  for compile in "${compile_values[@]}"; do
      echo "Running benchmark with batch_size=$batch_size, compile=$compile..."
      python -W ignore "$benchmark_script" --batch_size "$batch_size" --compile "$compile"
      echo "============================================================================="
  done
done

if [ "$downloaded_checkpoint" = true ]; then
  echo "Cleaning up checkpoint..."
  rm "$checkpoint_file"
else
  echo "No cleanup needed"
fi
