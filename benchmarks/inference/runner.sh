#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <output_filename>.md"
  exit 1
fi

output_markdown="results/$1"
benchmark_script="server.py"
output_file="results/temp_output.txt"
checkpoint_file="resnet18-f37072fd.pth"
downloaded_checkpoint=false

batch_size_values=(1 32 64 128 256)
compile_values=(true false)

if [ -f $checkpoint_file ]; then
  echo "Checkpoint exists."
else
  downloaded_checkpoint=true
  echo "Downloading checkpoint..."
  wget https://download.pytorch.org/models/resnet18-f37072fd.pth
fi

if [ -e "$output_file" ]; then
  rm "$output_file"
fi
touch $output_file

for batch_size in "${batch_size_values[@]}"; do
  for compile in "${compile_values[@]}"; do
      if [ "$compile" = true ]; then
        python -W ignore "$benchmark_script" --batch_size "$batch_size" --compile >> $output_file
      else
        python -W ignore "$benchmark_script" --batch_size "$batch_size" --no-compile >> $output_file
      fi
  done
done

echo "| bs, compile | torch.load() / s | warmup latency / s | avg latency / s | max latency / s | min latency / s | throughput samples/s | GPU util % |" > $output_markdown
echo "| ----------- | ---------------- | ------------------ | --------------- | --------------- | --------------- | -------------------- | ---------- |" >> $output_markdown

while IFS= read -r line; do
    batch_size=$(echo "$line" | jq -r '.batch_size')
    compile=$(echo "$line" | jq -r '.compile')
    torch_load=$(echo "$line" | jq -r '.torch_load_time' | awk '{printf "%.5f", $0}')
    warmup_latency=$(echo "$line" | jq -r '.warmup_latency' | awk '{printf "%.5f", $0}')
    avg_latency=$(echo "$line" | jq -r '.average_latency' | awk '{printf "%.5f", $0}')
    max_latency=$(echo "$line" | jq -r '.max_latency' | awk '{printf "%.5f", $0}')
    min_latency=$(echo "$line" | jq -r '.min_latency' | awk '{printf "%.5f", $0}')
    throughput=$(echo "$line" | jq -r '.throughput' | awk '{printf "%.5f", $0}')
    gpu_util=$(echo "$line" | jq -r '.GPU_utilization' | awk '{printf "%.5f", $0}')
    echo "| $batch_size, $compile | $torch_load | $warmup_latency | $avg_latency | $max_latency | $min_latency | $throughput | $gpu_util |"
done < $output_file >> $output_markdown

rm "$output_file"

if [ "$downloaded_checkpoint" = true ]; then
  echo "Cleaning up checkpoint..."
  rm "$checkpoint_file"
else
  echo "No cleanup needed"
fi
