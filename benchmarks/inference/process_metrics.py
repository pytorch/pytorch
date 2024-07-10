"""
This file will take the csv outputs from server.py, calculate the mean and
variance of the warmup_latency, average_latency, throughput and gpu_util
and write these to the corresponding `results/output_{batch_size}_{compile}.md`
file, appending to the file if it exists or creatng a new one otherwise.
"""

import argparse
import os

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse output files")
    parser.add_argument("--csv", type=str, help="Path to csv file")
    parser.add_argument("--name", type=str, help="Name of experiment")
    args = parser.parse_args()

    input_csv = "./results/" + args.csv
    df = pd.read_csv(input_csv)

    batch_size = int(os.path.basename(args.csv).split("_")[1])
    compile = os.path.basename(args.csv).split("_")[-1].split(".")[0]

    # Calculate mean and standard deviation for a subset of metrics
    metrics = ["warmup_latency", "average_latency", "throughput", "gpu_util"]
    means = dict()
    stds = dict()

    for metric in metrics:
        means[metric] = df[metric].mean()
        stds[metric] = df[metric].std()

    output_md = f"results/output_{batch_size}_{compile}.md"
    write_header = os.path.isfile(output_md) is False

    with open(output_md, "a+") as f:
        if write_header:
            f.write(f"## Batch Size {batch_size} Compile {compile}\n\n")
            f.write(
                "| Experiment | Warmup_latency (s) | Average_latency (s) | Throughput (samples/sec) | GPU Utilization (%) |\n"
            )
            f.write(
                "| ---------- | ------------------ | ------------------- | ------------------------ | ------------------- |\n"
            )

        line = f"| {args.name} |"
        for metric in metrics:
            line += f" {means[metric]:.3f} +/- {stds[metric]:.3f} |"
        f.write(line + "\n")
