import argparse
import csv
import dataclasses
import os
import time

from generate import run_llama2_7b_bf16, run_llama2_7b_int8, run_mixtral_8x7b_int8

import torch
import torch.nn as nn


@dataclasses.dataclass
class Experiment:
    name: str
    metric: str
    target: float
    actual: float


def do_inference(mod, x, num_samples: int = 5):
    total_time = 0
    start = -1

    for i in range(start, num_samples):
        torch.cuda.synchronize("cuda")

        t0 = time.perf_counter()
        mod(x)

        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue

        torch.cuda.synchronize("cuda")
        total_time += time.perf_counter() - t0

    total_time = total_time / num_samples

    return total_time


def run_multi_layer_norm():
    class MultiLayerNorm(nn.Module):
        def __init__(self, num_layers, normalized_shape, eps=1e-5, bias=True):
            super().__init__()
            self.num_layers = num_layers
            self.norm_layers = nn.ModuleList(
                [
                    nn.LayerNorm(normalized_shape, eps=eps, bias=bias)
                    for _ in range(num_layers)
                ]
            )

        def forward(self, x):
            for layer_norm in self.norm_layers:
                x = layer_norm(x)
            return x

    mod = MultiLayerNorm(num_layers=8, normalized_shape=4096).to("cuda")
    mod = torch.compile(mod)
    input = torch.randn([512, 1024, 4096], dtype=torch.bfloat16, device="cuda")
    inference_time = do_inference(mod, input)

    memory_bandwidth = input.numel() * input.dtype.itemsize / inference_time / 1e9

    return [
        Experiment(
            "multi_layer_norm", "memory_bandwidth(GB/s)", 92, f"{memory_bandwidth:.02f}"
        )
    ]


def output_csv(output_file, headers, row):
    if os.path.exists(output_file):
        with open(output_file) as fd:
            lines = list(csv.reader(fd)) or [[]]
            if headers and len(headers) > len(lines[0]):
                # if prior results failed the header might not be filled in yet
                lines[0] = headers
            else:
                headers = lines[0]
    else:
        lines = [headers]

    if output_file != DEFAULT_OUTPUT_FILE:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    lines.append([(f"{x:.6f}" if isinstance(x, float) else x) for x in row])
    with open(output_file, "w") as fd:
        writer = csv.writer(fd, lineterminator="\n")
        for line in lines:
            writer.writerow(list(line) + ["0"] * (len(headers) - len(line)))


DEFAULT_OUTPUT_FILE = "gpt_fast_benchmark.csv"

all_experiments = {
    # A list of GPT models: LlaMa, Mixtral, etc.
    run_llama2_7b_bf16,
    run_llama2_7b_int8,
    run_mixtral_8x7b_int8,
    # A list of micro-benchmarks.
    run_multi_layer_norm,
}


def main(output_file=DEFAULT_OUTPUT_FILE):
    results = []

    for func in all_experiments:
        lst = func()
        for x in lst:
            results.append(dataclasses.astuple(x))

    headers = [field.name for field in dataclasses.fields(Experiment)]

    for row in results:
        output_csv(output_file, headers, row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="Set the output CSV file to save the benchmark results",
    )
    args = parser.parse_args()

    main(output_file=args.output)
