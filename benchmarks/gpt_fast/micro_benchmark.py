import argparse
import dataclasses
import time

import torch
import torch.nn as nn


@dataclasses.dataclass
class Experiment:
    name: str
    metric: str
    target: float
    actual: float


DEFAULT_OUTPUT_FILE = "micro_benchmark.csv"


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


def run_multi_layernorm():
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


all_experiments = {
    run_multi_layernorm,
}


def main(output_file=DEFAULT_OUTPUT_FILE):
    results = []

    for func in all_experiments:
        lst = func()
        for x in lst:
            results.append(dataclasses.astuple(x))

    headers = [field.name for field in dataclasses.fields(Experiment)]

    from benchmark import output_csv

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
