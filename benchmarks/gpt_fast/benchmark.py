import argparse
import csv
import dataclasses
import os

from generate import run_llama2_7b_bf16, run_llama2_7b_int8, run_mixtral_8x7b_int8
from triton.testing import do_bench

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

WARMUP_ITER = 5

A100_80G_BF16_TFLOPS = 312


@dataclasses.dataclass
class Experiment:
    name: str
    metric: str
    target: float
    actual: float


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dtype):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden_dim, dtype=dtype),
                nn.LayerNorm(hidden_dim, dtype=dtype),
                nn.Linear(hidden_dim, output_dim, dtype=dtype),
                nn.LayerNorm(output_dim, dtype=dtype),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def run_mlp_layer_norm_gelu():
    dtype_flops_utilization_map = {
        torch.bfloat16: "0.71",
    }
    input_shapes = [1024, 4096, 8192, 16384]
    intermediate_size = 14336
    results = []
    for dtype, expected_flops_utilization in dtype_flops_utilization_map.items():
        flops_utilization = 0
        for D in input_shapes:
            mod = SimpleMLP(
                input_dim=D, hidden_dim=intermediate_size, output_dim=D, dtype=dtype
            ).to("cuda")

            x = torch.randn(D, device="cuda", dtype=torch.bfloat16)

            with FlopCounterMode(display=False) as mode:
                mod(x)

            flops = mode.get_total_flops()

            compiled_mod = torch.compile(mod, dynamic=False)

            for _ in range(WARMUP_ITER):
                compiled_mod(x)

            us_per_iter = do_bench(lambda: compiled_mod(x)) * 1000
            flops_utilization += us_per_iter * flops / 1e9 / A100_80G_BF16_TFLOPS

        flops_utilization = flops_utilization / len(input_shapes)
        dtype_str = str(dtype).replace("torch.", "")
        results.append(
            Experiment(
                f"mlp_layer_norm_gelu_{dtype_str}",
                "flops_utilization",
                expected_flops_utilization,
                f"{flops_utilization:.02f}",
            )
        )
    return results


def run_layer_norm():
    dtype_memory_bandwidth_map = {
        torch.bfloat16: "1017",
    }
    input_shapes = [1024, 4096, 8192, 16384]
    BS = 4096
    results = []
    for dtype, expected_memory_bandwidth in dtype_memory_bandwidth_map.items():
        memory_bandwidth = 0
        for D in input_shapes:
            mod = nn.LayerNorm(D).to("cuda")

            x = torch.randn(BS, D, device="cuda", dtype=dtype)

            compiled_mod = torch.compile(mod, dynamic=False)

            for _ in range(WARMUP_ITER):
                compiled_mod(x)

            us_per_iter = do_bench(lambda: compiled_mod(x)) * 1000
            memory_bandwidth += (1e6 / us_per_iter) * 2 * BS * D * dtype.itemsize / 1e9

        memory_bandwidth = memory_bandwidth / len(input_shapes)
        dtype_str = str(dtype).replace("torch.", "")
        results.append(
            Experiment(
                f"layer_norm_{dtype_str}",
                "memory_bandwidth(GB/s)",
                expected_memory_bandwidth,
                f"{memory_bandwidth:.02f}",
            )
        )
    return results


@torch._inductor.config.patch(coordinate_descent_tuning=True)
def run_gather_gemv():
    E = 8
    dtype_memory_bandwidth_map = {
        torch.int8: "1113",
        torch.bfloat16: "1249",
    }
    input_shapes = [1024, 4096, 8192, 16384]
    results = []
    for dtype, expected_memory_bandwidth in dtype_memory_bandwidth_map.items():
        memory_bandwidth = 0
        for D in input_shapes:

            def gather_gemv(W, score_idxs, x):
                return W[score_idxs].to(x.dtype) @ x

            W = torch.randn(E, D, D, device="cuda").to(dtype=dtype)
            x = torch.randn(D, device="cuda", dtype=torch.bfloat16)
            score_idxs = torch.tensor([3, 5], device="cuda")

            compiled_fn = torch.compile(gather_gemv, dynamic=False)

            for _ in range(WARMUP_ITER):
                compiled_fn(W, score_idxs, x)

            us_per_iter = do_bench(lambda: compiled_fn(W, score_idxs, x)) * 1000
            memory_bandwidth += (1e6 / us_per_iter) * 2 * D * D * dtype.itemsize / 1e9

        memory_bandwidth = memory_bandwidth / len(input_shapes)
        dtype_str = str(dtype).replace("torch.", "")
        results.append(
            Experiment(
                f"gather_gemv_{dtype_str}",
                "memory_bandwidth(GB/s)",
                expected_memory_bandwidth,
                f"{memory_bandwidth:.02f}",
            )
        )
    return results


@torch._inductor.config.patch(coordinate_descent_tuning=True)
def run_gemv():
    dtype_memory_bandwidth_map = {
        torch.int8: "990",
        torch.bfloat16: "1137",
    }
    input_shapes = [1024, 4096, 8192, 16384]
    results = []
    for dtype, expected_memory_bandwidth in dtype_memory_bandwidth_map.items():
        memory_bandwidth = 0
        for D in input_shapes:

            def gemv(W, x):
                return W.to(x.dtype) @ x

            W = torch.randn(D, D, device="cuda").to(dtype=dtype)
            x = torch.randn(D, device="cuda", dtype=torch.bfloat16)

            compiled_fn = torch.compile(gemv, dynamic=False)

            for _ in range(WARMUP_ITER):
                compiled_fn(W, x)

            us_per_iter = do_bench(lambda: compiled_fn(W, x)) * 1000
            memory_bandwidth += (1e6 / us_per_iter) * D * D * dtype.itemsize / 1e9

        memory_bandwidth = memory_bandwidth / len(input_shapes)
        dtype_str = str(dtype).replace("torch.", "")
        results.append(
            Experiment(
                f"gemv_{dtype_str}",
                "memory_bandwidth(GB/s)",
                expected_memory_bandwidth,
                f"{memory_bandwidth:.02f}",
            )
        )
    return results


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
    run_mlp_layer_norm_gelu,
    run_layer_norm,
    run_gather_gemv,
    run_gemv,
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
