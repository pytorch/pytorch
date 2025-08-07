import os
import sys


os.environ["TORCH_LOGS"] = "inductor"

import itertools
import logging
import time
from abc import abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from tabulate import tabulate
from tqdm import tqdm
from triton.testing import do_bench

import torch
from torch._inductor import config as inductor_config
from torch.testing._internal.inductor_utils import _quantize_rowwise


log: logging.Logger = logging.getLogger(__name__)


inductor_config.autotune_num_choices_displayed = None
# force autotuning, but reuse compilation artifacts
inductor_config.autotune_local_cache = False
# uncomment for better debugging
# inductor_config.force_disable_caches = True

USE_FAST_ACCUM = True

UNITS = {
    "name": "",
    "forward_time": " (us)",
    "teraflops": " (TFLOPS)",
    "compilation_time": " (s)",
}
PERF_OVER_ATEN_STR: str = "perf_over_aten (%)"

OP_NAMES = [
    "mm",
    # "addmm",
    # "bmm",
    # "_scaled_mm",
]

SHAPES = [
    # M, N, K
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (8192, 8192, 8192),
]

BATCH_SIZES = [
    # For non-bmm testing, still need to specify something
    8,
]

DTYPES = [
    torch.float16,
    torch.bfloat16,
    # torch.float8_e4m3fn,
]

# triton knobs
ENABLE_PERSISTENT_TMA_MATMULS = [
    False,
    True,
]

# cutlass knobs
CUTLASS_INSTANTIATION_LEVELS = [
    "0",
    # "1111",
    # "2222",
    "3332",
    # "9992",
]


def benchmark_torch_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    return do_bench(lambda: func(*args, **kwargs), warmup=100, rep=10000) * 1e3


@dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    max_autotune: bool = True
    coordinate_descent_tuning: bool = True
    max_autotune_gemm_backends: str = "ATEN"

    @abstractmethod
    def name(self) -> str:
        pass

    def to_options(self) -> dict[str, Any]:
        return {
            "max_autotune": self.max_autotune,
            "coordinate_descent_tuning": self.coordinate_descent_tuning,
            "max_autotune_gemm_backends": self.max_autotune_gemm_backends,
        }


@dataclass(frozen=True, kw_only=True)
class AtenExperimentConfig(ExperimentConfig):
    def name(self) -> str:
        return "aten"


@dataclass(frozen=True, kw_only=True)
class CutlassExperimentConfig(ExperimentConfig):
    cutlass_instantiation_level: str

    def name(self) -> str:
        level_name = (
            self.cutlass_instantiation_level
            if self.cutlass_instantiation_level != "0"
            else "default"
        )
        return f"cutlass_lvl_{level_name}"

    def to_options(self) -> dict[str, Any]:
        return {
            **super().to_options(),
            "cuda.cutlass_instantiation_level": self.cutlass_instantiation_level,
        }


@dataclass(frozen=True, kw_only=True)
class TritonExperimentConfig(ExperimentConfig):
    enable_persistent_tma_matmul: bool = False

    def name(self) -> str:
        if self.enable_persistent_tma_matmul:
            return "triton_persistent_tma"
        else:
            return "triton"

    def to_options(self) -> dict[str, Any]:
        return {
            **super().to_options(),
            "triton.enable_persistent_tma_matmul": self.enable_persistent_tma_matmul,
        }


@dataclass(frozen=True, kw_only=True)
class ExperimentGroupConfig:
    op_name: str
    shape: tuple[int, int, int]
    dtype: torch.dtype
    batch_size: int

    experiments: list[ExperimentConfig] = field(default_factory=list)

    def name(self) -> str:
        M, N, K = self.shape
        B = self.batch_size
        sizes = (
            f"(BS: {B}, {M}x{K}, {K}x{N})"
            if self.op_name == "bmm"
            else f"({M}x{K}, {K}x{N})"
        )
        return f"{self.op_name} {sizes} {self.dtype}"


@dataclass(frozen=True, kw_only=True)
class ExperimentResults:
    name: str
    forward_time: float
    teraflops: float
    compilation_time: float

    def asdict(self):
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class ExperimentGroup:
    config: ExperimentGroupConfig
    results: list[ExperimentResults] = field(default_factory=list)


def get_inputs(
    config: ExperimentGroupConfig,
) -> tuple[torch.Tensor, ...]:
    op_name = config.op_name
    M, N, K = config.shape
    batch_size = config.batch_size
    dtype = config.dtype
    device = torch.device("cuda")

    if op_name == "mm":
        A = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(N, K, dtype=dtype, device=device).t()
        return A, B
    elif op_name == "addmm":
        A = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(N, K, dtype=dtype, device=device).t()
        C = torch.randn(N, dtype=dtype, device=device)
        return C, A, B
    elif op_name == "bmm":
        A = torch.randn(batch_size, M, K, dtype=dtype, device=device)
        B = torch.randn(batch_size, N, K, dtype=dtype, device=device).permute(0, 2, 1)
        return A, B
    elif op_name == "_scaled_mm":
        # For _scaled_mm, we only support fp8e4m3 with rowwise scaling
        if dtype != torch.float8_e4m3fn:
            raise ValueError(f"_scaled_mm only supports fp8e4m3, got {dtype}")

        # Create input tensors in bfloat16 first, then quantize to fp8
        input_dtype = torch.bfloat16
        x = torch.randn(M, K, dtype=input_dtype, device=device)
        w = torch.randn(N, K, dtype=input_dtype, device=device)

        # Quantize using rowwise scaling
        w_fp8, w_inverse_scale = _quantize_rowwise(w, dtype)
        w_t_fp8 = w_fp8.t()
        w_inverse_scale = w_inverse_scale.t()  # scale_b should be (1, N)

        x_fp8, x_inverse_scale = _quantize_rowwise(x, dtype)

        # Return inputs for _scaled_mm: (input, weight_t, scale_a, scale_b, bias, out, out_dtype, use_fast_accum)
        return (
            x_fp8,
            w_t_fp8,
            x_inverse_scale,
            w_inverse_scale,
            None,
            None,
            torch.bfloat16,
            USE_FAST_ACCUM,
        )
    else:
        raise ValueError(f"Unknown op {op_name}")


def run_single_experiment_group(
    group_config: ExperimentGroupConfig,
) -> list[ExperimentResults]:
    inputs = get_inputs(group_config)
    op = getattr(torch, group_config.op_name)

    results = []

    for config in group_config.experiments:
        torch._dynamo.reset()
        torch._inductor.utils.clear_caches()
        compiled_op = torch.compile(
            op,
            options=config.to_options(),
        )

        start_time = time.perf_counter()
        try:
            _ = compiled_op(*inputs)
        except Exception as e:
            import traceback

            log.warning(
                f"Benchmark config {config.name()} failed: {e}, "  # noqa: G004
                f"traceback: {traceback.format_exc()}"
            )
            results.append(
                ExperimentResults(
                    name=config.name(),
                    forward_time=float("inf"),
                    teraflops=0.0,
                    compilation_time=float("inf"),
                )
            )
            continue
        compilation_time = time.perf_counter() - start_time

        forward_time = benchmark_torch_function_in_microseconds(
            compiled_op,
            *inputs,
        )

        flops = calculate_flops(
            group_config.op_name,
            group_config.shape,
            group_config.batch_size,
        )
        teraflops = flops / (forward_time * 1e-6) / 1e12

        results.append(
            ExperimentResults(
                name=config.name(),
                forward_time=forward_time,
                teraflops=teraflops,
                compilation_time=compilation_time,
            )
        )

    return results


def generate_experiment_groups(
    op_names: list[str],
    shapes: list[tuple[int, int, int]],
    dtypes: list[torch.dtype],
    enable_persistent_tma_matmuls: list[bool],
    cutlass_instantiation_levels: list[str],
    batch_sizes: list[int],
) -> list[ExperimentGroupConfig]:
    groups = []
    for (
        op_name,
        shape,
        dtype,
        batch_size,
    ) in itertools.product(op_names, shapes, dtypes, batch_sizes):
        group = ExperimentGroupConfig(
            op_name=op_name,
            shape=shape,
            dtype=dtype,
            batch_size=batch_size,
        )
        experiments = generate_experiment_configs(
            enable_persistent_tma_matmuls, cutlass_instantiation_levels
        )
        group.experiments.extend(experiments)
        groups.append(group)

    return groups


def generate_experiment_configs(
    enable_persistent_tma_matmuls: list[bool], cutlass_instantiation_levels: list[str]
) -> list[ExperimentConfig]:
    configs = []

    # add aten configs
    configs.append(
        AtenExperimentConfig(
            max_autotune_gemm_backends="ATEN",
        )
    )

    # add triton configs
    for enable_persistent_tma_matmul in enable_persistent_tma_matmuls:
        configs.append(
            TritonExperimentConfig(
                max_autotune_gemm_backends="TRITON",
                enable_persistent_tma_matmul=enable_persistent_tma_matmul,
            )
        )

    # add cutlass configs
    for cutlass_instantiation_level in cutlass_instantiation_levels:
        configs.append(
            CutlassExperimentConfig(
                max_autotune_gemm_backends="CUTLASS",
                cutlass_instantiation_level=cutlass_instantiation_level,
            )
        )

    return configs


def calculate_table_data(results: list[ExperimentResults]) -> dict:
    table_data = defaultdict(list)
    aten_perf: Optional[float] = None

    for experiment_result in results:
        for key, value in experiment_result.asdict().items():
            assert key in UNITS, f"Unknown key {key}"
            table_data[key + UNITS[key]].append(value)

        if experiment_result.name == "aten":
            aten_perf = experiment_result.forward_time
            table_data[PERF_OVER_ATEN_STR].append("NA")
        elif aten_perf is not None:
            perf_over_aten = (
                (experiment_result.forward_time - aten_perf) / aten_perf * 100
            )
            table_data[PERF_OVER_ATEN_STR].append(perf_over_aten)
        else:
            # fallback in case aten is not in experiment group
            table_data[PERF_OVER_ATEN_STR].append("NA")

    return table_data


def calculate_flops(op_name: str, shape: tuple[int, int, int], batch_size: int) -> int:
    """
    Calculate the number of floating point operations based on operation type and shape.
    """
    M, N, K = shape

    if op_name == "bmm":
        return 2 * batch_size * M * N * K
    elif op_name == "addmm":
        return 2 * M * N * K + M * N
    elif op_name == "_scaled_mm":
        return 2 * M * N * K
    else:
        return 2 * M * N * K


def get_printable_results(experiment_groups: list[ExperimentGroup]) -> list[str]:
    edge_over_aten = defaultdict(list)
    output = []

    for experiment_group in experiment_groups:
        group_config_name = experiment_group.config.name()
        output.append(f"\nExperiment group: {group_config_name}")

        table_data = calculate_table_data(experiment_group.results)
        for name, edge in zip(table_data["name"], table_data[PERF_OVER_ATEN_STR]):
            edge_over_aten[name].append(edge)
        output.append(
            tabulate(table_data, headers="keys", tablefmt="pretty", floatfmt=".3f")
        )

    if "aten" in edge_over_aten:
        output.append("\nAverage edge over aten (max(-edge, 0), higher is better):")
        for name in edge_over_aten:
            if name != "aten":
                values = [
                    max(-v, 0.0)
                    for v in edge_over_aten[name]
                    if v != float("inf") and v != "NA"
                ]
                valid_count = len(values)
                average_edge = sum(values) / valid_count if values else "No valid data"
                output.append(
                    f"{name}: {average_edge} (from {valid_count} valid values)"
                )
        output.append("\n")

    return "\n".join(output)


def main():
    seed = 123
    torch.manual_seed(seed)
    results = []
    log.info("Starting benchmarking...")
    configs = list(
        generate_experiment_groups(
            OP_NAMES,
            SHAPES,
            DTYPES,
            ENABLE_PERSISTENT_TMA_MATMULS,
            CUTLASS_INSTANTIATION_LEVELS,
            BATCH_SIZES,
        )
    )
    for i, group_config in enumerate(tqdm(configs)):
        group_results = run_single_experiment_group(group_config)  # noqa: G004
        results.append(
            ExperimentGroup(config=group_config, results=group_results),
        )
        sys.stderr.write(
            f"\nINTERMEDIATE results: {i + 1}/{len(configs)} \n"
            + get_printable_results(results)
        )
    print("\nFINAL results...")
    print(get_printable_results(results))


if __name__ == "__main__":
    main()
