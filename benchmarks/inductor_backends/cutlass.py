import os


os.environ["TORCH_LOGS"] = "inductor"

import itertools
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


inductor_config.autotune_num_choices_displayed = None
# force autotuning, but reuse compilation artifacts
inductor_config.autotune_local_cache = False
# uncomment for better debugging
# inductor_config.force_disable_caches = True


UNITS = {
    "name": "",
    "forward_time": " (us)",
    "compilation_time": " (s)",
}

OP_NAMES = ["mm"]

SHAPES = [
    # M, N, K
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (8192, 8192, 8192),
]

DTYPES = [
    torch.float16,
    torch.bfloat16,
]

# triton knobs
ENABLE_PERSISTENT_TMA_MATMULS = [
    False,
    True,
]

# cutlass knobs
CUTLASS_INSTANTIATION_LEVELS = [
    "0",
    "1111",
    "2222",
    # not ready yet
    # "3333",
]


def benchmark_torch_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    return do_bench(lambda: func(*args, **kwargs)) * 1e3


@dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    autotune_fallback_to_aten: bool = False
    max_autotune: bool = True
    coordinate_descent_tuning: bool = True
    max_autotune_gemm_backends: str = "ATEN"

    @abstractmethod
    def name(self) -> str:
        pass

    def to_options(self) -> dict[str, Any]:
        return {
            "autotune_fallback_to_aten": self.autotune_fallback_to_aten,
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

    experiments: list[ExperimentConfig] = field(default_factory=list)

    def name(self) -> str:
        M, N, K = self.shape
        sizes = f"({M}x{K}, {K}x{N})"
        return f"{self.op_name} {sizes} {self.dtype}"


@dataclass(frozen=True, kw_only=True)
class ExperimentResults:
    name: str
    forward_time: float
    compilation_time: float

    def asdict(self):
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class ExperimentGroup:
    config: ExperimentGroupConfig
    results: list[ExperimentResults] = field(default_factory=list)


def get_inputs(
    config: ExperimentGroupConfig,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    op_name = config.op_name
    M, N, K = config.shape
    dtype = config.dtype
    device = torch.device("cuda")

    if op_name == "mm":
        A = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(N, K, dtype=dtype, device=device).t()
        C = None
        return A, B, C
    else:
        raise ValueError(f"Unknown op {op_name}")


def run_single_experiment_group(
    group_config: ExperimentGroupConfig,
) -> list[ExperimentResults]:
    A, B, C = get_inputs(group_config)
    op = getattr(torch, group_config.op_name)

    results = []

    for config in group_config.experiments:
        torch._dynamo.reset()
        torch._inductor.utils.clear_inductor_caches()
        compiled_op = torch.compile(op, fullgraph=True, options=config.to_options())

        start_time = time.perf_counter()
        try:
            _ = compiled_op(A, B)
        except Exception as e:
            print(f"Benchmark config {config.name()} failed: {e}")
            results.append(
                ExperimentResults(
                    name=config.name(),
                    forward_time=float("inf"),
                    compilation_time=float("inf"),
                )
            )
            continue
        compilation_time = time.perf_counter() - start_time

        forward_time = benchmark_torch_function_in_microseconds(
            compiled_op,
            A,
            B,
        )

        results.append(
            ExperimentResults(
                name=config.name(),
                forward_time=forward_time,
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
) -> list[ExperimentGroupConfig]:
    groups = []
    for op_name, shape, dtype in itertools.product(op_names, shapes, dtypes):
        group = ExperimentGroupConfig(
            op_name=op_name,
            shape=shape,
            dtype=dtype,
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


def tabulate_group_results(results: list[ExperimentResults]):
    table_data = defaultdict(list)
    aten_perf: Optional[float] = None
    perf_over_aten_str: str = "perf_over_aten (%)"

    for experiment_result in results:
        for key, value in experiment_result.asdict().items():
            assert key in UNITS, f"Unknown key {key}"
            table_data[key + UNITS[key]].append(value)

        if experiment_result.name == "aten":
            aten_perf = experiment_result.forward_time
            table_data[perf_over_aten_str].append("NA")
        elif aten_perf is not None:
            perf_over_aten = (
                (experiment_result.forward_time - aten_perf) / aten_perf * 100
            )
            table_data[perf_over_aten_str].append(perf_over_aten)
        else:
            # fallback in case aten is not in experiment group
            table_data[perf_over_aten_str].append("NA")

    return tabulate(table_data, headers="keys", tablefmt="pretty", floatfmt=".3f")


def print_results(experiment_groups: list[ExperimentGroup]):
    for experiment_group in experiment_groups:
        group_config_name = experiment_group.config.name()
        print(f"\nExperiment group: {group_config_name}")
        print(tabulate_group_results(experiment_group.results))


def main():
    seed = 123
    torch.manual_seed(seed)
    results = []
    for group_config in tqdm(
        generate_experiment_groups(
            OP_NAMES,
            SHAPES,
            DTYPES,
            ENABLE_PERSISTENT_TMA_MATMULS,
            CUTLASS_INSTANTIATION_LEVELS,
        )
    ):
        group_results = run_single_experiment_group(group_config)
        results.append(
            ExperimentGroup(config=group_config, results=group_results),
        )

    print_results(results)


if __name__ == "__main__":
    main()
