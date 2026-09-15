from __future__ import annotations

import argparse
import os
from itertools import cycle
from typing import TYPE_CHECKING


# Native reductions register during import, so enable the rollout first.
os.environ["PYTORCH_SUM_INNER_TREE"] = "1"

from triton.testing import do_bench, do_bench_cudagraph

import torch
import torch._inductor.config as inductor_config
from torch._inductor.utils import fresh_inductor_cache


if TYPE_CHECKING:
    from collections.abc import Callable


PROVIDERS = ["eager_aten", "eager_inner_tree", "inductor_default"]
M_VALUES = [512, 1024, 2048, 4096, 8192, 16384]
N_VALUES = [2**exponent for exponent in range(9, 18)]
MEMORY_BUDGET = 40 * (1 << 30)
CUDA_GRAPH_WORKING_SET_BYTES = 256 * (1 << 20)
COLORS = {
    "eager_aten": "C0",
    "inductor_default": "C1",
    "eager_inner_tree": "C2",
}
DTYPE = torch.float32


def _gbps(m: int, n: int, milliseconds: float) -> float:
    return m * n * DTYPE.itemsize / (milliseconds * 1e-3) / 1e9


def _sum_rows(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.sum(1)


def _circular_runner(
    function: Callable[[torch.Tensor], torch.Tensor],
    tensors: tuple[torch.Tensor, ...],
) -> Callable[[], torch.Tensor]:
    tensor_iterator = cycle(tensors)
    return lambda: function(next(tensor_iterator))


def measure(m: int, n: int, provider: str, cudagraph: bool) -> float:
    tensor_bytes = m * n * DTYPE.itemsize
    if tensor_bytes > MEMORY_BUDGET:
        return float("nan")

    torch.manual_seed(0)
    input_count = (
        max(
            1,
            (CUDA_GRAPH_WORKING_SET_BYTES + tensor_bytes - 1) // tensor_bytes,
        )
        if cudagraph
        else 1
    )
    input_buffer = torch.randn(input_count, m, n, device="cuda", dtype=DTYPE)
    tensors = input_buffer.unbind()
    runner = _circular_runner(_sum_rows, tensors)
    benchmark = do_bench_cudagraph if cudagraph else do_bench
    try:
        if provider.startswith("eager"):
            if provider == "eager_inner_tree":
                runner()
                milliseconds = benchmark(runner)
            else:
                with torch.backends.python_native.cutedsl.disabled():
                    runner()
                    milliseconds = benchmark(runner)
        else:
            torch._dynamo.reset()
            with fresh_inductor_cache():
                compiled = torch.compile(_sum_rows)
                runner = _circular_runner(compiled, tensors)
                runner()
                milliseconds = benchmark(runner)
        return _gbps(m, n, milliseconds)
    finally:
        del runner
        del tensors
        del input_buffer
        torch.cuda.empty_cache()


def _assert_inner_tree_route() -> None:
    tensor = torch.randn(1024, 300, device="cuda", dtype=DTYPE)
    try:
        inner_tree = torch.sum(tensor, 1)
        with torch.backends.python_native.cutedsl.disabled():
            aten = torch.sum(tensor, 1)
        assert not torch.equal(inner_tree, aten), (  # noqa: S101
            "eager_inner_tree did not route to CuTeDSL; refusing to label ATen "
            "results as inner-tree results"
        )
    finally:
        del tensor
        torch.cuda.empty_cache()


def _plot(data: dict[str, dict[int, list[float]]], cudagraph: bool) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 2, figsize=(12, 13))
    flat_axes = axes.flatten()
    for index, m in enumerate(M_VALUES):
        axis = flat_axes[index]
        for provider in PROVIDERS:
            axis.plot(
                N_VALUES,
                data[provider][m],
                marker="o",
                label=provider,
                color=COLORS[provider],
            )
        axis.set_xscale("log", base=2)
        axis.set_title(f"M={m}")
        axis.set_xlabel("N")
        axis.set_ylabel("GB/s")
        axis.legend(fontsize=8)
        axis.grid(True, alpha=0.3)
    mode = "CUDA Graph" if cudagraph else "standard"
    figure.suptitle(f"Inner-tree SUM reduction {mode} GB/s vs N", y=1.0)
    figure.tight_layout()
    suffix = "_cudagraph" if cudagraph else ""
    output_path = f"strict_numerics_row_reduction{suffix}_gbps_vs_n.png"
    figure.savefig(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cudagraph", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "needs CUDA"  # noqa: S101
    _assert_inner_tree_route()
    print("providers:", PROVIDERS)
    measurement = (
        "CUDA Graph replay with a "
        f"{CUDA_GRAPH_WORKING_SET_BYTES // (1 << 20)} MiB circular input buffer"
        if args.cudagraph
        else "standard"
    )
    print("measurement:", measurement)

    data = {provider: {m: [] for m in M_VALUES} for provider in PROVIDERS}
    with inductor_config.patch({"force_disable_caches": True}):
        for m in M_VALUES:
            for n in N_VALUES:
                for provider in PROVIDERS:
                    data[provider][m].append(measure(m, n, provider, args.cudagraph))
            row = "  ".join(
                f"{provider}={data[provider][m][-1]:.0f}" for provider in PROVIDERS
            )
            print(f"M={m:>6} (N={N_VALUES[-1]}): {row}")

    output_path = _plot(data, args.cudagraph)
    print(f"saved plot to {output_path}")


if __name__ == "__main__":
    main()
