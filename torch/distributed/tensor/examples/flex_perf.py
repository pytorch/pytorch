import csv
import gc
import itertools
import random
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import partial, wraps
from typing import Callable, Literal, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F

from jsonargparse import CLI
from tabulate import tabulate
from torch._inductor.runtime.benchmarking import benchmarker
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    create_mask,
    flex_attention,
    noop_mask,
)
from tqdm import tqdm


def cleanup_memory():
    """Aggressively free GPU memory"""
    torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def safe_backend(backend_name=None):
    """Decorator that wraps backend functions with error handling"""

    def decorator(func):
        @wraps(func)
        def wrapper(config, *args, **kwargs):
            try:
                return func(config, *args, **kwargs)
            except torch.OutOfMemoryError:
                print(
                    f"[SKIP] OOM for {backend_name or func.__name__} with shape {config.shape}"
                )
                cleanup_memory()
            except Exception as e:
                print(
                    f"[SKIP] Error for {backend_name or func.__name__} with shape {config.shape}: {e}"
                )

            return ExperimentResults(
                fwd_time=float("nan"),
                bwd_time=float("nan") if config.calculate_bwd_time else None,
            )

        return wrapper

    return decorator


# Type definitions
Backend = Literal["math", "efficient", "cudnn", "fav2", "fav3", "fakv", "og-eager"]
AttentionType = Literal[
    "noop",
    "causal",
    "rel",
    "head_bias",
    "alibi",
    "sliding_window",
    "document_mask",
    "prefix_lm",
    "softcap",
]
DtypeString = Literal["bfloat16", "float16", "float32"]
SpeedupType = Literal["fwd", "bwd"]


torch._dynamo.config.automatic_dynamic_shapes = False
# Needed since changing args to function causes recompiles
torch._dynamo.config.recompile_limit = 1000


def benchmark_torch_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    # warmup
    for _ in range(5):
        func(*args, **kwargs)
    return benchmarker.benchmark_gpu(lambda: func(*args, **kwargs)) * 1e3


@dataclass(frozen=True)
class ExperimentConfig:
    shape: tuple[int, int, int, int, int, int]  # [B, Hq, M, Hkv, N, D]
    attn_type: AttentionType
    dtype: torch.dtype
    calculate_bwd_time: bool
    cal_bandwidth: bool
    backends: list[Backend]

    def __post_init__(self):
        assert (
            len(self.shape) == 6
        ), "Shape must be of length 6"  # [B, Hq, M, Hkv, N, D]

    def asdict(self):
        # Convert the dataclass instance to a dictionary
        d = asdict(self)
        # Remove the 'calculate_bwd_time' and `cal_bandwidth` key
        d.pop("calculate_bwd_time", None)
        d.pop("cal_bandwidth", None)
        d["shape(B,Hq,M,Hkv,N,D)"] = d.pop("shape")
        d.pop("backends", None)
        return d


@dataclass(frozen=True)
class Times:
    eager_time: float
    compiled_time: float


@dataclass(frozen=True)
class ExperimentResults:
    fwd_time: float
    bwd_time: Optional[float]
    sparsity: Optional[float] = None
    fwd_tflops: Optional[float] = None
    fwd_bandwidth: Optional[float] = None
    bwd_tflops: Optional[float] = None
    bwd_bandwidth: Optional[float] = None


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    results: dict[str, ExperimentResults]  # backend -> ExperimentResults

    def asdict(self):
        dict1 = self.config.asdict()
        dict2 = self.results
        return {**dict1, **dict2}


def generate_inputs(
    batch_size: int,
    q_heads: int,
    q_sequence_length: int,
    kv_heads: int,
    kv_sequence_length: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    nested_tensors: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    q_shape = (batch_size, q_sequence_length, q_heads * head_dim)
    kv_shape = (batch_size, kv_sequence_length, kv_heads * head_dim)

    assert q_heads % kv_heads == 0

    make_q = partial(
        torch.rand, q_shape, device=device, dtype=dtype, requires_grad=requires_grad
    )
    make_kv = partial(
        torch.rand, kv_shape, device=device, dtype=dtype, requires_grad=requires_grad
    )

    if nested_tensors:
        query = (
            make_q()
            .view(1, q_sequence_length * batch_size, q_heads, head_dim)
            .transpose(1, 2)
        )
        key = (
            make_kv()
            .view(1, batch_size * kv_sequence_length, kv_heads, head_dim)
            .transpose(1, 2)
        )
        value = (
            make_kv()
            .view(1, batch_size * kv_sequence_length, kv_heads, head_dim)
            .transpose(1, 2)
        )
    else:
        query = (
            make_q()
            .view(batch_size, q_sequence_length, q_heads, head_dim)
            .transpose(1, 2)
        )
        key = (
            make_kv()
            .view(batch_size, kv_sequence_length, kv_heads, head_dim)
            .transpose(1, 2)
        )
        value = (
            make_kv()
            .view(batch_size, kv_sequence_length, kv_heads, head_dim)
            .transpose(1, 2)
        )
    return query, key, value


def generate_jagged_inputs(
    shape: tuple[int, int, int, int, int, int],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, Hq, M, Hkv, N, D = shape

    def offsets_to_lengths(
        offsets: torch.Tensor, device: Union[str, torch.device]
    ) -> torch.Tensor:
        """Converts a list of offsets to a list of lengths. Reverse op of attn_gym.masks.document_mask.length_to_offsets

        Args:
            offsets: A 1D tensor of offsets
            device: The device to place the output tensor on
        """
        lengths = offsets[1:] - offsets[:-1]
        return lengths

    flatten_q = query.transpose(1, 2).flatten(start_dim=0, end_dim=1)
    flatten_k = key.transpose(1, 2).flatten(start_dim=0, end_dim=1)
    flatten_v = value.transpose(1, 2).flatten(start_dim=0, end_dim=1)

    q_list = [
        flatten_q[offsets[i] : offsets[i + 1]].clone().detach().to(query.dtype)
        for i in range(len(offsets) - 1)
    ]
    q = torch.nested.as_nested_tensor(q_list, device=query.device)

    k_list = [
        flatten_k[offsets[i] : offsets[i + 1]].clone().detach().to(key.dtype)
        for i in range(len(offsets) - 1)
    ]
    k = torch.nested.as_nested_tensor(k_list, device=key.device)
    v_list = [
        flatten_v[offsets[i] : offsets[i + 1]].clone().detach().to(value.dtype)
        for i in range(len(offsets) - 1)
    ]
    v = torch.nested.as_nested_tensor(v_list, device=value.device)

    return q, k, v


def query_key_value_clones(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clones the query, key, and value tensors and moves them to the specified dtype."""
    if dtype is None:
        dtype = query.dtype
    query_ref = query.clone().detach().to(dtype).requires_grad_(query.requires_grad)
    key_ref = key.clone().detach().to(dtype).requires_grad_(key.requires_grad)
    value_ref = value.clone().detach().to(dtype).requires_grad_(value.requires_grad)
    return query_ref, key_ref, value_ref


@safe_backend("SDPA")
def run_single_backend_sdpa(
    config: ExperimentConfig,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out_compile: torch.Tensor,
    score_mod: Optional[Callable],
    block_mask: Optional[BlockMask],
    mask_kwargs: dict,
    backend: Backend,
) -> ExperimentResults:
    backend_context = get_backend_context(backend)
    with backend_context:
        eager_sdpa = generate_eager_sdpa(
            config.attn_type, config.shape, config.dtype, block_mask, score_mod
        )

        if config.attn_type == "document_mask":
            q_eager, k_eager, v_eager = generate_jagged_inputs(
                config.shape, query, key, value, mask_kwargs["offsets"]
            )
            q_eager = q_eager.transpose(1, 2).requires_grad_(query.requires_grad)
            k_eager = k_eager.transpose(1, 2).requires_grad_(key.requires_grad)
            v_eager = v_eager.transpose(1, 2).requires_grad_(value.requires_grad)
        else:
            q_eager, k_eager, v_eager = query_key_value_clones(query, key, value)

        if eager_sdpa is None:
            return ExperimentResults(
                fwd_time=float("nan"),
                bwd_time=float("nan") if config.calculate_bwd_time else None,
            )
        forward_eager_time = benchmark_torch_function_in_microseconds(
            eager_sdpa, q_eager, k_eager, v_eager
        )
        bwd_time = None if not config.calculate_bwd_time else float("nan")
        if config.calculate_bwd_time:
            out_eager = eager_sdpa(q_eager, k_eager, v_eager)
            # TODO: debug backward pass for njt
            if not config.attn_type == "document_mask":
                d_out = torch.randn_like(out_eager.transpose(1, 2)).transpose(1, 2)
                backward_eager_time = benchmark_torch_function_in_microseconds(
                    out_eager.backward, d_out, retain_graph=True
                )
                bwd_time = backward_eager_time

        result = ExperimentResults(
            fwd_time=forward_eager_time,
            bwd_time=bwd_time,
            sparsity=0.5 if config.attn_type in ("causal", "document_mask") else 0.0,
        )
        return add_metrics_to_result(config, result)


@safe_backend("FlashAttention")
def run_single_backend_FA(
    config: ExperimentConfig,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out_compile: torch.Tensor,
    score_mod: Optional[Callable],
    block_mask: Optional[BlockMask],
    mask_kwargs: dict,
    backend: Backend,
) -> ExperimentResults:
    assert backend in ["fav2", "fav3", "fakv"]
    # Generate callable for specific backend.
    if backend in ["fav2", "fav3"]:
        FA = generate_FA_callable(
            config.attn_type, config.shape, config.dtype, backend, **mask_kwargs
        )
    elif backend == "fakv":
        FA = generate_FD_callable(config.attn_type, config.shape, config.dtype)

    q_FA, k_FA, v_FA = query_key_value_clones(query, key, value)
    q_FA, k_FA, v_FA = q_FA.transpose(1, 2), k_FA.transpose(1, 2), v_FA.transpose(1, 2)
    if config.attn_type == "document_mask":
        q_FA = q_FA.flatten(start_dim=0, end_dim=1)
        k_FA = k_FA.flatten(start_dim=0, end_dim=1)
        v_FA = v_FA.flatten(start_dim=0, end_dim=1)

    if FA is None:
        return ExperimentResults(
            fwd_time=float("nan"),
            bwd_time=float("nan") if config.calculate_bwd_time else None,
        )

    forward_FA_time = benchmark_torch_function_in_microseconds(
        FA, q=q_FA, k=k_FA, v=v_FA
    )
    backward_FA_time = None
    if config.calculate_bwd_time:
        out_FA = FA(q=q_FA, k=k_FA, v=v_FA)
        d_out = torch.randn_like(out_FA)
        backward_FA_time = benchmark_torch_function_in_microseconds(
            out_FA.backward, d_out, retain_graph=True
        )

    result = ExperimentResults(
        fwd_time=forward_FA_time,
        bwd_time=backward_FA_time if config.calculate_bwd_time else None,
        sparsity=(
            0.5
            if config.attn_type in ("causal", "document_mask", "alibi", "softcap")
            else 0.0
        ),
    )
    return add_metrics_to_result(config, result)


@safe_backend("flex_attention")
def run_flex_attention(
    config: ExperimentConfig,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Optional[Callable],
    block_mask: Optional[BlockMask],
    kernel_options: Optional[dict] = None,
    dynamic: bool = False,
    max_autotune: bool = False,
) -> ExperimentResults:
    if max_autotune:
        compiled_sdpa = torch.compile(
            flex_attention, dynamic=dynamic, mode="max-autotune-no-cudagraphs"
        )
    else:
        compiled_sdpa = torch.compile(flex_attention, dynamic=dynamic)

    out_compile = compiled_sdpa(
        query=query,
        key=key,
        value=value,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=True,
        kernel_options=kernel_options,
    )

    forward_compiled_time = benchmark_torch_function_in_microseconds(
        compiled_sdpa,
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=True,
        kernel_options=kernel_options,
    )

    backward_compile_time = None
    if config.calculate_bwd_time:
        try:
            d_out = torch.randn_like(out_compile)
            backward_compile_time = benchmark_torch_function_in_microseconds(
                out_compile.backward, d_out, retain_graph=True
            )
        except Exception as e:
            print(
                f"[SKIP] Backward pass failed for flex_attention with shape {config.shape}: {e}"
            )
            cleanup_memory()
            backward_compile_time = float("nan")

    sparsity = block_mask.sparsity() / 100.0 if block_mask is not None else 0.0
    sparsity = sparsity if config.attn_type != "document_mask" else 0.5

    result = ExperimentResults(
        fwd_time=forward_compiled_time,
        bwd_time=backward_compile_time,
        sparsity=sparsity,
    )
    return add_metrics_to_result(config, result)


def run_single_experiment(
    config: ExperimentConfig,
    dynamic=False,
    max_autotune=False,
    kernel_options_override: Optional[dict] = None,
) -> dict[str, ExperimentResults]:
    device = torch.device("cuda")
    batch_size, q_heads, q_seq_len, kv_heads, kv_seq_len, head_dim = config.shape
    query, key, value = generate_inputs(
        batch_size,
        q_heads,
        q_seq_len,
        kv_heads,
        kv_seq_len,
        head_dim,
        config.dtype,
        device,
        requires_grad=config.calculate_bwd_time,
        nested_tensors=config.attn_type == "document_mask",
    )
    score_mod = generate_score_mod(config.attn_type, config.shape)
    block_mask, mask_kwargs = generate_block_mask(config.attn_type, config.shape)
    kernel_options = get_kernel_options(config.attn_type, config.shape)

    # Merge in overrides: supports either a flat dict (applies to all)
    # or a mapping keyed by attention type, with optional "global" fallback.
    if kernel_options_override:
        selected_override = kernel_options_override
        # Detect keyed-by-attn-type mapping
        attn_keys = {
            "noop",
            "causal",
            "rel",
            "head_bias",
            "alibi",
            "sliding_window",
            "document_mask",
            "prefix_lm",
            "softcap",
            "global",
        }
        if isinstance(kernel_options_override, dict) and any(
            k in attn_keys for k in kernel_options_override.keys()
        ):
            selected_override = kernel_options_override.get(
                config.attn_type, kernel_options_override.get("global")
            )
        if selected_override is not None:
            if kernel_options is None:
                kernel_options = dict(selected_override)
            else:
                merged = dict(kernel_options)
                merged.update(selected_override)
                kernel_options = merged

    if max_autotune:
        compiled_sdpa = torch.compile(
            flex_attention, dynamic=dynamic, mode="max-autotune-no-cudagraphs"
        )
    else:
        compiled_sdpa = torch.compile(flex_attention, dynamic=dynamic)

    out_compile = compiled_sdpa(
        query=query,
        key=key,
        value=value,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=True,
        kernel_options=kernel_options,
    )

    forward_compiled_time = benchmark_torch_function_in_microseconds(
        compiled_sdpa,
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=True,
        kernel_options=kernel_options,
    )

    results = {}
    for backend in config.backends:
        if backend in ["fav2", "fav3", "fakv"]:
            results[backend] = run_single_backend_FA(
                config,
                query,
                key,
                value,
                out_compile,
                score_mod,
                block_mask,
                mask_kwargs,
                backend,
            )
        else:  # sdpa
            results[backend] = run_single_backend_sdpa(
                config,
                query,
                key,
                value,
                out_compile,
                score_mod,
                block_mask,
                mask_kwargs,
                backend,
            )

    if config.calculate_bwd_time:
        try:
            d_out = torch.randn_like(out_compile)
            backward_compile_time = benchmark_torch_function_in_microseconds(
                out_compile.backward, d_out, retain_graph=True
            )
        except Exception as e:
            print(
                f"[SKIP] Backward pass failed for {config.attn_type} with shape {config.shape}: {e}"
            )
            cleanup_memory()
            backward_compile_time = float("nan")
    sparsity = block_mask.sparsity() / 100.0 if block_mask is not None else 0.0
    sparsity = sparsity if config.attn_type != "document_mask" else 0.5

    compiled_result = ExperimentResults(
        fwd_time=forward_compiled_time,
        bwd_time=backward_compile_time if config.calculate_bwd_time else None,
        sparsity=sparsity,
    )
    results["flex_attn"] = add_metrics_to_result(config, compiled_result)

    return results


def calculate_speedup(
    results: ExperimentResults, baseline_results: ExperimentResults, type: SpeedupType
) -> float:
    if type == "fwd":
        return baseline_results.fwd_time / results.fwd_time
    elif type == "bwd":
        assert results.bwd_time is not None
        assert baseline_results.bwd_time is not None
        return baseline_results.bwd_time / results.bwd_time
    else:
        raise ValueError(f"Invalid type {type}")


def calculate_bandwidth(
    config: ExperimentConfig, results: ExperimentResults, type: SpeedupType
) -> float:
    B, Hq, M, Hkv, N, D = config.shape
    sparsity = (
        (results.sparsity if results.sparsity is not None else 0.0) if M == 1 else 0.0
    )
    bytes_per_element = torch.finfo(config.dtype).bits / 8

    if type == "fwd":
        # Forward pass memory accesses
        query_size = B * Hq * M * D * bytes_per_element
        kv_size = B * Hkv * N * D * bytes_per_element * 2  # K + V
        output_size = B * Hq * M * D * bytes_per_element
        total_size = (
            query_size + kv_size * (1 - sparsity) + output_size
        ) / 1e9  # In GB
        time_in_seconds = results.fwd_time / 1e6
        return total_size / time_in_seconds / 1e3  # TB/s

    elif type == "bwd":
        # Backward pass memory accesses
        # Delta computation - no sparsity applied
        o_reads_delta = 1 * B * Hq * M * D * bytes_per_element  # O for Delta
        do_reads_delta = 1 * B * Hq * M * D * bytes_per_element  # dO for Delta
        delta_writes = 1 * B * Hq * M * 1 * bytes_per_element  # Delta write

        # dQ kernel reads
        q_reads_dq = 1 * B * Hq * M * D * bytes_per_element  # Q (full)
        k_reads_dq = (
            1 * B * Hkv * N * D * bytes_per_element * (1 - sparsity)
        )  # K (sparse)
        v_reads_dq = (
            1 * B * Hkv * N * D * bytes_per_element * (1 - sparsity)
        )  # V (sparse)
        do_reads_dq = 1 * B * Hq * M * D * bytes_per_element  # dO (full)
        delta_reads_dq = 1 * B * Hq * M * 1 * bytes_per_element  # Delta (full)

        # dK/dV kernel reads
        q_reads_dk = 1 * B * Hq * M * D * bytes_per_element  # Q (full)
        k_reads_dk = (
            1 * B * Hkv * N * D * bytes_per_element * (1 - sparsity)
        )  # K (sparse)
        v_reads_dk = (
            1 * B * Hkv * N * D * bytes_per_element * (1 - sparsity)
        )  # V (sparse)
        do_reads_dk = 1 * B * Hq * M * D * bytes_per_element  # dO (full)
        delta_reads_dk = 1 * B * Hq * M * 1 * bytes_per_element  # Delta (full)

        # Gradient writes - always full
        dq_writes = 1 * B * Hq * M * D * bytes_per_element  # dQ (full)
        dk_writes = 1 * B * Hkv * N * D * bytes_per_element  # dK (full)
        dv_writes = 1 * B * Hkv * N * D * bytes_per_element  # dV (full)

        total_reads = (
            o_reads_delta
            + do_reads_delta
            + q_reads_dq
            + k_reads_dq
            + v_reads_dq
            + do_reads_dq
            + delta_reads_dq
            + q_reads_dk
            + k_reads_dk
            + v_reads_dk
            + do_reads_dk
            + delta_reads_dk
        )
        total_writes = delta_writes + dq_writes + dk_writes + dv_writes

        total_size = (total_reads + total_writes) / 1e9  # In GB
        assert results.bwd_time is not None
        time_in_seconds = results.bwd_time / 1e6
        return total_size / time_in_seconds / 1e3  # TB/s

    else:
        raise ValueError(f"Invalid type {type}")


def calculate_tflops(
    config: ExperimentConfig, results: ExperimentResults, type: SpeedupType
) -> float:
    (B, Hq, M, Hkv, N, D) = config.shape
    sparsity = results.sparsity if results.sparsity is not None else 0.0

    # Forward pass FLOPs
    qk_flops = M * N * D * 2
    softmax_flops = M * N * 2  # Not counting online softmax overhead
    o_flops = M * D * N * 2
    fwd_flops = B * Hq * (qk_flops + softmax_flops + o_flops) * (1 - sparsity)

    if type == "fwd":
        return fwd_flops / results.fwd_time / 1e6  # in TFLOPs
    elif type == "bwd":
        # Backward pass is 2.5x forward FLOPs (2.0 bwd + 0.5 recompute)
        bwd_flops = fwd_flops * 2.5
        assert results.bwd_time is not None
        return bwd_flops / results.bwd_time / 1e6  # in TFLOPs
    else:
        raise ValueError(f"Invalid type {type}")


def add_metrics_to_result(
    config: ExperimentConfig, results: ExperimentResults
) -> ExperimentResults:
    """Calculate TFLOPs and bandwidth for an ExperimentResults and return updated copy."""
    import math

    # Forward metrics
    if (
        results.fwd_time is None
        or math.isnan(results.fwd_time)
        or results.fwd_time <= 0
    ):
        fwd_tflops = float("nan")
        fwd_bandwidth = float("nan")
    else:
        fwd_tflops = calculate_tflops(config, results, type="fwd")
        fwd_bandwidth = calculate_bandwidth(config, results, type="fwd")

    # Backward metrics
    if (
        results.bwd_time is None
        or math.isnan(results.bwd_time)
        or results.bwd_time <= 0
    ):
        bwd_tflops = float("nan")
        bwd_bandwidth = float("nan")
    else:
        bwd_tflops = calculate_tflops(config, results, type="bwd")
        bwd_bandwidth = calculate_bandwidth(config, results, type="bwd")

    # Return new ExperimentResults with metrics
    return ExperimentResults(
        fwd_time=results.fwd_time,
        bwd_time=results.bwd_time,
        sparsity=results.sparsity,
        fwd_tflops=fwd_tflops,
        fwd_bandwidth=fwd_bandwidth,
        bwd_tflops=bwd_tflops,
        bwd_bandwidth=bwd_bandwidth,
    )


def get_average_speedups(
    results: list[Experiment], type: SpeedupType, backend: Backend
) -> list[dict]:
    # Calculate speedups
    speedups = [
        calculate_speedup(r.results["flex_attn"], r.results[backend], type)
        for r in results
    ]

    # Find indices of max and min speedups
    max_speedup_index = np.nanargmax(speedups)
    min_speedup_index = np.nanargmin(speedups)

    # Get the config dictionaries
    max_config_dict = results[max_speedup_index].config.asdict()
    min_config_dict = results[min_speedup_index].config.asdict()

    # Create table data
    table_data = [
        {
            "Type": "Average",
            "Speedup": np.nanmean(speedups),
            **dict.fromkeys(max_config_dict),
        },
        {"Type": "Max", "Speedup": speedups[max_speedup_index], **max_config_dict},
        {"Type": "Min", "Speedup": speedups[min_speedup_index], **min_config_dict},
    ]

    return table_data


def print_results(
    results: list[Experiment],
    save_path: Optional[str] = None,
    show_speedups: bool = False,
):
    table_data = defaultdict(list)

    # Add basic config columns first
    for experiment in results:
        for key, value in experiment.config.asdict().items():
            table_data[key].append(value)

    # Add metrics grouped by backend
    all_backends = results[0].config.backends + ["flex_attn", "cp_flex_attn"]
    for backend in all_backends:
        if backend in results[0].results:
            # Forward metrics grouped by backend
            fwd_times = [r.results[backend].fwd_time for r in results]
            table_data[f"{backend}_fwd_time"] = fwd_times

            fwd_tflops = [r.results[backend].fwd_tflops for r in results]
            table_data[f"{backend}_fwd_TFlops"] = fwd_tflops

            fwd_bandwidth = [r.results[backend].fwd_bandwidth for r in results]
            table_data[f"{backend}_fwd_mem_bw (TB/s)"] = fwd_bandwidth

            # Backward metrics (if available)
            if results[0].config.calculate_bwd_time:
                bwd_times = [r.results[backend].bwd_time for r in results]
                table_data[f"{backend}_bwd_time"] = bwd_times

                bwd_tflops = [r.results[backend].bwd_tflops for r in results]
                table_data[f"{backend}_bwd_TFlops"] = bwd_tflops

                bwd_bandwidth = [r.results[backend].bwd_bandwidth for r in results]
                table_data[f"{backend}_bwd_mem_bw (TB/s)"] = bwd_bandwidth

    # Calculate speedups (optional)
    if show_speedups:
        for backend in results[0].config.backends:
            fwd_speedups = [
                calculate_speedup(
                    r.results["flex_attn"], r.results[backend], type="fwd"
                )
                for r in results
            ]
            table_data[f"{backend}_fwd_speedup"] = fwd_speedups

        if results[0].config.calculate_bwd_time:
            for backend in results[0].config.backends:
                bwd_speedups = [
                    calculate_speedup(
                        r.results["flex_attn"], r.results[backend], type="bwd"
                    )
                    for r in results
                ]
                table_data[f"{backend}_bwd_speedup"] = bwd_speedups

    print(tabulate(table_data, headers="keys", tablefmt="github", floatfmt=".3f"))

    # Show speedup summaries (optional)
    if show_speedups:
        for backend in results[0].config.backends:
            if (
                f"fwd_{backend}_speedup" in table_data
                and not np.isnan(table_data[f"fwd_{backend}_speedup"]).all()
            ):
                print("\n")
                print(f"FWD Speedups vs. {backend}".center(125, "="))
                print("\n")
                average_data = get_average_speedups(
                    results, type="fwd", backend=backend
                )
                print(
                    tabulate(
                        average_data, headers="keys", tablefmt="github", floatfmt=".3f"
                    )
                )

                if results[0].config.calculate_bwd_time:
                    print("\n")
                    print(f"BWD Speedups vs. {backend}".center(125, "="))
                    print("\n")
                    average_data = get_average_speedups(
                        results, type="bwd", backend=backend
                    )
                    print(
                        tabulate(
                            average_data,
                            headers="keys",
                            tablefmt="github",
                            floatfmt=".3f",
                        )
                    )

    if save_path is not None:
        with open(save_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=table_data.keys())
            writer.writeheader()
            for i in range(len(next(iter(table_data.values())))):
                row = {k: v[i] for k, v in table_data.items()}
                writer.writerow(row)
        print(f"\nResults saved to {save_path}")


# Generate score_mods and BlockMasks
softcap_value = 50
dropout_p = 0.0


def generate_score_mod(
    attn_type: AttentionType, shape: tuple[int, int, int, int, int, int]
) -> Optional[Callable]:
    B, Hq, M, Hkv, N, D = shape
    is_decoding = M == 1
    from attn_gym.mods import generate_alibi_bias, generate_tanh_softcap

    def relative_bias(score, b, h, m, n):
        return score + (m - n)

    def head_bias(score, b, h, m, n):
        return score + 2 * h

    function_dict = {
        "noop": None,
        "causal": None,
        "rel": relative_bias,
        "head_bias": head_bias,
        "alibi": generate_alibi_bias(Hq),
        "sliding_window": None,
        "document_mask": None,
        "prefix_lm": None,
        "softcap": generate_tanh_softcap(softcap_value, approx=True),
    }

    score_mod = function_dict[attn_type]
    is_decoding = M == 1
    if is_decoding and score_mod:
        offset = torch.tensor(N // 2).to("cuda")

        def score_mod_w_offset(score, b, h, m, n):
            return score_mod(score, b, h, m + offset, n)

        new_score_mod = score_mod_w_offset
    else:
        new_score_mod = score_mod

    return new_score_mod


sliding_window_size = 512
prefix_length = 512


def generate_block_mask(
    attn_type: AttentionType, shape: tuple[int, int, int, int, int, int]
) -> tuple[BlockMask, dict]:
    B, Hq, M, Hkv, N, D = shape
    is_decoding = M == 1

    def causal(b, h, m, n):
        return m >= n

    def gen_offset(off):
        def offset(b, h, m, n):
            return m + off >= n

        return offset

    from attn_gym.masks import (
        generate_doc_mask_mod,
        generate_prefix_lm_mask,
        generate_sliding_window,
    )
    from attn_gym.masks.document_mask import length_to_offsets

    def generate_random_lengths(total_length, num_documents):
        # Initialize all lengths to 1 to ensure each document has at least one token
        lengths = [1] * num_documents
        remaining_length = total_length - num_documents

        # Randomly distribute the remaining length
        for _ in range(remaining_length):
            index = random.randint(0, num_documents - 1)
            lengths[index] += 1
        return lengths

    mask_mod_kwargs = {}

    assert attn_type != "document_mask" or not is_decoding
    if attn_type == "document_mask":
        random.seed(0)
        lengths = generate_random_lengths(N * B, B)
        mask_mod_kwargs = dict(offsets=length_to_offsets(lengths, "cuda"))

    mask_mod_dict = {
        "noop": None,
        "causal": causal,
        "rel": None,
        "head_bias": None,
        "alibi": causal,
        "sliding_window": generate_sliding_window(sliding_window_size),
        "document_mask": partial(generate_doc_mask_mod, mask_mod=causal),
        "prefix_lm": generate_prefix_lm_mask(prefix_length),
        "softcap": causal,
    }

    mask_mod = mask_mod_dict[attn_type]

    if mask_mod_kwargs:
        mask_mod = mask_mod(**mask_mod_kwargs)

    if is_decoding and mask_mod:
        cached_seq_len = torch.tensor(N // 2).to("cuda")

        def decoding_w_cached_seq_len(b, h, m, n):
            return mask_mod(b, h, m + cached_seq_len, n)

        new_mask_mod = decoding_w_cached_seq_len
    else:
        new_mask_mod = mask_mod

    mask_shape = (1, 1, M, N) if attn_type != "document_mask" else (1, 1, M * B, N * B)
    compiled_block_mask = torch.compile(create_block_mask)
    if new_mask_mod:
        block_mask = compiled_block_mask(new_mask_mod, *mask_shape, "cuda")
    else:
        block_mask = compiled_block_mask(noop_mask, *mask_shape, "cuda")
    return block_mask, mask_mod_kwargs


def get_kernel_options(
    attn_type: AttentionType, shape: tuple[int, int, int, int, int, int]
) -> Optional[dict]:
    B, Hq, M, Hkv, N, D = shape
    is_decoding = M == 1
    kernel_opt_training_dict = {
        "noop": None,
        "causal": None,
        "rel": None,
        "head_bias": None,
        "alibi": None,
        "sliding_window": None,
        "document_mask": (
            {
                "BLOCK_N": 32,
                "BLOCK_M": 128,
                "fwd_num_warps": 8,
                "fwd_num_stages": 4,
                "BLOCK_M1": 64,
                "BLOCK_N1": 64,
                "BLOCK_M2": 64,
                "BLOCK_N2": 64,
            }
            if torch.cuda.get_device_capability() >= (8, 0) and D <= 128
            else None
        ),
        "prefix_lm": None,
        "softcap": None,
    }

    def get_default_split_k(B: int, H: int, Mk: int) -> int:
        num_SM = torch.cuda.get_device_properties("cuda").multi_processor_count
        """Heuristic for the number of splits from xformer"""
        bh = max(B * H, 1)  # NOTE: Handle B*h=0 case
        split_k = num_SM // bh * 2  # Each SM should at least get one block.
        split_k = max(split_k, 1)

        return split_k

    kernel_opt_decoding_dict = {
        "noop": None,
        "causal": {"SPLIT_KV": get_default_split_k(B, Hkv, N) * 2},
        "rel": None,
        "head_bias": None,
        "alibi": {"SPLIT_KV": get_default_split_k(B, Hkv, N) * 2},
        "sliding_window": None,
        "document_mask": None,
        "prefix_lm": None,
        "softcap": {"SPLIT_KV": get_default_split_k(B, Hkv, N) * 2},
    }

    return (
        kernel_opt_decoding_dict[attn_type]
        if is_decoding
        else kernel_opt_training_dict[attn_type]
    )


# Setup Backend


def get_backend_context(backend: Backend):
    """
    Returns a context manager for the specified backend.
    Args:
        backend (str): The name of the backend to use.
                       Valid options are 'fav2', 'cudnn', 'math', 'efficient', 'fav3', 'fakv', 'og-eager'.
    Returns:
        A context manager for the specified backend.
    Raises:
        ValueError: If an invalid backend is specified.
    """
    backends = {
        "fav2": nullcontext(),
        "cudnn": sdpa_kernel(SDPBackend.CUDNN_ATTENTION),
        "math": sdpa_kernel(SDPBackend.MATH),
        "efficient": sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION),
        "fav3": nullcontext(),
        "fakv": nullcontext(),
        "og-eager": nullcontext(),
    }

    if backend not in backends:
        raise ValueError(
            f"Unknown backend: {backend}. Valid options are: {', '.join(backends.keys())}"
        )

    return backends[backend]


def generate_FA_callable(
    attn_type: AttentionType,
    shape: tuple[int, int, int, int, int, int],
    dtype: torch.dtype,
    backend: Backend,
    **kwargs,
) -> Optional[Callable]:
    if dtype not in [torch.float16, torch.bfloat16]:
        return None
    if backend == "fav2":
        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
        except ImportError:
            print(
                "Flash attention 2 is not installed. Please install it to run fav2 backend. "
            )
            raise
    elif backend == "fav3":
        try:
            from flash_attn.flash_attn_interface import (
                flash_attn_func,
                flash_attn_varlen_func,
            )
        except ImportError:
            print(
                "Flash attention 3 is not installed. Please install it to run fav3 backend. "
            )
            raise
    else:
        print("Unknown backend " + backend)
        return None

    B, Hq, M, Hkv, N, D = shape

    FA_kwargs = {}
    if attn_type == "alibi":
        h = torch.arange(Hq, dtype=torch.float32, device="cuda")
        alibi_slopes = torch.exp2(-((h + 1) * 8.0 / Hq))
        FA_kwargs = dict(alibi_slopes=alibi_slopes)
    elif attn_type == "document_mask":
        FA_kwargs["cu_seqlens_q"] = kwargs["offsets"].to(torch.int32)
        FA_kwargs["cu_seqlens_k"] = kwargs["offsets"].to(torch.int32)

        def offsets_to_lengths(
            offsets: torch.Tensor, device: Union[str, torch.device]
        ) -> torch.Tensor:
            lengths = offsets[1:] - offsets[:-1]
            return lengths

        lengths = offsets_to_lengths(kwargs["offsets"], "cpu")
        max_length = torch.max(lengths)
        FA_kwargs["max_seqlen_q"] = max_length
        FA_kwargs["max_seqlen_k"] = max_length

    FA_dict = {
        "noop": partial(flash_attn_func, causal=False),
        "causal": partial(flash_attn_func, causal=True),
        "rel": None,
        "head_bias": None,
        "alibi": partial(flash_attn_func, causal=True, **FA_kwargs),
        "sliding_window": partial(
            flash_attn_func, window_size=(sliding_window_size, 0), causal=True
        ),
        "document_mask": partial(flash_attn_varlen_func, causal=True, **FA_kwargs),
        "prefix_lm": None,
        "softcap": partial(flash_attn_func, softcap=softcap_value, causal=True),
    }

    return FA_dict[attn_type]


def generate_FD_callable(
    attn_type: AttentionType,
    shape: tuple[int, int, int, int, int, int],
    dtype: torch.dtype,
) -> Optional[Callable]:
    if dtype not in [torch.float16, torch.bfloat16]:
        return None
    try:
        from flash_attn import flash_attn_with_kvcache
    except ImportError:
        print(
            "Flash attention 2 is not installed. Please install it to run fakv backend. "
        )
        raise

    B, Hq, M, Hkv, N, D = shape

    assert M == 1

    def flash_attn_with_kvcache_renamed(q, k, v, **kwargs):
        return flash_attn_with_kvcache(q, k_cache=k, v_cache=v, **kwargs)

    FA_kwargs = {}
    if attn_type == "alibi":
        h = torch.arange(Hq, dtype=torch.float32, device="cuda")
        alibi_slopes = torch.exp2(-((h + 1) * 8.0 / Hq))
        FA_kwargs = dict(alibi_slopes=alibi_slopes)

    FD_dict = {
        "noop": partial(flash_attn_with_kvcache_renamed, causal=False),
        "causal": partial(flash_attn_with_kvcache_renamed, cache_seqlens=N // 2),
        "rel": None,
        "head_bias": None,
        "alibi": partial(
            flash_attn_with_kvcache_renamed, cache_seqlens=N // 2, **FA_kwargs
        ),
        "sliding_window": partial(
            flash_attn_with_kvcache_renamed,
            cache_seqlens=N // 2,
            window_size=(sliding_window_size, 0),
        ),
        "document_mask": None,
        "prefix_lm": None,
        "softcap": partial(flash_attn_with_kvcache_renamed, softcap=softcap_value),
    }

    return FD_dict[attn_type]


def generate_attn_mask_linear_score_mod(
    shape: tuple[int, int, int, int],
    block_mask: BlockMask,
    score_mod: Callable,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    B, Hq, M, N = shape
    if block_mask is None and score_mod is None:
        return None
    b = torch.arange(B, dtype=torch.int32, device="cuda")
    h = torch.arange(Hq, dtype=torch.int32, device="cuda")
    m = torch.arange(M, dtype=torch.int32, device="cuda")
    n = torch.arange(N, dtype=torch.int32, device="cuda")

    score = torch.zeros(B, Hq, M, N, dtype=dtype, device="cuda")
    bias = score_mod(
        score,
        b[:, None, None, None],
        h[None, :, None, None],
        m[None, None, :, None],
        n[None, None, None, :],
    )
    bool_mask = create_mask(block_mask.mask_mod, B, Hq, M, N, device="cuda")
    attn_mask = bias.masked_fill(bool_mask.logical_not(), float("-inf"))
    return attn_mask.to(dtype)


def generate_eager_sdpa(
    attn_type: AttentionType,
    shape: tuple[int, int, int, int, int, int],
    dtype: torch.dtype,
    block_mask: Optional[BlockMask],
    score_mod: Optional[Callable] = None,
    **kwargs,
) -> Optional[Callable]:
    B, Hq, M, Hkv, N, D = shape
    is_decoding = M == 1
    if attn_type == "sliding_window" or attn_type == "prefix_lm":
        assert block_mask is not None
        attn_mask = create_mask(block_mask.mask_mod, 1, 1, M, N, device="cuda")
    elif attn_type == "rel":
        assert block_mask is not None and score_mod is not None
        attn_mask = generate_attn_mask_linear_score_mod(
            (1, 1, M, N), block_mask, score_mod, dtype
        )
    elif attn_type == "head_bias":
        h = torch.arange(Hq, dtype=torch.int32, device="cuda")
        attn_mask = (2 * h[None, :, None, None]).broadcast_to(1, Hq, M, N).to(dtype)
    elif attn_type == "alibi":
        assert block_mask is not None and score_mod is not None
        attn_mask = generate_attn_mask_linear_score_mod(
            (1, Hq, M, N), block_mask, score_mod, dtype
        )
    else:
        attn_mask = None

    sdpa_dict = {
        "noop": partial(
            F.scaled_dot_product_attention, is_causal=False, enable_gqa=(Hq != Hkv)
        ),
        "causal": partial(
            F.scaled_dot_product_attention, is_causal=True, enable_gqa=(Hq != Hkv)
        ),
        "rel": partial(
            F.scaled_dot_product_attention, is_causal=False, enable_gqa=(Hq != Hkv)
        ),
        "head_bias": partial(
            F.scaled_dot_product_attention, is_causal=False, enable_gqa=(Hq != Hkv)
        ),
        "alibi": partial(
            F.scaled_dot_product_attention, is_causal=False, enable_gqa=(Hq != Hkv)
        ),
        "sliding_window": partial(
            F.scaled_dot_product_attention, is_causal=False, enable_gqa=(Hq != Hkv)
        ),
        "document_mask": (
            partial(
                F.scaled_dot_product_attention, is_causal=True, enable_gqa=(Hq != Hkv)
            )
            if Hq == Hkv
            else None
        ),
        "prefix_lm": partial(
            F.scaled_dot_product_attention, is_causal=False, enable_gqa=(Hq != Hkv)
        ),
        "softcap": None,
    }

    if is_decoding and attn_type == "causal":
        assert block_mask is not None
        attn_mask = create_mask(block_mask.mask_mod, 1, 1, M, N, device="cuda")
        sdpa_dict["causal"] = partial(
            F.scaled_dot_product_attention, is_causal=False, enable_gqa=(Hq != Hkv)
        )

    return (
        partial(sdpa_dict[attn_type], attn_mask=attn_mask)
        if sdpa_dict[attn_type]
        else None
    )


def generate_experiment_configs(
    calculate_bwd: bool,
    dtype: torch.dtype,
    batch_sizes: list[int],
    num_heads: list[tuple[int, int]],
    seq_lens: list[int],
    head_dims: list[int],
    score_mods_str: list[AttentionType],
    decoding: bool,
    kv_cache_size: Optional[list[int]],
    cal_bandwidth: bool,
    backends: list[Backend],
) -> list[ExperimentConfig]:
    assert not (calculate_bwd and decoding), "Decoding does not support backward"

    if decoding:
        q_kv_seq_lens = [(1, i) for i in seq_lens]  # only testing query length == 1
    else:
        q_kv_seq_lens = [(i, i) for i in seq_lens]  # only testing q_len == kv_len
    dtypes = [dtype]

    all_configs = []
    for (
        bsz,
        (q_heads, kv_heads),
        (q_seq_len, kv_seq_len),
        head_dim,
        attn_type,
        dtype,
    ) in itertools.product(
        kv_cache_size if kv_cache_size else batch_sizes,
        num_heads,
        q_kv_seq_lens,
        head_dims,
        score_mods_str,
        dtypes,
    ):
        if kv_cache_size:
            head_size_bytes = torch.finfo(dtype).bits / 8 * head_dim
            bsz = int(
                (bsz * 1024 * 1024) // (kv_heads * kv_seq_len * head_size_bytes * 2)
            )
            if bsz <= 0:
                continue

        assert q_heads % kv_heads == 0

        all_configs.append(
            ExperimentConfig(
                shape=(bsz, q_heads, q_seq_len, kv_heads, kv_seq_len, head_dim),
                attn_type=attn_type,
                dtype=dtype,
                calculate_bwd_time=calculate_bwd,
                cal_bandwidth=cal_bandwidth,
                backends=backends,
            )
        )

    return all_configs


def heads_input_type(s: str) -> tuple[int, int]:
    try:
        hq, hkv = map(int, s.split(","))
        return hq, hkv
    except Exception as e:
        raise ValueError("Heads must be Hq,Hkv") from e


def main(
    dynamic: bool = False,
    calculate_bwd: bool = False,
    dtype: DtypeString = "bfloat16",
    b: list[int] = [2, 8, 16],
    nh: list[str] = ["16,16", "16,2"],
    s: list[int] = [512, 1024, 4096],
    d: list[int] = [64, 128],
    mods: list[AttentionType] = ["noop", "causal", "alibi", "sliding_window"],
    backend: list[Backend] = ["efficient"],
    max_autotune: bool = False,
    decoding: bool = False,
    kv_size: Optional[list[int]] = None,
    throughput: bool = True,
    show_speedups: bool = False,
    save_path: Optional[str] = None,
    kernel_options: Optional[dict] = None,
) -> None:
    """Run sweep over sizes and score mods for flex attention.

    Usage Examples:
        # Generate a config template
        python benchmarks/flex_perf.py --print_config > my_config.yaml

        # Use a config file
        python benchmarks/flex_perf.py --config benchmarks/configs/config_basic.yaml

        # Override config with CLI args
        python benchmarks/flex_perf.py --config benchmarks/configs/config_basic.yaml --dtype float16 --max_autotune

        # Pure CLI usage
        python benchmarks/flex_perf.py --b 4 8 --s 1024 2048 --mods causal alibi --backend efficient fav2

    Available config files in benchmarks/configs/:
        - config_basic.yaml: Basic benchmark setup
        - config_comprehensive.yaml: Full feature sweep with backward pass
        - config_decoding.yaml: Decoding-specific benchmarks
        - config_memory_bound.yaml: Memory-constrained scenarios

    Args:
        dynamic: Runs a dynamic shapes version of compiled flex attention
        calculate_bwd: Calculate backward pass times
        dtype: Data type for tensors (bfloat16, float16, float32)
        b: Batch sizes to benchmark
        nh: Number of query and key/value heads in format "Hq,Hkv"
        s: Sequence lengths to benchmark
        d: Head dimensions to benchmark
        mods: Score modifications: noop, causal, rel, head_bias, alibi, sliding_window, document_mask, prefix_lm, softcap
        backend: Backends for attention computation: math, efficient, cudnn, fav2, fav3, fakv
        max_autotune: Turn on max-autotune optimization
        decoding: Benchmark decoding mode (query sequence length = 1)
        kv_size: Key/value cache size in MiB (ignores batch size if specified)
        throughput: Calculate kernel memory bandwidth & computational throughput (always True)
        show_speedups: Show speedup calculations in output
        save_path: Path to save the results CSV file
        kernel_options: Dict of overrides merged into defaults from get_kernel_options. Either a
            flat dict applied to all types, or a dict keyed by attention type, with optional
            "global" key as fallback.
    """
    # Convert dtype string to torch dtype
    import torch

    dtype = getattr(torch, dtype)

    # Parse head configurations
    nh_parsed = [heads_input_type(h) for h in nh]

    # Always calculate throughput
    throughput = True

    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    results = []
    experiment_count = 0
    for exp_config in tqdm(
        generate_experiment_configs(
            calculate_bwd,
            dtype,
            b,
            nh_parsed,
            s,
            d,
            mods,
            decoding,
            kv_size,
            throughput,
            backend,
        )
    ):
        experiment_result = run_single_experiment(
            exp_config,
            dynamic=dynamic,
            max_autotune=max_autotune,
            kernel_options_override=kernel_options,
        )
        results.append(Experiment(exp_config, experiment_result))

        experiment_count += 1
        # Periodic memory cleanup every 10 experiments
        if experiment_count % 10 == 0:
            cleanup_memory()

    print_results(results, save_path, show_speedups=show_speedups)


if __name__ == "__main__":
    CLI(main, as_positional=False)
