import argparse
import csv
import itertools
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    create_mask,
    flex_attention,
    noop_mask,
)


torch._dynamo.config.automatic_dynamic_shapes = False
# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


from torch._inductor.runtime.benchmarking import benchmarker


def benchmark_torch_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    # warmup
    for _ in range(5):
        func(*args, **kwargs)
    return benchmarker.benchmark_gpu(lambda: func(*args, **kwargs)) * 1e3


@dataclass(frozen=True)
class ExperimentConfig:
    shape: Tuple[int]  # [B, Hq, M, Hkv, N, D]
    attn_type: str
    dtype: torch.dtype
    calculate_bwd_time: bool
    cal_bandwidth: bool

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
        return d


@dataclass(frozen=True)
class Times:
    eager_time: float
    compiled_time: float
    FA_time: float | None


@dataclass(frozen=True)
class ExperimentResults:
    fwd_times: Times
    bwd_times: Optional[Times]


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    results: ExperimentResults

    def asdict(self):
        dict1 = self.config.asdict()
        dict2 = asdict(self.results)
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
):
    q_shape = (batch_size, q_sequence_length, q_heads * head_dim)
    kv_shape = (batch_size, kv_sequence_length, kv_heads * head_dim)

    assert q_heads % kv_heads == 0

    num_h_groups = q_heads // kv_heads

    make_q = partial(
        torch.rand, q_shape, device=device, dtype=dtype, requires_grad=requires_grad
    )
    make_kv = partial(
        torch.rand, kv_shape, device=device, dtype=dtype, requires_grad=requires_grad
    )
    query = (
        make_q().view(batch_size, q_sequence_length, q_heads, head_dim).transpose(1, 2)
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


def extend_offsets(offsets, B):
    batch_lengths = offsets[-1]
    offsets_ref = offsets[:-1]
    offsets = [offsets_ref + batch_lengths * i for i in range(B)]
    offsets.append(
        torch.tensor(
            [batch_lengths * B], device=offsets_ref.device, dtype=offsets_ref.dtype
        )
    )
    offsets = torch.cat(offsets, dim=0)
    return offsets


def generate_jagged_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    offsets: torch.Tensor,
):
    B, Hq, M, D = query.shape
    _, Hkv, N, _ = key.shape

    def offsets_to_lengths(
        offsets: torch.Tensor, device: Union[str, torch.device]
    ) -> torch.tensor:
        """Converts a list of offsets to a list of lengths. Reverse op of attn_gym.masks.document_mask.length_to_offsets

        Args:
            offsets: A 1D tensor of offsets
            device: The device to place the output tensor on
        """
        lengths = offsets[1:] - offsets[:-1]
        return lengths

    offsets = extend_offsets(offsets, B)

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
    dtype: torch.dtype = None,
):
    """Clones the query, key, and value tensors and moves them to the specified dtype."""
    if dtype is None:
        dtype = query.dtype
    query_ref = query.clone().detach().to(dtype).requires_grad_(query.requires_grad)
    key_ref = key.clone().detach().to(dtype).requires_grad_(key.requires_grad)
    value_ref = value.clone().detach().to(dtype).requires_grad_(value.requires_grad)
    return query_ref, key_ref, value_ref


def run_single_experiment(
    config: ExperimentConfig,
    dynamic=False,
    max_autotune=False,
    run_FA=False,
    og_eager=False,
) -> ExperimentResults:
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
    )
    is_decoding = q_seq_len == 1

    score_mod = generate_score_mod(config.attn_type, config.shape)
    block_mask, mask_kwargs = generate_block_mask(config.attn_type, config.shape)
    if run_FA:
        FA = (
            generate_FD_callable(config.attn_type, config.shape, config.dtype)
            if is_decoding
            else generate_FA_callable(
                config.attn_type, config.shape, config.dtype, **mask_kwargs
            )
        )
    else:
        FA = None

    if og_eager:
        from score_mod_helper import generate_OG_pytorch

        eager_sdpa = generate_OG_pytorch(
            config.attn_type, config.shape, config.dtype, block_mask
        )
    else:
        eager_sdpa = generate_eager_sdpa(
            config.attn_type, config.shape, config.dtype, block_mask
        )

    if max_autotune:
        compiled_sdpa = torch.compile(
            flex_attention, dynamic=dynamic, mode="max-autotune-no-cudagraphs"
        )
    else:
        compiled_sdpa = torch.compile(flex_attention, dynamic=dynamic)

    if config.attn_type == "document_mask":
        q_eager, k_eager, v_eager = generate_jagged_inputs(
            query, key, value, **mask_kwargs
        )
        q_eager = q_eager.transpose(1, 2).requires_grad_(query.requires_grad)
        k_eager = k_eager.transpose(1, 2).requires_grad_(key.requires_grad)
        v_eager = v_eager.transpose(1, 2).requires_grad_(value.requires_grad)
    else:
        q_eager, k_eager, v_eager = query_key_value_clones(query, key, value)

    q_FA, k_FA, v_FA = query_key_value_clones(query, key, value)
    q_FA, k_FA, v_FA = q_FA.transpose(1, 2), k_FA.transpose(1, 2), v_FA.transpose(1, 2)
    if config.attn_type == "document_mask":
        q_FA = q_FA.flatten(start_dim=0, end_dim=1)
        k_FA = k_FA.flatten(start_dim=0, end_dim=1)
        v_FA = v_FA.flatten(start_dim=0, end_dim=1)

    out_compile = compiled_sdpa(
        query=query,
        key=key,
        value=value,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=True,
    )

    if eager_sdpa:
        out_eager = eager_sdpa(query=q_eager, key=k_eager, value=v_eager)
        if config.attn_type in ["document_mask"]:
            flatten_o_eager = torch.cat(torch.unbind(out_eager.transpose(1, 2)))
            flatten_o_compile = out_compile.transpose(1, 2).flatten(
                start_dim=0, end_dim=1
            )
            torch.testing.assert_close(
                flatten_o_eager, flatten_o_compile, atol=1e-2, rtol=1e-2
            )
        elif not (
            config.attn_type in ["rel", "alibi"]
            and config.dtype in [torch.float16, torch.bfloat16]
        ):  # rel has accuracy issue with 16bit floats
            torch.testing.assert_close(out_eager, out_compile, atol=1e-2, rtol=1e-2)

    if FA:
        out_FA = FA(q=q_FA, k=k_FA, v=v_FA)
        if config.attn_type in ["document_mask"]:
            B, Hq, M, Hkv, N, D = config.shape
            out_FA_updated = out_FA.reshape(B, -1, Hq, D)
        else:
            out_FA_updated = out_FA
        torch.testing.assert_close(
            out_FA_updated, out_compile.transpose(1, 2), atol=1e-2, rtol=1e-2
        )

    if eager_sdpa:
        forward_eager_time = benchmark_torch_function_in_microseconds(
            eager_sdpa, query=q_eager, key=k_eager, value=v_eager
        )
    else:
        forward_eager_time = float("nan")

    forward_compiled_time = benchmark_torch_function_in_microseconds(
        compiled_sdpa,
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=True,
    )
    if FA:
        forward_FA_time = benchmark_torch_function_in_microseconds(
            FA, q=q_FA, k=k_FA, v=v_FA
        )
    else:
        forward_FA_time = float("nan")

    if config.calculate_bwd_time:
        # TODO: debug backward pass for njt
        if eager_sdpa and not config.attn_type == "document_mask":
            dOut = torch.randn_like(out_eager.transpose(1, 2)).transpose(1, 2)
            backward_eager_time = benchmark_torch_function_in_microseconds(
                out_eager.backward, dOut, retain_graph=True
            )
        else:
            backward_eager_time = float("nan")

        dOut = torch.randn_like(out_compile)
        backward_compile_time = benchmark_torch_function_in_microseconds(
            out_compile.backward, dOut, retain_graph=True
        )

        if FA:
            dOut = torch.randn_like(out_FA)
            backward_FA_time = benchmark_torch_function_in_microseconds(
                out_FA.backward, dOut, retain_graph=True
            )
        else:
            backward_FA_time = float("nan")

        return ExperimentResults(
            fwd_times=Times(
                forward_eager_time,
                forward_compiled_time,
                forward_FA_time if run_FA else None,
            ),
            bwd_times=Times(
                backward_eager_time,
                backward_compile_time,
                backward_FA_time if run_FA else None,
            ),
        )
    else:
        return ExperimentResults(
            fwd_times=Times(
                forward_eager_time,
                forward_compiled_time,
                forward_FA_time if run_FA else None,
            ),
            bwd_times=None,
        )


def calculate_speedup(results: ExperimentResults, type: str) -> float:
    if type == "fwd":
        return results.fwd_times.eager_time / results.fwd_times.compiled_time
    elif type == "bwd":
        assert results.bwd_times is not None
        return results.bwd_times.eager_time / results.bwd_times.compiled_time
    elif type == "FA_fwd":
        return results.fwd_times.FA_time / results.fwd_times.compiled_time
    elif type == "FA_bwd":
        assert results.bwd_times is not None
        return results.bwd_times.FA_time / results.bwd_times.compiled_time
    else:
        raise ValueError(f"Invalid type {type}")


def calculate_bandwidth(
    config: ExperimentConfig, results: ExperimentResults, type: str
) -> float:
    if type == "fwd":
        batch_size, q_heads, q_seq_len, kv_heads, kv_seq_len, head_dim = config.shape
        query_size = (
            batch_size
            * q_heads
            * q_seq_len
            * head_dim
            * torch.finfo(config.dtype).bits
            / 8
        )
        kv_size = (
            batch_size
            * kv_heads
            * kv_seq_len
            * head_dim
            * torch.finfo(config.dtype).bits
            / 8
            * 2
        )
        output_size = query_size
        total_size = (query_size + kv_size + output_size) / 1e9  # In GB
        time_in_seconds = results.fwd_times.compiled_time / 1e6
        return total_size / time_in_seconds / 1e3
    else:
        raise ValueError(f"Invalid type {type}")


def calculate_tflops(config: ExperimentConfig, results: ExperimentResults) -> float:
    (B, Hq, M, Hkv, N, D) = config.shape
    qk_flops = M * N * D * 2
    softmax_flops = M * N * 2  # Not counting online softmax overhead
    o_flops = M * D * N * 2
    # Not counting split k overhead
    total_flops = B * Hq * (qk_flops + softmax_flops + o_flops)
    return total_flops / results.fwd_times.compiled_time / 1e6  # in TFLOPs/


def get_average_speedups(results: List[Experiment], type: str):
    # Calculate speedups
    speedups = [calculate_speedup(r.results, type) for r in results]

    # Find indices of max and min speedups
    max_speedup_index = np.argmax(speedups)
    min_speedup_index = np.argmin(speedups)

    # Get the config dictionaries
    max_config_dict = results[max_speedup_index].config.asdict()
    min_config_dict = results[min_speedup_index].config.asdict()

    # Create table data
    table_data = [
        {
            "Type": "Average",
            "Speedup": np.mean(speedups),
            **dict.fromkeys(max_config_dict),
        },
        {"Type": "Max", "Speedup": speedups[max_speedup_index], **max_config_dict},
        {"Type": "Min", "Speedup": speedups[min_speedup_index], **min_config_dict},
    ]

    return table_data


def print_results(
    results: List[Experiment], cal_FA: bool, save_path: Optional[str] = None
):
    table_data = defaultdict(list)
    for experiment in results:
        for key, value in experiment.asdict().items():
            if key == "fwd_times":
                for name, time in value.items():
                    if time:
                        table_data[f"fwd_{name}"].append(float(time))
            elif key == "bwd_times":
                if experiment.config.calculate_bwd_time:
                    for name, time in value.items():
                        if time:
                            table_data[f"bwd_{name}"].append(float(time))
            else:
                table_data[key].append(value)

    # Calculate speedups
    fwd_speedups = [calculate_speedup(r.results, type="fwd") for r in results]
    table_data["fwd_speedup"] = fwd_speedups

    if cal_FA:
        fwd_speedup_FA = [calculate_speedup(r.results, type="FA_fwd") for r in results]
        table_data["fwd_speedup_FA"] = fwd_speedup_FA

    # Calculate mem + computational throughput
    if results[0].config.cal_bandwidth:
        fwd_bandwidth = [
            calculate_bandwidth(r.config, r.results, type="fwd") for r in results
        ]
        table_data["fwd_mem_bw (TB/s)"] = fwd_bandwidth
        fwd_tflops = [calculate_tflops(r.config, r.results) for r in results]
        table_data["TFlops/s"] = fwd_tflops

    if results[0].config.calculate_bwd_time:
        bwd_speedups = [calculate_speedup(r.results, type="bwd") for r in results]
        table_data["bwd_speedup"] = bwd_speedups

        if cal_FA:
            bwd_speedup_FA = [
                calculate_speedup(r.results, type="FA_bwd") for r in results
            ]
            table_data["bwd_speedup_FA"] = bwd_speedup_FA

    print(tabulate(table_data, headers="keys", tablefmt="github", floatfmt=".3f"))
    print("\n")
    print("FWD Speedups".center(125, "="))
    print("\n")
    average_data = get_average_speedups(results, type="fwd")
    print(tabulate(average_data, headers="keys", tablefmt="github", floatfmt=".3f"))

    if results[0].config.calculate_bwd_time:
        print("\n")
        print("BWD Speedups".center(125, "="))
        print("\n")
        average_data = get_average_speedups(results, type="bwd")
        print(tabulate(average_data, headers="keys", tablefmt="github", floatfmt=".3f"))

    if save_path is not None:
        with open(save_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=table_data.keys())
            writer.writeheader()
            for i in range(len(next(iter(table_data.values())))):
                row = {k: v[i] for k, v in table_data.items()}
                writer.writerow(row)
        print(f"\nResults saved to {save_path}")


###################### Generate score_mods and BlockMasks ######################
softcap_value = 50
dropout_p = 0.0


def generate_score_mod(attn_type: str, shape: Tuple[int]) -> Callable | None:
    B, Hq, M, Hkv, N, D = shape
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
        "prefix_ml": None,
        "softcap": generate_tanh_softcap(softcap_value, approx=True),
        "transfusion": generate_tanh_softcap(softcap_value, approx=True),
    }
    return function_dict[attn_type]


sliding_window_size = 512
prefix_length = 512


def generate_block_mask(attn_type: str, shape: Tuple[int]):
    B, Hq, M, Hkv, N, D = shape
    is_decoding = M == 1

    def causal(b, h, m, n):
        return m >= n

    def gen_offset(off):
        def offset(b, h, m, n):
            return m + off >= n

        return offset

    import jaxtyping

    class TorchTyping:
        def __init__(self, abstract_dtype):
            self.abstract_dtype = abstract_dtype

        def __getitem__(self, shapes: str):
            return self.abstract_dtype[torch.Tensor, shapes]

    Int = TorchTyping(jaxtyping.Int)

    def transfusion_attn_mask(modalities: Int["b m 3"]):
        def modality(offset, length):
            def mask_fn(b, h, q_idx, kv_idx):
                return (q_idx >= offset) & (kv_idx < (offset + length))

            return mask_fn

        modalities = modalities.long()

        def mask_mod(b, h, q_idx, kv_idx):
            mask = causal(b, h, q_idx, kv_idx)

            modality_batch = modalities[b]

            for _, offset, length in modality_batch:
                mask = mask | modality(offset, length)(b, h, q_idx, kv_idx)

            return mask

        return mask_mod

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

    def generate_randome_modalities(num_modalities, max_length):
        lengths = generate_random_lengths(max_length, num_modalities * 2 + 1)
        modalities = []
        acc_offset = 0
        for modality_id in range(num_modalities):
            acc_offset += lengths[modality_id * 2]
            offset = acc_offset
            length = lengths[modality_id * 2 + 1]
            acc_offset += length
            modalities.append((0, offset, length))
        return modalities

    RawModalityPositions = list[list[tuple[int, int, int]]]

    def modality_positions_to_tensor(
        modalities: RawModalityPositions, pad_value=0, device=None
    ) -> Int["b m 2"] | Int["b m 3"]:
        from torch.nn.utils.rnn import pad_sequence

        pad_sequence = partial(pad_sequence, batch_first=True)

        modalities: list[torch.Tensor] = [
            torch.tensor(modality, device=device) for modality in modalities
        ]
        modalities = pad_sequence(modalities, padding_value=pad_value)

        if modalities.ndim == 2:
            modalities = modalities.reshape(*modalities.shape, 3)

        return modalities.long()

    mask_mod_kwargs = {}

    assert attn_type != "document_mask" or not is_decoding
    if attn_type == "document_mask":
        random.seed(0)
        lengths = generate_random_lengths(N, 4)
        mask_mod_kwargs = dict(offsets=length_to_offsets(lengths, "cuda"))
    elif attn_type == "transfusion":
        random.seed(0)
        modalities = modality_positions_to_tensor(
            [generate_randome_modalities(2, N) for i in range(B)], device="cuda"
        )
        mask_mod_kwargs = dict(modalities=modalities)

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
        "transfusion": transfusion_attn_mask,
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

    if new_mask_mod:
        block_mask = create_block_mask(new_mask_mod, 1, 1, M, N, "cuda")
    else:
        block_mask = create_block_mask(noop_mask, 1, 1, M, N, "cuda")

    return block_mask, mask_mod_kwargs


def generate_FA_callable(
    attn_type: str, shape: Tuple[int], dtype: torch.dtype, **kwargs
) -> Callable | None:
    if dtype not in [torch.float16, torch.bfloat16]:
        return None
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    B, Hq, M, Hkv, N, D = shape

    FA_kwargs = {}
    if attn_type == "alibi":
        h = torch.arange(Hq, dtype=torch.float32, device="cuda")
        alibi_slopes = -torch.exp2(-((h + 1) * 8.0 / Hq))
        FA_kwargs = dict(alibi_slopes=alibi_slopes)
    elif attn_type == "document_mask":
        extended_offsets = extend_offsets(kwargs["offsets"], B)
        FA_kwargs["cu_seqlens_q"] = extended_offsets.to(torch.int32)
        FA_kwargs["cu_seqlens_k"] = extended_offsets.to(torch.int32)

        def offsets_to_lengths(
            offsets: torch.Tensor, device: Union[str, torch.device]
        ) -> torch.tensor:
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
        "transfusion": None,
    }

    return FA_dict[attn_type]


def generate_FD_callable(
    attn_type: str, shape: Tuple[int], dtype: torch.dtype
) -> Callable | None:
    if dtype not in [torch.float16, torch.bfloat16]:
        return None
    from flash_attn import flash_attn_with_kvcache

    B, Hq, M, Hkv, N, D = shape

    def flash_attn_with_kvcache_renamed(q, k, v, **kwargs):
        return flash_attn_with_kvcache(q, k_cache=k, v_cache=v, **kwargs)

    FA_kwargs = {}
    if attn_type == "alibi":
        h = torch.arange(Hq, dtype=torch.float32, device="cuda")
        alibi_slopes = -torch.exp2(-((h + 1) * 8.0 / Hq))
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
        "softcap": None,
        "transfusion": None,
    }

    return FD_dict[attn_type]


def generate_eager_sdpa(
    attn_type: str,
    shape: Tuple[int],
    dtype: torch.dtype,
    block_mask: BlockMask,
    **kwargs,
) -> Callable | None:
    B, Hq, M, Hkv, N, D = shape
    is_decoding = M == 1
    if attn_type == "sliding_window" or attn_type == "prefix_ml":
        attn_mask = create_mask(block_mask.mask_mod, 1, 1, M, N, device="cuda")
    elif attn_type == "rel":
        m = torch.arange(M, dtype=int, device="cuda")
        n = torch.arange(N, dtype=int, device="cuda")
        attn_mask = (m[:, None] - n[None, :]).to(dtype)
    elif attn_type == "head_bias":
        h = torch.arange(Hq, dtype=int, device="cuda")
        attn_mask = (2 * h[None, :, None, None]).broadcast_to(1, Hq, M, N).to(dtype)
    elif attn_type == "alibi":
        h = torch.arange(Hq, dtype=torch.float32, device="cuda")
        scale = torch.exp2(-((h + 1) * 8.0 / Hq))
        q = torch.arange(M, dtype=int, device="cuda")[None, None, :, None]
        kv = torch.arange(N, dtype=int, device="cuda")[None, None, None, :]
        causal_mask = torch.ones(M, N, dtype=torch.bool, device="cuda").tril(diagonal=0)
        attn_mask = torch.zeros(1, Hq, M, N, dtype=torch.float32, device="cuda")
        attn_mask.masked_fill_(causal_mask.logical_not(), float("-inf"))
        attn_mask += (q - kv) * scale[None, :, None, None]

        attn_mask = attn_mask.to(dtype)
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
        "document_mask": partial(
            F.scaled_dot_product_attention, is_causal=True, enable_gqa=(Hq != Hkv)
        )
        if Hq == Hkv
        else None,
        "prefix_lm": partial(
            F.scaled_dot_product_attention, is_causal=False, enable_gqa=(Hq != Hkv)
        ),
        "softcap": None,
        "transfusion": None,
    }

    if is_decoding:
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
    batch_sizes: List[int],
    num_heads: List[Tuple[int, int]],
    seq_lens: List[int],
    head_dims: List[int],
    score_mods_str: List[str],
    decoding: bool,
    kv_cache_size: List[int],
    cal_bandwidth: bool,
) -> List[ExperimentConfig]:
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
            )
        )

    return all_configs


def main(args):
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    results = []
    for config in tqdm(
        generate_experiment_configs(
            args.calculate_bwd,
            args.dtype,
            args.b,
            args.nh,
            args.s,
            args.d,
            args.mods,
            args.decoding,
            args.kv_cache_size,
            args.throughput,
        )
    ):
        results.append(
            Experiment(
                config,
                run_single_experiment(
                    config,
                    dynamic=args.dynamic,
                    max_autotune=args.max_autotune,
                    run_FA=args.flash_attn,
                    og_eager=args.og_eager,
                ),
            )
        )

    print_results(results, args.flash_attn, args.save_path)


def heads_input_type(s):
    try:
        hq, hkv = map(int, s.split(","))
        return hq, hkv
    except Exception as e:
        raise argparse.ArgumentTypeError("Heads must be Hq,Hkv") from e


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Run sweep over sizes and score mods for flex attention"
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Runs a dynamic shapes version of compiled flex attention.",
    )
    parser.add_argument(
        "--calculate-bwd", action="store_true", help="Calculate backward pass times"
    )

    parser.add_argument("-dtype", type=str, help="dtype", default="bfloat16")

    parser.add_argument(
        "-b", type=int, nargs="+", help="batch sizes", default=[2, 8, 16]
    )
    parser.add_argument(
        "-nh",
        type=heads_input_type,
        nargs="+",
        help="# of q-heads,kv-heads",
        default=[(16, 16), (16, 2)],
    )
    parser.add_argument(
        "-s", type=int, nargs="+", help="sequence lengths", default=[512, 1024, 4096]
    )
    parser.add_argument("-d", type=int, nargs="+", help="head dims", default=[64, 128])
    parser.add_argument(
        "-mods",
        type=str,
        nargs="+",
        help="score mods: noop, causal, rel, head_bias, alibi, sliding_window, document_mask, prefix_ml, softcap",
        default=["noop", "causal", "rel", "head_bias"],
    )
    parser.add_argument(
        "--max-autotune", action="store_true", help="Turn on max-autotune"
    )
    parser.add_argument(
        "--decoding",
        action="store_true",
        help="Benchmark Decoding (query sequence length = 1)",
    )
    parser.add_argument(
        "--kv-cache-size",
        type=int,
        nargs="+",
        required=False,
        help="""
key/value cache size in MiB.
Ignores -b batch size and calculate batch size from kv_cache size instead when specified.
""",
    )
    parser.add_argument(
        "--throughput",
        action="store_true",
        help="Calculate kernel memory bandwidth & computational throughput. ",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="Path to save the results JSON file (optional)",
        default=None,
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        help="Run Flash Attention (depends on package flash_attn)",
    )
    parser.add_argument(
        "--og-eager",
        action="store_true",
        help="Compare with origin eager implementation from original paper proposing the attention variant. ",
    )
    # Parse arguments
    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)

    main(args)
