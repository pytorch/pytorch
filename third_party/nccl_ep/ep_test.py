#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information.

"""Python EP test — replicates ep_test.cu using the nccl.ep Pythonic API.

Usage:
    mpirun -np <N> python ep_test.py [OPTIONS]

Options (identical to ep_test.cu):
    -a {ll,ht}                          Algorithm mode (default: ll)
    -m                                   Disable max_dispatch_tokens_per_rank (HT only, not yet supported)
    -s {none,dispatch,combine,both}      Send-only mode (default: none)
    -c                                   Enable cached mode (HT only)
    -r                                   Enable random mode
    -t NUM                               Number of tokens (default: 50)
    -d NUM                               Hidden dimension size (default: 7168)
"""

from __future__ import annotations

import argparse
import ctypes
import random
import struct
import sys

import numpy as np
from cuda.bindings import runtime as cudart
from cuda.core import Device
from mpi4py import MPI

import nccl.core as nccl_core
import nccl.ep as nccl_ep


# ---------------------------------------------------------------------------
# Custom allocator callbacks. These wrap cudaMalloc/cudaFree — functionally
# equivalent to using the default allocator path (NCCL EP falls back to
# cudaMalloc/cudaFree when alloc_fn is NULL), but exercise the pluggable
# allocator hooks. The decorated functions MUST stay alive at module scope for
# the lifetime of any nccl_ep.Group that referenced them; if GC'd, NCCL EP's stored
# function pointers become dangling.
# ---------------------------------------------------------------------------

@nccl_ep.AllocFn
def _alloc_fn(out_ptr, size, _context):
    err, ptr = cudart.cudaMalloc(size)
    out_ptr[0] = ctypes.c_void_p(int(ptr))
    return int(err)


@nccl_ep.FreeFn
def _free_fn(ptr, _context):
    err, = cudart.cudaFree(ptr)
    return int(err)


_ALLOC_FN_ADDR = ctypes.cast(_alloc_fn, ctypes.c_void_p).value
_FREE_FN_ADDR  = ctypes.cast(_free_fn,  ctypes.c_void_p).value


# ---------------------------------------------------------------------------
# Host <-> Device transfers via cudaMemcpyAsync against raw device pointers.
# ---------------------------------------------------------------------------

_H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
_D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
_D2D = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice


def _check_cuda(err) -> None:
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")


def h2d(dev_ptr: int, src_arr: np.ndarray, stream) -> None:
    err, = cudart.cudaMemcpyAsync(
        dev_ptr, src_arr.ctypes.data, src_arr.nbytes, _H2D, int(stream.handle),
    )
    _check_cuda(err)


def d2h(dst_arr: np.ndarray, dev_ptr: int, stream) -> None:
    err, = cudart.cudaMemcpyAsync(
        dst_arr.ctypes.data, dev_ptr, dst_arr.nbytes, _D2H, int(stream.handle),
    )
    _check_cuda(err)


def d2d(dst_ptr: int, src_ptr: int, nbytes: int, stream) -> None:
    err, = cudart.cudaMemcpyAsync(dst_ptr, src_ptr, nbytes, _D2D, int(stream.handle))
    _check_cuda(err)


# ---------------------------------------------------------------------------
# Device tensor helper: pairs a raw cudaMalloc allocation with its nccl_ep.NDTensor.
# Sized at create-time so the host side can compute h2d/d2h byte counts
# without re-deriving from sizes.
# ---------------------------------------------------------------------------

_DTYPE_BYTES = {
    nccl_core.INT8: 1,
    nccl_core.UINT8: 1,
    nccl_core.INT32: 4,
    nccl_core.UINT32: 4,
    nccl_core.INT64: 8,
    nccl_core.UINT64: 8,
    nccl_core.FLOAT16: 2,
    nccl_core.BFLOAT16: 2,
    nccl_core.FLOAT32: 4,
}


class DevTensor:
    """Owning pair of a cudaMalloc'd device buffer and its ``nccl_ep.Tensor``."""

    def __init__(self, ndim: int, datatype, *sizes: int) -> None:
        nbytes = _DTYPE_BYTES[datatype]
        for s in sizes:
            nbytes *= s
        err, dev_ptr = cudart.cudaMalloc(nbytes)
        _check_cuda(err)
        self.data = int(dev_ptr)
        self.nbytes = nbytes
        self.tensor: nccl_ep.Tensor | None = nccl_ep.Tensor(
            self.data, dtype=int(datatype), shape=sizes,
        )

    def destroy(self) -> None:
        if self.tensor is not None:
            self.tensor = None  # release the Python-owned descriptor
        if self.data:
            cudart.cudaFree(self.data)
            self.data = 0


def make_tensor(ndim: int, datatype, *sizes: int) -> DevTensor:
    return DevTensor(ndim, datatype, *sizes)


def free_tensor(t: DevTensor | None) -> None:
    if t is not None:
        t.destroy()


# ---------------------------------------------------------------------------
# bfloat16 conversion (matches the C++ float_to_bf16 exactly)
# ---------------------------------------------------------------------------

def float_to_bf16(f: float) -> int:
    x = struct.unpack("I", struct.pack("f", f))[0]
    rounding_bias = 0x00007FFF + ((x >> 16) & 1)
    return ((x + rounding_bias) >> 16) & 0xFFFF


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def main():  # noqa: C901 — kept as a single function to mirror ep_test.cu
    mpi_comm = MPI.COMM_WORLD
    my_rank = mpi_comm.Get_rank()
    n_ranks = mpi_comm.Get_size()

    parser = argparse.ArgumentParser(description="EP Test (Python)")
    parser.add_argument("-a", choices=["ll", "ht"], default="ll", help="Algorithm mode")
    parser.add_argument("-m", action="store_true", help="Disable max_dispatch_tokens_per_rank (HT only)")
    parser.add_argument("-s", choices=["none", "dispatch", "combine", "both"], default="none",
                        help="Send-only mode")
    parser.add_argument("-c", action="store_true", help="Enable cached mode (HT only)")
    parser.add_argument("-r", action="store_true", help="Enable random mode")
    parser.add_argument("-t", type=int, default=50, help="Number of tokens")
    parser.add_argument("-d", type=int, default=7168, help="Hidden dimension size")
    args = parser.parse_args()

    algorithm = nccl_ep.Algorithm.LOW_LATENCY if args.a == "ll" else nccl_ep.Algorithm.HIGH_THROUGHPUT
    disable_max_tokens = args.m
    dispatch_send_only = 1 if args.s in ("dispatch", "both") else 0
    combine_send_only = 1 if args.s in ("combine", "both") else 0
    cached_mode = args.c
    random_mode = args.r
    num_tokens = args.t
    hidden = args.d

    if n_ranks not in (2, 4) and n_ranks % 8 != 0:
        if my_rank == 0:
            print("Error: nRanks must be 2, 4 or multiple of 8 for this test")
        sys.exit(1)

    if disable_max_tokens:
        if my_rank == 0:
            if algorithm != nccl_ep.Algorithm.HIGH_THROUGHPUT:
                print("Error: -m is only applicable to HT mode (-a ht)")
            else:
                print("Error: -m (NCCL_EP_AUTO for max_dispatch_tokens_per_rank) is not yet supported.\n"
                      "       This feature will be available in a future release for HT mode.")
        sys.exit(1)

    ELEMENTS_TESTED_PER_TOKEN = 10
    top_k = min(8, n_ranks)
    num_experts = min(256, top_k * n_ranks)
    num_local_experts = num_experts // n_ranks
    local_experts_start = num_local_experts * my_rank
    local_experts_end = local_experts_start + num_local_experts

    if num_experts % n_ranks != 0:
        if my_rank == 0:
            print(f"Error: num_experts ({num_experts}) must be divisible by nRanks ({n_ranks})")
        sys.exit(1)
    if top_k > num_local_experts:
        if my_rank == 0:
            print(f"Error: top_k ({top_k}) must be <= num_local_experts ({num_local_experts})")
        sys.exit(1)

    # Local rank = rank within the per-node sub-communicator.
    local_comm = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local_comm.Get_rank()

    device = Device(local_rank)
    device.set_current()
    stream = device.create_stream()

    # NCCL communicator: rank 0 generates a unique ID, MPI broadcasts it,
    # then every rank initializes its own Communicator.
    unique_id = nccl_core.get_unique_id() if my_rank == 0 else None
    unique_id = mpi_comm.bcast(unique_id, root=0)
    comm = nccl_core.Communicator.init(nranks=n_ranks, rank=my_rank, unique_id=unique_id)

    # -- EP group -----------------------------------------------------------
    # max_recv_tokens_per_rank is required for HT (assertion fires on 0);
    # LL auto-derives nRanks * max_dispatch_tokens_per_rank when left at 0, but we
    # set it explicitly to keep both paths consistent.
    config = nccl_ep.GroupConfig(
        algorithm=algorithm,
        num_experts=num_experts,
        max_dispatch_tokens_per_rank=num_tokens,
        max_recv_tokens_per_rank=num_tokens * n_ranks,
        max_token_bytes=hidden * 2,  # bfloat16
        alloc=nccl_ep.AllocConfig(alloc_fn=_ALLOC_FN_ADDR, free_fn=_FREE_FN_ADDR),
    )

    algorithm_name = "LOW_LATENCY" if algorithm == nccl_ep.Algorithm.LOW_LATENCY else "HIGH_THROUGHPUT"
    extra = " (no max_dispatch_tokens_per_rank)" if disable_max_tokens else ""
    print(f"Rank {my_rank}: Testing ncclEpCreateGroup with algorithm: {algorithm_name}{extra}")

    ep_group = nccl_ep.Group.create(comm, config)

    # -- topk_idx tensor [num_tokens, top_k] int64 --------------------------
    topk_idx = make_tensor(2, nccl_core.INT64, num_tokens, top_k)

    if random_mode:
        random.seed(my_rank + 42)
        first_experts = np.array(
            [random.randint(0, num_experts - 1) for _ in range(num_tokens)],
            dtype=np.int64,
        )
        # Each row: [first, first+1, ..., first+top_k-1] modulo num_experts.
        offsets = np.arange(top_k, dtype=np.int64)
        topk_idx_host = ((first_experts[:, None] + offsets) % num_experts).astype(np.int64)
        if my_rank == 0:
            print("Random mode enabled: first expert random, rest deterministic (no repetitions)")
    else:
        topk_idx_host = np.empty((num_tokens, top_k), dtype=np.int64)
        for i in range(num_tokens):
            for j in range(top_k):
                topk_idx_host[i, j] = (local_experts_end + j) % num_experts

    h2d(topk_idx.data, topk_idx_host, stream)

    # -- recv_expert_counter for handle (only when disable_max_tokens) ------
    handle_recv_expert_counter: DevTensor | None = None
    handle_recv_total_counter: DevTensor | None = None
    handle_layout_info: nccl_ep.LayoutInfo | None = None
    if disable_max_tokens:
        handle_recv_expert_counter = make_tensor(1, nccl_core.INT32, num_local_experts)
        handle_recv_total_counter = make_tensor(1, nccl_core.INT32, 1)
        handle_layout_info = nccl_ep.LayoutInfo(
            expert_counters=handle_recv_expert_counter.tensor,
            recv_total_counter=handle_recv_total_counter.tensor,
        )

    # -- EP handle ----------------------------------------------------------
    print(f"Rank {my_rank}: Testing ncclEpCreateHandle")
    # LL branch below builds a 3D recv tensor and only fills expert_counters
    # in dispatch_layout — that's the EXPERT_MAJOR layout's signature. HT
    # branch builds a 2D recv tensor with topk_weights/topk_idx, matching FLAT.
    handle_layout = (
        nccl_ep.Layout.EXPERT_MAJOR
        if algorithm == nccl_ep.Algorithm.LOW_LATENCY
        else nccl_ep.Layout.FLAT
    )
    ep_handle = ep_group.create_handle(
        handle_layout, topk_idx.tensor,
        layout_info=handle_layout_info,
        config=nccl_ep.HandleConfig(),
        stream=stream,
    )
    stream.sync()

    if disable_max_tokens:
        total_host = np.zeros(1, dtype=np.int32)
        d2h(total_host, handle_recv_total_counter.data, stream)
        stream.sync()
        num_recv_tokens = int(total_host[0])
    else:
        num_recv_tokens = config.max_dispatch_tokens_per_rank * num_local_experts
    assert num_recv_tokens > 0

    dispatch_config = nccl_ep.DispatchConfig(send_only=dispatch_send_only, round_scales=0)

    is_ll = algorithm == nccl_ep.Algorithm.LOW_LATENCY

    # -- input/output tensors for dispatch ----------------------------------
    input_tokens = make_tensor(2, nccl_core.BFLOAT16, num_tokens, hidden)
    topk_weights = make_tensor(2, nccl_core.FLOAT32, num_tokens, top_k)

    if is_ll:
        output_tokens = make_tensor(
            3, nccl_core.BFLOAT16,
            num_local_experts, config.max_dispatch_tokens_per_rank * n_ranks, hidden,
        )
    else:
        output_tokens = make_tensor(2, nccl_core.BFLOAT16, num_recv_tokens, hidden)

    local_tensor_recv_count: DevTensor | None = None
    if is_ll:
        local_tensor_recv_count = make_tensor(1, nccl_core.INT32, num_local_experts)

    output_topk_weights: DevTensor | None = None
    output_topk_idx: DevTensor | None = None
    if not is_ll:
        output_topk_weights = make_tensor(2, nccl_core.FLOAT32, num_recv_tokens, top_k)
        output_topk_idx = make_tensor(2, nccl_core.INT64, num_recv_tokens, top_k)

    # Fill input tokens: first ELEMENTS_TESTED_PER_TOKEN values = 0x1000 + my_rank
    input_host = np.zeros((num_tokens, hidden), dtype=np.uint16)
    input_host[:, :ELEMENTS_TESTED_PER_TOKEN] = 0x1000 + my_rank
    h2d(input_tokens.data, input_host, stream)

    # Fill topk_weights: 1.0 / top_k for every entry.
    tw_host = np.full(num_tokens * top_k, 1.0 / top_k, dtype=np.float32)
    h2d(topk_weights.data, tw_host, stream)

    # Build the named-struct ABI bundles for dispatch.
    if is_ll:
        dispatch_inputs = nccl_ep.DispatchInputs(tokens=input_tokens.tensor)
        dispatch_outputs = nccl_ep.DispatchOutputs(tokens=output_tokens.tensor)
        dispatch_layout = nccl_ep.LayoutInfo(
            expert_counters=local_tensor_recv_count.tensor,
        )
    else:
        dispatch_inputs = nccl_ep.DispatchInputs(
            tokens=input_tokens.tensor,
            topk_weights=topk_weights.tensor,
        )
        dispatch_outputs = nccl_ep.DispatchOutputs(
            tokens=output_tokens.tensor,
            topk_weights=output_topk_weights.tensor,
            topk_idx=output_topk_idx.tensor,
        )
        dispatch_layout = None

    print(f"Rank {my_rank}: Testing dispatch (send_only={bool(dispatch_send_only)})")
    ep_handle.dispatch(
        dispatch_inputs, dispatch_outputs,
        layout_info=dispatch_layout,
        config=dispatch_config,
        stream=stream,
    )

    print(f"Rank {my_rank}: Testing complete (after dispatch)")
    ep_handle.complete(stream=stream)
    stream.sync()

    # Read recv_count for verification.
    recv_count_host: np.ndarray | None = None
    if is_ll:
        recv_count_host = np.empty(num_local_experts, dtype=np.int32)
        d2h(recv_count_host, local_tensor_recv_count.data, stream)
        stream.sync()
    elif disable_max_tokens and handle_recv_expert_counter is not None:
        recv_count_host = np.empty(num_local_experts, dtype=np.int32)
        d2h(recv_count_host, handle_recv_expert_counter.data, stream)
        stream.sync()

    recv_from_expert_start = (local_experts_start + num_experts - num_local_experts) % num_experts
    recv_rank = recv_from_expert_start // num_local_experts

    # Verify dispatch output (deterministic mode only).
    dispatch_check_passed = True

    if not random_mode and is_ll and recv_count_host is not None:
        total_elems = num_local_experts * config.max_dispatch_tokens_per_rank * n_ranks * hidden
        output_host = np.empty(total_elems, dtype=np.uint16)
        d2h(output_host, output_tokens.data, stream)
        stream.sync()

        max_t = config.max_dispatch_tokens_per_rank * n_ranks
        for e in range(num_local_experts):
            if recv_count_host[e] != num_tokens:
                print(f"Recv_count check failed! Rank {my_rank}, expert {e}: "
                      f"expected {num_tokens}, got {int(recv_count_host[e])}")
                dispatch_check_passed = False
                break
            for t in range(min(int(recv_count_host[e]), max_t)):
                token_off = (e * max_t + t) * hidden
                for j in range(ELEMENTS_TESTED_PER_TOKEN):
                    expected = 0x1000 + recv_rank
                    actual = int(output_host[token_off + j])
                    if actual != expected:
                        print(f"Dispatch data check failed! Rank {my_rank}, expert {e}, "
                              f"token {t}, element {j}: expected {expected}, got {actual}")
                        dispatch_check_passed = False
                        break
                if not dispatch_check_passed:
                    break
            if not dispatch_check_passed:
                break

    elif not random_mode:
        if recv_count_host is not None:
            for e in range(num_local_experts):
                if recv_count_host[e] != num_tokens:
                    print(f"Recv_count check failed! Rank {my_rank}, expert {e}: "
                          f"expected {num_tokens}, got {int(recv_count_host[e])}")
                    dispatch_check_passed = False
                    break

        output_host = np.empty(num_recv_tokens * hidden, dtype=np.uint16)
        d2h(output_host, output_tokens.data, stream)
        stream.sync()
        check_count = num_recv_tokens if disable_max_tokens else num_tokens
        for i in range(check_count):
            for j in range(ELEMENTS_TESTED_PER_TOKEN):
                expected = 0x1000 + recv_rank
                actual = int(output_host[i * hidden + j])
                if actual != expected:
                    print(f"Dispatch check failed! Rank {my_rank}, token {i}, "
                          f"element {j}: expected {expected}, got {actual}")
                    dispatch_check_passed = False
                    break
            if not dispatch_check_passed:
                break

    # Verify HT recv_topk_weights / recv_topk_idx.
    if not random_mode and not is_ll:
        recv_tw = np.empty(num_recv_tokens * top_k, dtype=np.float32)
        d2h(recv_tw, output_topk_weights.data, stream)
        recv_ti = np.empty(num_recv_tokens * top_k, dtype=np.int64)
        d2h(recv_ti, output_topk_idx.data, stream)
        stream.sync()

        # Only the first num_tokens rows are meaningful for the per-rank check.
        expected_weight = np.float32(1.0 / top_k)
        window_tw = recv_tw[:num_tokens * top_k]
        window_ti = recv_ti[:num_tokens * top_k]
        w_bad = window_tw != expected_weight
        i_bad = (window_ti < 0) | (window_ti >= num_experts)
        weight_errors = int(w_bad.sum())
        idx_errors = int(i_bad.sum())

        for off in np.flatnonzero(w_bad)[:5]:
            i, j = int(off) // top_k, int(off) % top_k
            print(f"Rank {my_rank}: recv_topk_weights[{i}][{j}] = {window_tw[off]}, "
                  f"expected {expected_weight}")
        for off in np.flatnonzero(i_bad)[:5]:
            i, j = int(off) // top_k, int(off) % top_k
            print(f"Rank {my_rank}: recv_topk_idx[{i}][{j}] = {int(window_ti[off])}, "
                  f"expected range [0, {num_experts})")

        if weight_errors:
            print(f"Rank {my_rank}: recv_topk_weights verification failed with {weight_errors} errors")
        if idx_errors:
            print(f"Rank {my_rank}: recv_topk_idx verification failed with {idx_errors} errors")
        if weight_errors == 0 and idx_errors == 0:
            print(f"Rank {my_rank}: {algorithm_name} recv_topk_weights / recv_topk_idx verification passed")
        else:
            dispatch_check_passed = False

    if random_mode:
        print(f"Rank {my_rank}: {algorithm_name} Dispatch flow completed (random mode, checks skipped)")
    elif dispatch_check_passed:
        print(f"Rank {my_rank}: {algorithm_name} Dispatch flow passed successfully")
    else:
        print(f"Rank {my_rank}: Exiting test due to dispatch failure")
        sys.exit(1)

    # ===================================================================
    # Combine
    # ===================================================================
    print(f"Rank {my_rank}: Testing {algorithm_name} Combine flow")

    if is_ll:
        expert_outputs = make_tensor(
            3, nccl_core.BFLOAT16,
            num_local_experts, config.max_dispatch_tokens_per_rank * n_ranks, hidden,
        )
        eo_host = np.zeros(config.max_dispatch_tokens_per_rank * hidden, dtype=np.uint16)
        for t in range(config.max_dispatch_tokens_per_rank):
            for j in range(ELEMENTS_TESTED_PER_TOKEN):
                eo_host[t * hidden + j] = float_to_bf16(float((j + 1) * 2))
        stride_bytes = config.max_dispatch_tokens_per_rank * hidden * n_ranks * 2
        for e in range(num_local_experts):
            h2d(expert_outputs.data + e * stride_bytes, eo_host, stream)
    else:
        expert_outputs = make_tensor(2, nccl_core.BFLOAT16, num_recv_tokens, hidden)
        eo_host = np.zeros(num_recv_tokens * hidden, dtype=np.uint16)
        for t in range(num_recv_tokens):
            for j in range(ELEMENTS_TESTED_PER_TOKEN):
                eo_host[t * hidden + j] = float_to_bf16(float((j + 1) * 2))
        h2d(expert_outputs.data, eo_host, stream)

    combined_output = make_tensor(2, nccl_core.BFLOAT16, num_tokens, hidden)

    if is_ll:
        combine_inputs = nccl_ep.CombineInputs(tokens=expert_outputs.tensor)
        combine_outputs = nccl_ep.CombineOutputs(
            tokens=combined_output.tensor,
            topk_weights=topk_weights.tensor,  # per-token routing weights on receive side
        )
    else:
        combine_inputs = nccl_ep.CombineInputs(tokens=expert_outputs.tensor)
        combine_outputs = nccl_ep.CombineOutputs(tokens=combined_output.tensor)

    print(f"Rank {my_rank}: Testing combine (send_only={bool(combine_send_only)})")
    ep_handle.combine(
        combine_inputs, combine_outputs,
        config=nccl_ep.CombineConfig(send_only=combine_send_only),
        stream=stream,
    )

    print(f"Rank {my_rank}: Testing complete (after combine)")
    ep_handle.complete(stream=stream)
    stream.sync()

    # Verify combine output.
    combine_errors = 0
    if not random_mode:
        co_host = np.empty(num_tokens * hidden, dtype=np.uint16)
        d2h(co_host, combined_output.data, stream)
        stream.sync()
        for i in range(num_tokens):
            for j in range(ELEMENTS_TESTED_PER_TOKEN):
                expected = float_to_bf16(float((j + 1) * 2))
                actual = int(co_host[i * hidden + j])
                if actual != expected:
                    print(f"Combine check failed! Rank {my_rank}, token {i}, "
                          f"element {j}: expected {expected}, got {actual}")
                    combine_errors += 1
                    if combine_errors >= 5:
                        break
            if combine_errors >= 5:
                break

    if random_mode:
        print(f"Rank {my_rank}: Combine flow completed (random mode, checks skipped)")
    elif combine_errors == 0:
        print(f"Rank {my_rank}: Combine verification PASSED! "
              f"All {num_tokens} tokens with {hidden} elements each correctly combined")
    else:
        print(f"Rank {my_rank}: Combine verification FAILED with {combine_errors} errors")
        sys.exit(1)

    # ===================================================================
    # Cached mode (HT only): repeat dispatch+combine and compare outputs.
    # ===================================================================
    cached_tensors: list[DevTensor] = []
    if cached_mode:
        if is_ll:
            print(f"Rank {my_rank}: Error - cached mode is only supported in HT modes (not LL)")
            sys.exit(1)

        print(f"Rank {my_rank}: Testing cached mode ({algorithm_name})")

        # save first-phase outputs
        first_d0 = first_co = None
        if not random_mode:
            first_d0 = np.empty(num_recv_tokens * hidden, dtype=np.uint16)
            d2h(first_d0, output_tokens.data, stream)

            first_co = np.empty(num_tokens * hidden, dtype=np.uint16)
            d2h(first_co, combined_output.data, stream)
            stream.sync()

        # New output tensors for the second dispatch / combine.
        cached_out_tokens = make_tensor(2, nccl_core.BFLOAT16, num_recv_tokens, hidden)
        cached_combined_output = make_tensor(2, nccl_core.BFLOAT16, num_tokens, hidden)
        cached_combined_tw = make_tensor(2, nccl_core.FLOAT32, num_tokens, top_k)
        cached_tensors.extend([cached_out_tokens, cached_combined_output, cached_combined_tw])

        # topk_weights input for combine (copy from dispatch output).
        cached_ctw_in = make_tensor(2, nccl_core.FLOAT32, num_recv_tokens, top_k)
        cached_tensors.append(cached_ctw_in)
        d2d(cached_ctw_in.data, output_topk_weights.data,
            num_recv_tokens * top_k * 4, stream)

        print(f"Rank {my_rank}: Testing cached mode - second dispatch "
              f"(send_only={bool(dispatch_send_only)})")
        ep_handle.dispatch(
            nccl_ep.DispatchInputs(tokens=input_tokens.tensor),
            nccl_ep.DispatchOutputs(tokens=cached_out_tokens.tensor),
            config=dispatch_config,
            stream=stream,
        )

        print(f"Rank {my_rank}: Testing cached mode - second complete (dispatch)")
        ep_handle.complete(stream=stream)
        stream.sync()

        print(f"Rank {my_rank}: Testing cached mode - second combine "
              f"(send_only={bool(combine_send_only)})")
        ep_handle.combine(
            nccl_ep.CombineInputs(
                tokens=expert_outputs.tensor,
                topk_weights=cached_ctw_in.tensor,
            ),
            nccl_ep.CombineOutputs(
                tokens=cached_combined_output.tensor,
                topk_weights=cached_combined_tw.tensor,
            ),
            config=nccl_ep.CombineConfig(send_only=combine_send_only),
            stream=stream,
        )

        print(f"Rank {my_rank}: Testing cached mode - second complete (combine)")
        ep_handle.complete(stream=stream)
        stream.sync()

        # Compare first vs second phase.
        cached_dispatch_errors = 0
        cached_combine_errors = 0

        if not random_mode:
            sec_d0 = np.empty(num_recv_tokens * hidden, dtype=np.uint16)
            d2h(sec_d0, cached_out_tokens.data, stream)

            sec_co = np.empty(num_tokens * hidden, dtype=np.uint16)
            d2h(sec_co, cached_combined_output.data, stream)

            sec_tw = np.empty(num_tokens * top_k, dtype=np.float32)
            d2h(sec_tw, cached_combined_tw.data, stream)
            stream.sync()

            d0_diff = first_d0 != sec_d0
            co_diff = first_co != sec_co
            tw_bad = sec_tw != np.float32(1.0 / top_k)

            cached_dispatch_errors = int(d0_diff.sum())
            cached_combine_errors = int(co_diff.sum()) + int(tw_bad.sum())

            for off in np.flatnonzero(d0_diff)[:5]:
                print(f"Rank {my_rank}: Cached dispatch output mismatch at {int(off)}: "
                      f"first={int(first_d0[off])}, second={int(sec_d0[off])}")
            for off in np.flatnonzero(co_diff)[:5]:
                print(f"Rank {my_rank}: Cached combine output mismatch at {int(off)}: "
                      f"first={int(first_co[off])}, second={int(sec_co[off])}")
            for off in np.flatnonzero(tw_bad)[:5]:
                print(f"Rank {my_rank}: Cached combine topk_weights mismatch at {int(off)}: "
                      f"expected={1.0/top_k}, got={float(sec_tw[off])}")

        if random_mode:
            print(f"Rank {my_rank}: Cached mode completed (random mode, checks skipped)")
        elif cached_dispatch_errors == 0 and cached_combine_errors == 0:
            print(f"Rank {my_rank}: Cached mode verification PASSED")
        else:
            print(f"Rank {my_rank}: Cached mode verification FAILED - "
                  f"dispatch errors: {cached_dispatch_errors}, combine errors: {cached_combine_errors}")
            sys.exit(1)

    # ===================================================================
    # Cleanup
    # ===================================================================
    for t in cached_tensors:
        free_tensor(t)
    free_tensor(expert_outputs)
    free_tensor(topk_weights)
    free_tensor(combined_output)
    free_tensor(topk_idx)
    if disable_max_tokens:
        free_tensor(handle_recv_expert_counter)
        free_tensor(handle_recv_total_counter)
    free_tensor(input_tokens)
    free_tensor(output_tokens)
    if not is_ll:
        free_tensor(output_topk_weights)
        free_tensor(output_topk_idx)
    if is_ll:
        free_tensor(local_tensor_recv_count)

    ep_handle.destroy()
    ep_group.destroy()
    comm.destroy()
    print(f"[MPI Rank {my_rank}] Success ")


if __name__ == "__main__":
    main()
