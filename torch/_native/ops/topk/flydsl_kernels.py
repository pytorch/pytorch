"""FlyDSL fp32 top-K kernels, picked by K and N in ``flydsl_impl.py``.

``topk_radix`` - one CTA per row, ``max(sort_len, _N_HIST_BINS)``
threads (``sort_len`` = K rounded up to a power of 2; the bin floor lets
phase 1 zero the 256 histogram bins one per thread).
  1. Four radix byte passes (MSB to LSB) over an LDS histogram; thread 0
     scans descending, narrowing ``s_prefix``/``s_mask`` until it holds
     the K-th largest ordinal and ``s_rem_k`` the ties to keep.
  2. Gather at-or-above that threshold. Deterministic: prefix-sum slots
     preserve input order, lowest-index ties win. Otherwise LDS atomic
     counters claim slots and ties resolve per run.
  3. Bitonic sort over ``sort_len`` slots; padding carries ordinal
     INT32_MIN, which no float maps to. Comparator is lex
     ``(ord desc, idx asc)`` when deterministic, ord-only otherwise.
  4. Write K (value, index) pairs.

``topk_register`` - one wave per row. Each lane bitonic-sorts its
``N / wave_size`` Int64 ``(ord << 32) | ~idx`` keys, then merges with
butterfly partners until lane 0 holds the top-K; only the final K writes
hit gmem.

``_f32_to_ord`` leaves positives alone and inverts all but the sign bit of
negatives, so ordinals compare as signed i32 in float order; NaN
canonicalises to INT32_MAX. Not injective on NaN, so ``decode_key``
re-reads gmem for that case. Phase 1's xor by ``_RADIX_SIGN_BIT`` is
separate: it puts the top byte in unsigned bin order for the scan.

``_make_topk_storage`` unions ``s_hist``, the phase-2 counters, ``s_scan``
and ``s_keys`` - they alias, each live only within its phase. Anything
spanning phases needs a struct member, not a union member.
"""

import math
from typing import Any, cast, Protocol

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, Array, Float32, gpu, Int32, Int64, range_constexpr
from flydsl.runtime.device import is_rdna_arch

import torch
from torch._native.flydsl.cache import CacheInfo
from torch._native.flydsl_utils import _resolve_rocm_arch
from torch._native.instrumentation import instrumented_flydsl_cache

from ._common import any_cow


class _CachedCompile(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    def cache_clear(self) -> None: ...

    def cache_info(self) -> CacheInfo: ...


_RADIX_BITS = 8
_RADIX_MASK = (1 << _RADIX_BITS) - 1
_RADIX_SIGN_BIT = 1 << (_RADIX_BITS - 1)
_NUM_RADIX_PASSES = 32 // _RADIX_BITS
_N_HIST_BINS = 1 << _RADIX_BITS
_VEC = 4


def _i32_const(x: int) -> int:
    return x - (1 << 32) if x >= (1 << 31) else x


def _f32_to_ord(val):
    bits = val.bitcast(Int32)
    ords = bits ^ ((bits >> fx.Int32(31)) & fx.Int32(0x7FFFFFFF))
    is_nan = (bits & fx.Int32(0x7FFFFFFF)) > fx.Int32(0x7F800000)
    return is_nan.select(fx.Int32(0x7FFFFFFF), ords)


def _make_key(val, idx):
    ord64 = fx.Int64(_f32_to_ord(val)) & fx.Int64(0xFFFFFFFF)
    inv_idx64 = fx.Int64(~idx) & fx.Int64(0xFFFFFFFF)
    return (ord64 << fx.Int64(32)) | inv_idx64


def _decode_key(key):
    ord32 = fx.Int32(key >> fx.Int64(32))
    inv_idx = fx.Int32(key & fx.Int64(0xFFFFFFFF))
    val_bits = ord32 ^ ((ord32 >> fx.Int32(31)) & fx.Int32(0x7FFFFFFF))
    val = val_bits.bitcast(Float32)
    idx = ~inv_idx
    return val, idx, ord32


def _make_topk_storage(sort_len: int, block_threads: int):
    @fx.struct
    class CounterStorage:
        s_write_ctr: Array[Int32, 1, 4]
        s_eq_ctr: Array[Int32, 1, 4]

    @fx.union
    class PhaseStorage:
        s_hist: Array[Int32, 256, 16]
        s_counters: CounterStorage
        s_scan: Array[Int32, block_threads + 1, 16]
        s_keys: Array[Int64, sort_len, 16]

    @fx.struct
    class SharedStorage:
        s_phase: PhaseStorage
        s_prefix: Array[Int32, 1, 4]
        s_mask: Array[Int32, 1, 4]
        s_rem_k: Array[Int32, 1, 4]
        s_vals: Array[Float32, sort_len, 16]
        s_idxs: Array[Int32, sort_len, 16]

    return SharedStorage


def _build_radix_select_topk_module(n: int, k: int, deterministic: bool, arch: str):
    num_stages = (k - 1).bit_length()
    sort_len = 1 << num_stages
    block_threads = max(sort_len, _N_HIST_BINS)
    warp_size = 32 if is_rdna_arch(arch) else 64
    num_warps = block_threads // warp_size
    tile = block_threads * _VEC
    vec_iters = n // tile
    vec_tail_start = vec_iters * tile
    scalar_tail_iters = (n - vec_tail_start + block_threads - 1) // block_threads

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def radix_select_topk_kernel(
        input: fx.Tensor,
        values: fx.Tensor,
        indices: fx.Tensor,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x

        input_buf = fx.rocdl.make_buffer_tensor(input)
        values_buf = fx.rocdl.make_buffer_tensor(values)
        indices_buf = fx.rocdl.make_buffer_tensor(indices)

        row_in = fx.slice(input_buf, (row, None))
        row_values = fx.slice(values_buf, (row, None))
        row_indices = fx.slice(indices_buf, (row, None))
        input_div = fx.logical_divide(row_in, fx.make_layout(_VEC, 1))
        copy_atom_v = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
        storage = fx.SharedAllocator().allocate(
            _make_topk_storage(sort_len, block_threads)
        )
        s_hist = storage.s_phase.s_hist.peek().view(fx.make_layout(_N_HIST_BINS, 1))
        s_prefix = storage.s_prefix.peek().view(fx.make_layout(1, 1))
        s_mask = storage.s_mask.peek().view(fx.make_layout(1, 1))
        s_rem_k = storage.s_rem_k.peek().view(fx.make_layout(1, 1))
        s_write_ctr = storage.s_phase.s_counters.s_write_ctr.peek().view(
            fx.make_layout(1, 1)
        )
        s_eq_ctr = storage.s_phase.s_counters.s_eq_ctr.peek().view(fx.make_layout(1, 1))
        s_scan = storage.s_phase.s_scan.peek().view(
            fx.make_layout(block_threads + 1, 1)
        )
        s_vals = storage.s_vals.peek().view(fx.make_layout(sort_len, 1))
        s_keys = storage.s_phase.s_keys.peek().view(fx.make_layout(sort_len, 1))
        s_idxs = storage.s_idxs.peek().view(fx.make_layout(sort_len, 1))

        def load_vec_f32(idx):
            r = fx.make_rmem_tensor(4, Float32)
            fx.copy_atom_call(copy_atom_v, fx.slice(input_div, (None, idx)), r)
            return fx.memref_load_vec(r)

        def decode_key(key):
            val, idx, ord32 = _decode_key(key)
            if ord32 == fx.Int32(0x7FFFFFFF):
                val = row_in[idx]
            return val, idx

        def atomic_add_i32_fetch(memref, val, offset):
            ptr = fx.to_llvm_ptr(fx.get_iter(memref) + offset)
            old = llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                ptr,
                arith.unwrap(val),
                llvm.AtomicOrdering.monotonic,
                syncscope="workgroup",
                alignment=4,
            ).result
            return fx.Int32(old)

        def warp_inclusive_prefix_i32(val, lane):
            for i in range_constexpr(int(math.log2(warp_size))):
                offset = 1 << i
                remote = gpu.shuffle_up(val, offset, warp_size)
                val = (lane >= fx.Int32(offset)).select(val + remote, val)
            return val

        def block_excl_prefix_i32(packed_local, scan):
            lane = tid % fx.Int32(warp_size)
            warp = tid // fx.Int32(warp_size)
            incl = warp_inclusive_prefix_i32(packed_local, lane)
            excl_intra = incl - packed_local
            if lane == fx.Int32(warp_size - 1):
                scan[warp] = incl
            gpu.barrier()

            if warp == 0:
                warp_val = 0
                if lane < fx.Int32(num_warps):
                    warp_val = scan[lane]
                warp_incl = warp_inclusive_prefix_i32(warp_val, lane)
                warp_excl = warp_incl - warp_val
                if lane < fx.Int32(num_warps):
                    scan[lane] = warp_excl
                if lane == fx.Int32(num_warps - 1):  # Store total in the extra slot.
                    scan[num_warps] = warp_incl
            gpu.barrier()

            packed_pfx = scan[warp] + excl_intra
            packed_total = scan[num_warps]
            gpu.barrier()
            return packed_pfx, packed_total

        if tid == 0:
            s_prefix[0] = 0
            s_mask[0] = 0
            s_rem_k[0] = k
        gpu.barrier()

        # Phase 1: radix byte passes, MSB to LSB.
        for byte_pos in range_constexpr(_NUM_RADIX_PASSES):
            shift = (_NUM_RADIX_PASSES - 1 - byte_pos) * _RADIX_BITS
            xor_val = _RADIX_SIGN_BIT if byte_pos == 0 else 0

            # Zero the 256-bin histogram.
            if tid < fx.Int32(_N_HIST_BINS):
                s_hist[tid] = 0
            gpu.barrier()

            # Accumulate the current byte histogram.
            prefix = s_prefix[0]
            decided_mask = s_mask[0]
            for step in range_constexpr(vec_iters):
                rvals = load_vec_f32(step * block_threads + tid)
                for vi in range_constexpr(_VEC):
                    ords = _f32_to_ord(rvals[vi])
                    if byte_pos == 0 or (ords & decided_mask) == prefix:
                        byte_val = (
                            (ords >> fx.Int32(shift)) & fx.Int32(_RADIX_MASK)
                        ) ^ fx.Int32(xor_val)
                        atomic_add_i32_fetch(s_hist, 1, byte_val)
            for step in range_constexpr(scalar_tail_iters):
                col = vec_tail_start + step * block_threads + tid
                if col < n:
                    ords = _f32_to_ord(row_in[col])
                    if byte_pos == 0 or (ords & decided_mask) == prefix:
                        byte_val = (
                            (ords >> fx.Int32(shift)) & fx.Int32(_RADIX_MASK)
                        ) ^ fx.Int32(xor_val)
                        atomic_add_i32_fetch(s_hist, 1, byte_val)
            gpu.barrier()

            # Select the radix threshold.
            if tid == 0:
                remaining_k = s_rem_k[0]
                acc = 0
                found = 0
                sel_bin = 0
                elems_above = 0

                for b in range_constexpr(_N_HIST_BINS):
                    bin_idx = _N_HIST_BINS - 1 - b
                    count = s_hist[bin_idx]
                    if found == 0:
                        if acc + count >= remaining_k:
                            sel_bin = fx.Int32(bin_idx)
                            elems_above = acc
                            found = 1
                    acc = acc + count

                actual_byte = sel_bin ^ fx.Int32(xor_val)
                s_prefix[0] = prefix | (actual_byte << fx.Int32(shift))
                s_mask[0] = decided_mask | fx.Int32(_i32_const(_RADIX_MASK << shift))
                s_rem_k[0] = remaining_k - elems_above
            gpu.barrier()

        # Phase 2: gather elements at-or-above the radix threshold.
        threshold = s_prefix[0]  # Values above threshold are in top-k.
        remaining_k = s_rem_k[0]  # Number of threshold ties to keep.

        if deterministic:
            n_above = fx.Int32(k) - remaining_k
            above_base = 0
            eq_base = 0
            for step in range_constexpr(vec_iters):
                tile_base = step * tile
                base = tile_base + tid * _VEC
                rvals = load_vec_f32(step * block_threads + tid)

                # Class encoding: 2 = above, 1 = eq, 0 = below.
                cls_reg = fx.make_rmem_tensor(_VEC, Int32)
                packed_local = 0
                for vi in range_constexpr(_VEC):
                    ords = _f32_to_ord(rvals[vi])
                    if ords > threshold:
                        cls_reg[vi] = 2
                        packed_local = packed_local + fx.Int32(1 << 16)
                    elif ords == threshold:
                        cls_reg[vi] = 1
                        packed_local = packed_local + 1
                    else:
                        cls_reg[vi] = 0

                packed_pfx, packed_step_total = block_excl_prefix_i32(
                    packed_local, s_scan
                )
                my_above = above_base + (packed_pfx >> fx.Int32(16))
                my_eq = eq_base + (packed_pfx & fx.Int32(0xFFFF))
                for vi in range_constexpr(_VEC):
                    val = rvals[vi]
                    cls = cls_reg[vi]
                    idx = fx.Int32(base + vi)
                    if cls == 2:
                        s_vals[my_above] = val
                        s_idxs[my_above] = idx
                        my_above = my_above + 1
                    elif cls == 1:
                        if my_eq < remaining_k:
                            pos = n_above + my_eq
                            s_vals[pos] = val
                            s_idxs[pos] = idx
                        my_eq = my_eq + 1

                above_base = above_base + (packed_step_total >> fx.Int32(16))
                next_eq_base = eq_base + (packed_step_total & fx.Int32(0xFFFF))
                eq_base = (next_eq_base < remaining_k).select(next_eq_base, remaining_k)

            # --- Scalar tail ---
            for step in range_constexpr(scalar_tail_iters):
                col = vec_tail_start + step * block_threads + tid
                valid = col < n
                packed_local = 0
                if valid:
                    ords = _f32_to_ord(row_in[col])
                    if ords > threshold:
                        packed_local = fx.Int32(1 << 16)
                    elif ords == threshold:
                        packed_local = 1

                packed_pfx, packed_step_total = block_excl_prefix_i32(
                    packed_local, s_scan
                )
                my_above = above_base + (packed_pfx >> fx.Int32(16))
                my_eq = eq_base + (packed_pfx & fx.Int32(0xFFFF))
                if valid:
                    idx = fx.Int32(col)
                    val = row_in[col]
                    ords = _f32_to_ord(val)
                    if ords > threshold:
                        s_vals[my_above] = val
                        s_idxs[my_above] = idx
                    elif ords == threshold:
                        if my_eq < remaining_k:
                            pos = n_above + my_eq
                            s_vals[pos] = val
                            s_idxs[pos] = idx

                above_base = above_base + (packed_step_total >> fx.Int32(16))
                next_eq_base = eq_base + (packed_step_total & fx.Int32(0xFFFF))
                eq_base = (next_eq_base < remaining_k).select(next_eq_base, remaining_k)
        else:
            if tid == 0:
                s_write_ctr[0] = 0
                s_eq_ctr[0] = 0
            gpu.barrier()

            def gather_candidate(val, idx, vals, idxs):
                ords = _f32_to_ord(val)
                if ords > threshold:
                    pos = atomic_add_i32_fetch(s_write_ctr, 1, 0)
                    vals[pos] = val
                    idxs[pos] = idx
                elif ords == threshold:
                    eq_pos = atomic_add_i32_fetch(s_eq_ctr, 1, 0)
                    if eq_pos < remaining_k:
                        pos = atomic_add_i32_fetch(s_write_ctr, 1, 0)
                        vals[pos] = val
                        idxs[pos] = idx

            for step in range_constexpr(vec_iters):
                tile_base = step * tile
                base = tile_base + tid * _VEC
                rvals = load_vec_f32(step * block_threads + tid)
                for vi in range_constexpr(_VEC):
                    idx = fx.Int32(base + vi)
                    gather_candidate(rvals[vi], idx, s_vals, s_idxs)
            for step in range_constexpr(scalar_tail_iters):
                col = vec_tail_start + step * block_threads + tid
                if col < n:
                    gather_candidate(row_in[col], fx.Int32(col), s_vals, s_idxs)

        gpu.barrier()

        # Phase 3: cooperative bitonic sort.
        active = tid < fx.Int32(k)
        sort_active = tid < fx.Int32(sort_len)
        sort_tid = sort_active.select(tid, 0)
        if sort_active:
            if active:
                s_keys[tid] = _make_key(s_vals[sort_tid], s_idxs[sort_tid])
            else:
                s_keys[tid] = fx.Int64(-(1 << 63))
        gpu.barrier()

        # The padded bitonic network builds ascending blocks first, then the
        # final stage merges everything into descending order.
        for stage in range_constexpr(num_stages):
            for sub_rev in range_constexpr(stage + 1):
                sub = stage - sub_rev
                step_size = 1 << sub
                raw_partner = tid ^ fx.Int32(step_size)
                partner_active = raw_partner < fx.Int32(sort_len)
                partner = (sort_active & partner_active).select(raw_partner, 0)
                my_key = s_keys[sort_tid]
                partner_key = s_keys[partner]
                gpu.barrier()

                self_lt_partner = my_key < partner_key
                self_gt_partner = my_key > partner_key
                block_dir = (tid >> fx.Int32(stage + 1)) & 1
                if stage == num_stages - 1:
                    block_dir = block_dir ^ 1
                if sort_active & partner_active:
                    if tid < partner:
                        if block_dir == 0:
                            if self_gt_partner:
                                s_keys[tid] = partner_key
                        else:
                            if self_lt_partner:
                                s_keys[tid] = partner_key
                    else:
                        if block_dir == 0:
                            if self_lt_partner:
                                s_keys[tid] = partner_key
                        else:
                            if self_gt_partner:
                                s_keys[tid] = partner_key
                gpu.barrier()

        # Phase 4: results to gmem.
        if tid < fx.Int32(k):
            val, idx = decode_key(s_keys[tid])
            row_values[tid] = val
            row_indices[tid] = fx.Int64(idx)

    @flyc.jit
    def launch_radix_select_topk(
        input: fx.Tensor,
        values: fx.Tensor,
        indices: fx.Tensor,
        rows_m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = radix_select_topk_kernel(input, values, indices)
        launcher.launch(
            grid=(rows_m, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_radix_select_topk


def _build_register_topk_module(n: int, k: int, arch: str, rows_per_cta: int = 2):
    threads_per_row = 32 if is_rdna_arch(arch) else 64
    block_threads = threads_per_row * rows_per_cta
    vec = n // threads_per_row
    # Process elements group by group to reduce VGPR usage and preserve occupancy.
    # Groups are K elements, widened to _VEC for 128-bit loads. N >= 1024 is a
    # power of two, with threads_per_row = 32 or 64.
    group = max(k, _VEC)
    if vec % group or group % _VEC:
        raise AssertionError(f"vec={vec} and group={group} must divide by _VEC")
    group_loads = group // _VEC
    num_groups = vec // group
    num_stages_group = int(math.log2(group))
    num_stages_k = int(math.log2(k)) if k > 1 else 0
    log2_threads_per_row = int(math.log2(threads_per_row))

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def register_topk_kernel(
        input: fx.Tensor,
        values: fx.Tensor,
        indices: fx.Tensor,
        rows_m: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % fx.Int32(threads_per_row)
        row_local = tid // fx.Int32(threads_per_row)
        row = bid * fx.Int32(rows_per_cta) + row_local
        row_safe = (row < rows_m).select(row, 0)
        in_bounds = row < rows_m

        input_buf = fx.rocdl.make_buffer_tensor(input)
        values_buf = fx.rocdl.make_buffer_tensor(values)
        indices_buf = fx.rocdl.make_buffer_tensor(indices)
        row_input = fx.slice(input_buf, (row_safe, None))
        row_values = fx.slice(values_buf, (row_safe, None))
        row_indices = fx.slice(indices_buf, (row_safe, None))
        input_div = fx.logical_divide(row_input, fx.make_layout(_VEC, 1))
        copy_atom_v = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)

        def load_vec_f32(idx):
            r = fx.make_rmem_tensor(_VEC, Float32)
            fx.copy_atom_call(copy_atom_v, fx.slice(input_div, (None, idx)), r)
            return fx.memref_load_vec(r)

        def decode_key(key):
            val, idx, ord32 = _decode_key(key)
            if ord32 == fx.Int32(0x7FFFFFFF):
                val = row_input[idx]
            return val, idx

        def compare_and_swap(arr, i: int, j: int, descending: bool):
            a = arr[i]
            b = arr[j]
            swap = a < b if descending else a > b
            arr[i] = swap.select(b, a)
            arr[j] = swap.select(a, b)

        def bitonic_sort_desc(arr, length: int, stages: int):
            if length > 1:
                for s in range_constexpr(stages):
                    for sub_rev in range_constexpr(s + 1):
                        step = 1 << (s - sub_rev)
                        for i in range_constexpr(length):
                            j = i ^ step
                            if j > i:
                                block_dir = (i >> (s + 1)) & 1
                                compare_and_swap(arr, i, j, block_dir == 0)

        def bitonic_merge_desc(arr, length: int, levels: int):
            if length > 1:
                for level in range_constexpr(levels):
                    merge_len = length >> level
                    step = merge_len // 2
                    for i in range_constexpr(length // merge_len):
                        start_i = i * merge_len
                        for j in range_constexpr(step):
                            compare_and_swap(arr, start_i + j, start_i + j + step, True)

        def topk_merge_desc(a, b):
            for i in range_constexpr(k):
                x = a[i]
                y = b[k - 1 - i]
                take_y = y > x
                a[i] = take_y.select(y, x)
            bitonic_merge_desc(a, k, num_stages_k)

        # Build the per-lane top-K one group at a time. A lane takes whole
        # 128-bit chunks and lanes stride the row by one chunk, so each load
        # step has the wave reading a contiguous _VEC * wave span. Which
        # columns a lane ends up owning is irrelevant: both the fold below and
        # the butterfly merge after it only ever keep the top-K of a union.
        topk = fx.make_rmem_tensor(k, Int64)
        buf = fx.make_rmem_tensor(group, Int64)
        for g in range_constexpr(num_groups):
            for j in range_constexpr(group_loads):
                chunk = fx.Int32((g * group_loads + j) * threads_per_row) + lane
                rvals = load_vec_f32(chunk)
                base = chunk * fx.Int32(_VEC)
                for e in range_constexpr(_VEC):
                    buf[j * _VEC + e] = _make_key(rvals[e], base + fx.Int32(e))
            bitonic_sort_desc(buf, group, num_stages_group)
            # buf is descending, so its leading K are the group's top-K and
            # topk_merge_desc reads exactly that prefix.
            if g == 0:
                for i in range_constexpr(k):
                    topk[i] = buf[i]
            else:
                topk_merge_desc(topk, buf)

        # Warp-cooperative top-K merge via butterfly shuffles.
        for s in range_constexpr(log2_threads_per_row):
            other = fx.make_rmem_tensor(k, Int64)
            for i in range_constexpr(k):
                peer = gpu.shuffle_xor(topk[i], 1 << s, threads_per_row)
                other[i] = peer
            topk_merge_desc(topk, other)

        if lane == 0 and in_bounds:
            for i in range_constexpr(k):
                val, idx = decode_key(topk[i])
                row_values[i] = val
                row_indices[i] = fx.Int64(idx)

    @flyc.jit
    def launch_register_topk(
        input: fx.Tensor,
        values: fx.Tensor,
        indices: fx.Tensor,
        rows_m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        num_blocks = (rows_m + rows_per_cta - 1) // rows_per_cta
        launcher = register_topk_kernel(input, values, indices, rows_m)
        launcher.launch(
            grid=(num_blocks, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_register_topk


def _read_only(tensor: torch.Tensor) -> Any:
    if not any_cow(tensor):
        return tensor
    from torch._native.const_tensor_wrapper import ConstTensorWrapper

    return ConstTensorWrapper(tensor)


def _make_compile_arg(tensor: torch.Tensor, *, read_only: bool = False) -> Any:
    tensor_arg = _read_only(tensor) if read_only else tensor
    return flyc.from_torch_tensor(tensor_arg).mark_shape_dynamic(0)


def _arch_for(device_index: int) -> str:
    # _resolve_rocm_arch forwards HSA_OVERRIDE_GFX_VERSION and the device's
    # gcnArchName through unchanged, so either may still carry feature flags
    # ("gfx950:sramecc+"); strip them the way the dispatcher predicate does.
    # That predicate already declined when the arch could not be resolved.
    resolved = _resolve_rocm_arch(device_index)
    return resolved.split(":", 1)[0]  # pyrefly: ignore[missing-attribute]


@instrumented_flydsl_cache(
    "aten::topk",
    key_fn=lambda n, k, rows_per_cta, arch, backend, device_index, *a, **kw: (
        f"register N={n} K={k} rows_per_cta={rows_per_cta} {arch} "
        f"backend={backend} device={device_index}"
    ),
)
def _compile_register_topk(
    n: int,
    k: int,
    rows_per_cta: int,
    arch: str,
    backend: str,
    device_index: int,
    *,
    compile_args,
) -> flyc.CompiledFunction:
    # backend and device_index are explicit cache keys, not kernel inputs:
    # FlyDSL reads the backend from its environment and the device from the
    # active HIP context. It reuses compiled artifacts across devices, but the
    # returned callable holds context-local module/function handles and so must
    # be cached per device.
    del backend, device_index
    input_2d, values_2d, indices_2d, rows_m, stream = compile_args
    launch = _build_register_topk_module(n, k, arch, rows_per_cta=rows_per_cta)
    return flyc.compile(
        launch,
        _make_compile_arg(input_2d, read_only=True),
        _make_compile_arg(values_2d),
        _make_compile_arg(indices_2d),
        rows_m,
        stream,
    )


def topk_register_out(
    input_2d: torch.Tensor,
    k: int,
    values_2d: torch.Tensor,
    indices_2d: torch.Tensor,
    *,
    rows_per_cta: int = 2,
) -> None:
    rows_m = input_2d.shape[0]
    n = input_2d.shape[1]
    device_index = input_2d.device.index

    with torch.cuda.device(input_2d.device):
        stream = torch.cuda.current_stream(input_2d.device)
        compiled = _compile_register_topk(
            n,
            k,
            rows_per_cta,
            _arch_for(device_index),
            flyc.compile_backend_name(),
            device_index,
            compile_args=(input_2d, values_2d, indices_2d, rows_m, stream),
        )
        compiled(_read_only(input_2d), values_2d, indices_2d, rows_m, stream)


def topk_register(
    input_2d: torch.Tensor, k: int, *, rows_per_cta: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    rows_m = input_2d.shape[0]
    values_2d = torch.empty((rows_m, k), device=input_2d.device, dtype=input_2d.dtype)
    indices_2d = torch.empty((rows_m, k), device=input_2d.device, dtype=torch.int64)
    topk_register_out(input_2d, k, values_2d, indices_2d, rows_per_cta=rows_per_cta)
    return values_2d, indices_2d


@instrumented_flydsl_cache(
    "aten::topk",
    key_fn=lambda n, k, deterministic, arch, backend, device_index, *a, **kw: (
        f"radix N={n} K={k} deterministic={deterministic} {arch} "
        f"backend={backend} device={device_index}"
    ),
)
def _compile_radix_select_topk(
    n: int,
    k: int,
    deterministic: bool,
    arch: str,
    backend: str,
    device_index: int,
    *,
    compile_args,
) -> flyc.CompiledFunction:
    del backend, device_index  # cache keys only; see _compile_register_topk
    input_2d, values_2d, indices_2d, rows_m, stream = compile_args
    launch = _build_radix_select_topk_module(n, k, deterministic, arch)
    return flyc.compile(
        launch,
        _make_compile_arg(input_2d, read_only=True),
        _make_compile_arg(values_2d),
        _make_compile_arg(indices_2d),
        rows_m,
        stream,
    )


def topk_radix_out(
    input_2d: torch.Tensor,
    k: int,
    values_2d: torch.Tensor,
    indices_2d: torch.Tensor,
    *,
    deterministic: bool = True,
) -> None:
    rows_m = input_2d.shape[0]
    n = input_2d.shape[1]
    device_index = input_2d.device.index

    with torch.cuda.device(input_2d.device):
        stream = torch.cuda.current_stream(input_2d.device)
        compiled = _compile_radix_select_topk(
            n,
            k,
            deterministic,
            _arch_for(device_index),
            flyc.compile_backend_name(),
            device_index,
            compile_args=(input_2d, values_2d, indices_2d, rows_m, stream),
        )
        compiled(_read_only(input_2d), values_2d, indices_2d, rows_m, stream)


def topk_radix(
    input_2d: torch.Tensor, k: int, *, deterministic: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    rows_m = input_2d.shape[0]
    values_2d = torch.empty((rows_m, k), device=input_2d.device, dtype=input_2d.dtype)
    indices_2d = torch.empty((rows_m, k), device=input_2d.device, dtype=torch.int64)
    topk_radix_out(input_2d, k, values_2d, indices_2d, deterministic=deterministic)
    return values_2d, indices_2d


_COMPILERS = {"register": _compile_register_topk, "radix": _compile_radix_select_topk}


def clear_topk_cache() -> None:
    for compiler in _COMPILERS.values():
        cast(_CachedCompile, compiler).cache_clear()


def topk_cache_info(kernel: str | None = None) -> CacheInfo:
    """Compile-cache stats for one kernel, or summed over both.

    Pass a kernel name to pin which specialization a shape dispatched to: the
    sum cannot tell a register compile from a radix one, so an assertion on it
    alone still passes when the gate picks the wrong kernel.
    """
    if kernel is not None:
        return cast(_CachedCompile, _COMPILERS[kernel]).cache_info()
    infos = [cast(_CachedCompile, c).cache_info() for c in _COMPILERS.values()]
    return CacheInfo(
        sum(i.hits for i in infos),
        sum(i.misses for i in infos),
        sum(i.currsize for i in infos),
    )
