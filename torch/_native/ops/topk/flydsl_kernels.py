"""FlyDSL top-K kernels, picked by K and N in ``flydsl_impl.py``.

``TopKSpec`` is the compile-time specialization (dtype and ``largest``). It is
closed over by the builders; it is not a kernel argument. Ordinals are signed
int32 so ``maxsi`` can compare them directly. Narrow values sit in the high
bits, sign at bit 31. ``largest=False`` complements that int32.

``topk_radix`` - one CTA per row, ``max(sort_len, _N_HIST_BINS)``
threads (``sort_len`` = K rounded up to a power of 2; the bin floor lets
phase 1 zero the histogram one bin per thread).
  1. Radix byte passes (MSB to LSB, one per byte of the ordinal) over an LDS
     histogram; thread 0 scans descending, narrowing ``s_prefix``/``s_mask``
     until it holds the K-th ordinal and ``s_rem_k`` the ties to keep.
  2. Gather at-or-above that threshold. Deterministic: prefix-sum slots
     preserve input order, lowest-index ties win. Otherwise LDS atomic
     counters claim slots and ties resolve per run.
  3. Bitonic sort over ``sort_len`` slots; padding is the int64 minimum.
     A real key's low half is ``~idx``, never zero, so the pad stays strictly
     below every real key. Comparator is always lexicographic
     ``(ord desc, idx asc)``; ``deterministic`` only controls which
     threshold-tied elements phase 2 gathers.
  4. Write K (value, index) pairs.

``topk_register`` - one wave per row. Each lane bitonic-sorts its
``N / wave_size`` Int64 ``(ord << 32) | ~idx`` keys, then merges with
butterfly partners until lane 0 holds the top-K; only the final K writes
hit gmem.

``TopKSpec.to_ord`` leaves positive floats alone and flips every non-sign bit
of a negative float, so the field compares as a signed integer in value
order. NaN collapses to the maximum code of that field (fp32 ``0x7FFFFFFF``,
fp16/bf16 ``0x7FFF``), the signed form of ATen ``TopKTypeConfig``'s all-ones
NaN. Not injective on NaN, so the caller re-reads gmem for that case.
Signed integers are already in signed order and are only shifted up. Phase
1's xor by ``_RADIX_SIGN_BIT`` is separate: it puts the top byte in unsigned
bin order for the scan.

``_make_topk_storage`` unions ``s_hist``, the phase-2 counters, ``s_scan``
and ``s_keys`` - they alias, each live only within its phase. The phase-2
barrier before bitonic sort is the hand-off: after it ``s_scan`` is dead and
``s_keys`` is live until phase 4. Anything spanning phases needs a struct
member, not a union member.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import cast

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import (
    Array,
    BFloat16,
    Float16,
    Float32,
    gpu,
    Int16,
    Int32,
    Int64,
    Int8,
    range_constexpr,
)
from flydsl.runtime.device import is_rdna_arch

import torch
from torch._native.flydsl.cache import CachedCompile, CacheInfo
from torch._native.flydsl.compile_args import make_compile_arg, read_only_tensor
from torch._native.flydsl.intrinsics import atomic_add, maxsi, minsi
from torch._native.flydsl_utils import _resolve_rocm_arch
from torch._native.instrumentation import instrumented_flydsl_cache


_RADIX_BITS = 8
_RADIX_MASK = (1 << _RADIX_BITS) - 1
_RADIX_SIGN_BIT = 1 << (_RADIX_BITS - 1)
_N_HIST_BINS = 1 << _RADIX_BITS
_REGISTER_ROWS_PER_CTA = 2
_ORD_BITS = 32


def _u32_to_i32(x: int) -> int:
    return x - (1 << 32) if x >= (1 << 31) else x


@dataclass(frozen=True)
class TopKSpec:
    """Encode a value as a signed int32 ordinal and decode it.

    The dtype selects an involutive, same-width bit transform. The spec
    left-aligns its result in an int32, then complements it for
    ``largest=False`` to reverse the order. NaNs collapse to a sentinel;
    the caller detects that sentinel and reloads the original value.
    """

    dtype: torch.dtype
    elem_ty: type
    int_ty: type
    width: int
    largest: bool
    # Involutive transform between same-width value bits and narrow ordinal.
    # A negative float flips every bit below its sign; an integer is the
    # identity. The dtype spec is already bound in. Not part of equality.
    val_to_narrow_ord: Callable = field(compare=False, hash=False)
    nan_sentinel: int | None = None
    nan_exp: int | None = None

    def cache_token(self) -> str:
        name = str(self.dtype).removeprefix("torch.")
        return f"{name} largest={self.largest}"

    @property
    def vec(self) -> int:
        # Values carried by one 16-byte buffer load.
        return (8 * 16) // self.width

    @property
    def passes(self) -> int:
        # Radix digits of the ordinal, counted down from bit 31.
        return self.width // _RADIX_BITS

    @property
    def left_shift(self) -> int:
        return _ORD_BITS - self.width

    @property
    def has_nan(self) -> bool:
        return self.nan_sentinel is not None

    def to_ord(self, val):
        # val -> ord(val), one int32.
        # 0. Reinterpret as same-width signed bits. Collapse a NaN before
        #    applying the reversible steps; the sentinel has a clear sign
        #    bit, so the narrow transform leaves it unchanged.
        raw = val.bitcast(self.int_ty)
        if self.has_nan:
            nan_exp = self.int_ty(cast(int, self.nan_exp))
            nan_sentinel = self.int_ty(cast(int, self.nan_sentinel))
            low_mask = (1 << (self.width - 1)) - 1
            is_nan = (raw & self.int_ty(low_mask)) > nan_exp
            raw = is_nan.select(nan_sentinel, raw)
        # 1. Apply the dtype's same-width transform.
        narrow = self.val_to_narrow_ord(raw)
        # 2. Widen and left-shift the low `width` bits to the top of the
        #    int32. Their sign lands at bit 31 and the bits below are zero.
        ord32 = fx.Int32(narrow)
        if self.left_shift:
            ord32 = ord32 << fx.Int32(self.left_shift)
        # 3. Complement when largest=False.
        if not self.largest:
            ord32 = ~ord32
        return ord32

    def from_ord(self, stored):
        # Inverse of to_ord, last step first. `stored` is that int32.
        # 3 undone. Complement back when largest=False.
        ord32 = ~stored if not self.largest else stored
        # 2 undone. Arithmetic shift fills ones above a negative field.
        # Truncating to `int_ty` keeps the low `width` bits.
        bits32 = ord32 if self.left_shift == 0 else ord32 >> fx.Int32(self.left_shift)
        # 1 undone. The same ordinal, at the field's own width, then bitcast
        # back to the value type. An integer bitcast to its own type is
        # removed by canonicalize.
        narrow = self.val_to_narrow_ord(self.int_ty(bits32))
        val = narrow.bitcast(self.elem_ty)
        # 0 cannot be undone: NaN collapse is not injective. The caller
        # detects `nan_sentinel` in `narrow` and reloads the original value.
        return val, narrow


def topk_spec(dtype: torch.dtype, largest: bool) -> TopKSpec:
    # torch dtype: flydsl dtype, same-width signed int, bit width, nan sentinel, nan exponent.
    # None for the NaN fields means the dtype has no NaN.
    def int_to_narrow_ord(bits, spec):
        return bits

    def float_to_narrow_ord(bits, spec):
        # S sign, E exponent, F fraction.
        # [S | E | F] -> [0 |  E |  F]  when S = 0
        #             -> [1 | ~E | ~F]  when S = 1
        # Constants use the field's integer type, so a 16-bit value stays 16-bit.
        # `(1 << (width - 1)) - 1` is the bits under the sign at any width.
        width = spec["width"]
        int_ty = spec["int_ty"]
        low_mask = (1 << (width - 1)) - 1
        sign = (bits >> int_ty(width - 1)) & int_ty(1)
        mask = (int_ty(0) - sign) & int_ty(low_mask)
        return bits ^ mask

    specs = {
        torch.float32: dict(
            elem_ty=Float32,
            int_ty=Int32,
            width=32,
            nan_sentinel=0x7FFFFFFF,
            nan_exp=0x7F800000,
            to_narrow_ord=float_to_narrow_ord,
        ),
        torch.float16: dict(
            elem_ty=Float16,
            int_ty=Int16,
            width=16,
            nan_sentinel=0x7FFF,
            nan_exp=0x7C00,
            to_narrow_ord=float_to_narrow_ord,
        ),
        torch.bfloat16: dict(
            elem_ty=BFloat16,
            int_ty=Int16,
            width=16,
            nan_sentinel=0x7FFF,
            nan_exp=0x7F80,
            to_narrow_ord=float_to_narrow_ord,
        ),
        torch.int32: dict(
            elem_ty=Int32,
            int_ty=Int32,
            width=32,
            to_narrow_ord=int_to_narrow_ord,
        ),
        torch.int16: dict(
            elem_ty=Int16,
            int_ty=Int16,
            width=16,
            to_narrow_ord=int_to_narrow_ord,
        ),
        torch.int8: dict(
            elem_ty=Int8,
            int_ty=Int8,
            width=8,
            to_narrow_ord=int_to_narrow_ord,
        ),
    }
    if dtype not in specs:
        raise NotImplementedError(f"FlyDSL topk has no ordinal spec for {dtype}")
    dtype_spec = specs[dtype]
    spec = TopKSpec(
        dtype=dtype,
        largest=largest,
        val_to_narrow_ord=partial(dtype_spec["to_narrow_ord"], spec=dtype_spec),
        **{key: value for key, value in dtype_spec.items() if key != "to_narrow_ord"},
    )
    if spec.width > _ORD_BITS or spec.width % _RADIX_BITS:
        raise NotImplementedError(
            f"FlyDSL topk ordinal does not fit a {_ORD_BITS}-bit key for {dtype}"
        )
    return spec


def _load_vec(spec: TopKSpec, copy_atom, input_div, idx):
    r = fx.make_rmem_tensor(spec.vec, spec.elem_ty)
    fx.copy_atom_call(copy_atom, fx.slice(input_div, (None, idx)), r)
    return fx.memref_load_vec(r)


def _make_key(val, idx, spec: TopKSpec):
    ord64 = fx.Int64(spec.to_ord(val)) & fx.Int64(0xFFFFFFFF)
    inv_idx64 = fx.Int64(~idx) & fx.Int64(0xFFFFFFFF)
    return (ord64 << fx.Int64(32)) | inv_idx64


def _split_key(key):
    stored = fx.Int32(key >> fx.Int64(32))
    idx = ~fx.Int32(key & fx.Int64(0xFFFFFFFF))
    return stored, idx


def _make_topk_storage(sort_len: int, block_threads: int, spec: TopKSpec):
    @fx.struct
    class CounterStorage:
        s_write_ctr: Array[Int32, 1, 4]
        s_eq_ctr: Array[Int32, 1, 4]

    @fx.union
    class PhaseStorage:
        s_hist: Array[Int32, _N_HIST_BINS, 16]
        s_counters: CounterStorage
        s_scan: Array[Int32, block_threads + 1, 16]
        s_keys: Array[Int64, sort_len, 16]

    @fx.struct
    class SharedStorage:
        s_phase: PhaseStorage
        s_prefix: Array[Int32, 1, 4]
        s_mask: Array[Int32, 1, 4]
        s_rem_k: Array[Int32, 1, 4]
        s_vals: Array[spec.elem_ty, sort_len, 16]
        s_idxs: Array[Int32, sort_len, 16]

    return SharedStorage


def _build_radix_select_topk_module(
    n: int, k: int, deterministic: bool, arch: str, spec: TopKSpec
):
    num_stages = (k - 1).bit_length()
    sort_len = 1 << num_stages
    block_threads = max(sort_len, _N_HIST_BINS)
    max_step_elems = block_threads * spec.vec
    if max_step_elems >= (1 << 15):
        raise AssertionError(
            "packed deterministic scan needs < 32768 elements per step, "
            f"got {max_step_elems}"
        )
    warp_size = 32 if is_rdna_arch(arch) else 64
    num_warps = block_threads // warp_size
    tile = block_threads * spec.vec
    vec_iters = n // tile
    vec_end = vec_iters * tile
    scalar_tail_iters = (n - vec_end + block_threads - 1) // block_threads

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
        input_div = fx.logical_divide(row_in, fx.make_layout(spec.vec, 1))
        copy_atom_v = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), spec.width)
        storage = fx.SharedAllocator().allocate(
            _make_topk_storage(sort_len, block_threads, spec)
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

        def decode_topk_key(key):
            stored, idx = _split_key(key)
            val, narrow = spec.from_ord(stored)
            if spec.has_nan:
                if narrow == spec.int_ty(spec.nan_sentinel):
                    val = row_in[idx]
            return val, idx

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

        def accumulate_radix_byte_hist(
            ords, byte_pos, prefix, decided_mask, shift, sign_bit_xor_val
        ):
            if byte_pos == 0 or (ords & decided_mask) == prefix:
                byte_val = (
                    (ords >> fx.Int32(shift)) & fx.Int32(_RADIX_MASK)
                ) ^ fx.Int32(sign_bit_xor_val)
                atomic_add(fx.get_iter(s_hist) + byte_val, 1)

        if tid == 0:
            s_prefix[0] = 0
            s_mask[0] = 0
            s_rem_k[0] = k
        gpu.barrier()

        # Phase 1: radix byte passes, MSB to LSB.
        for byte_pos in range_constexpr(spec.passes):
            # The ordinal occupies the top spec.width of the int32, so
            # pass 0 always reads bit 31. spec.passes stops at the bottom
            # of that field.
            shift = (_ORD_BITS // _RADIX_BITS - 1 - byte_pos) * _RADIX_BITS
            sign_bit_xor_val = _RADIX_SIGN_BIT if byte_pos == 0 else 0

            # Zero the histogram.
            if tid < fx.Int32(_N_HIST_BINS):
                s_hist[tid] = 0
            gpu.barrier()

            # Accumulate the current byte histogram.
            prefix = s_prefix[0]
            decided_mask = s_mask[0]
            for step in range_constexpr(vec_iters):
                rvals = _load_vec(
                    spec, copy_atom_v, input_div, step * block_threads + tid
                )
                for vi in range_constexpr(spec.vec):
                    accumulate_radix_byte_hist(
                        spec.to_ord(rvals[vi]),
                        byte_pos,
                        prefix,
                        decided_mask,
                        shift,
                        sign_bit_xor_val,
                    )
            for step in range_constexpr(scalar_tail_iters):
                col = vec_end + step * block_threads + tid
                if col < n:
                    accumulate_radix_byte_hist(
                        spec.to_ord(row_in[col]),
                        byte_pos,
                        prefix,
                        decided_mask,
                        shift,
                        sign_bit_xor_val,
                    )
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

                actual_byte = sel_bin ^ fx.Int32(sign_bit_xor_val)
                s_prefix[0] = prefix | (actual_byte << fx.Int32(shift))
                s_mask[0] = decided_mask | fx.Int32(_u32_to_i32(_RADIX_MASK << shift))
                s_rem_k[0] = remaining_k - elems_above
            gpu.barrier()

        # Phase 2: gather elements at-or-above the radix threshold.
        threshold = s_prefix[0]  # Values above threshold are in top-k.
        remaining_k = s_rem_k[0]  # Number of threshold ties to keep.

        if deterministic:
            n_above = fx.Int32(k) - remaining_k
            above_base = 0
            eq_base = 0

            def gather_deterministic_step(elems, above_base, eq_base, vals, idxs):
                num_elems = len(elems)
                # Class encoding: 2 = above, 1 = equal, 0 = below.
                cls_reg = fx.make_rmem_tensor(num_elems, Int32)
                packed_local = 0
                for elem_i in range_constexpr(num_elems):
                    valid, val, _ = elems[elem_i]
                    cls_reg[elem_i] = 0
                    if valid:
                        ords = spec.to_ord(val)
                        if ords > threshold:
                            cls_reg[elem_i] = 2
                            packed_local = packed_local + fx.Int32(1 << 16)
                        elif ords == threshold:
                            cls_reg[elem_i] = 1
                            packed_local = packed_local + 1

                packed_pfx, packed_step_total = block_excl_prefix_i32(
                    packed_local, s_scan
                )
                my_above = above_base + (packed_pfx >> fx.Int32(16))
                my_eq = eq_base + (packed_pfx & fx.Int32(0xFFFF))
                for elem_i in range_constexpr(num_elems):
                    valid, val, idx = elems[elem_i]
                    cls = cls_reg[elem_i]
                    if valid:
                        if cls == 2:
                            vals[my_above] = val
                            idxs[my_above] = idx
                            my_above = my_above + 1
                        elif cls == 1:
                            if my_eq < remaining_k:
                                pos = n_above + my_eq
                                vals[pos] = val
                                idxs[pos] = idx
                            my_eq = my_eq + 1

                above_base = above_base + (packed_step_total >> fx.Int32(16))
                next_eq_base = eq_base + (packed_step_total & fx.Int32(0xFFFF))
                eq_base = (next_eq_base < remaining_k).select(next_eq_base, remaining_k)
                return above_base, eq_base

            for step in range_constexpr(vec_iters):
                tile_base = step * tile
                base = tile_base + tid * spec.vec
                rvals = _load_vec(
                    spec, copy_atom_v, input_div, step * block_threads + tid
                )
                elems = []
                for vi in range_constexpr(spec.vec):
                    idx = fx.Int32(base + vi)
                    elems.append((True, rvals[vi], idx))
                above_base, eq_base = gather_deterministic_step(
                    elems, above_base, eq_base, s_vals, s_idxs
                )

            # --- Scalar tail ---
            for step in range_constexpr(scalar_tail_iters):
                col = vec_end + step * block_threads + tid
                valid = col < n
                col_safe = valid.select(col, 0)
                elems = [(valid, row_in[col_safe], fx.Int32(col))]
                above_base, eq_base = gather_deterministic_step(
                    elems, above_base, eq_base, s_vals, s_idxs
                )
        else:
            if tid == 0:
                s_write_ctr[0] = 0
                s_eq_ctr[0] = 0
            gpu.barrier()

            def gather_candidate(val, idx, vals, idxs):
                ords = spec.to_ord(val)
                if ords > threshold:
                    pos = atomic_add(fx.get_iter(s_write_ctr), 1)
                    vals[pos] = val
                    idxs[pos] = idx
                elif ords == threshold:
                    eq_pos = atomic_add(fx.get_iter(s_eq_ctr), 1)
                    if eq_pos < remaining_k:
                        pos = atomic_add(fx.get_iter(s_write_ctr), 1)
                        vals[pos] = val
                        idxs[pos] = idx

            for step in range_constexpr(vec_iters):
                tile_base = step * tile
                base = tile_base + tid * spec.vec
                rvals = _load_vec(
                    spec, copy_atom_v, input_div, step * block_threads + tid
                )
                for vi in range_constexpr(spec.vec):
                    idx = fx.Int32(base + vi)
                    gather_candidate(rvals[vi], idx, s_vals, s_idxs)
            for step in range_constexpr(scalar_tail_iters):
                col = vec_end + step * block_threads + tid
                if col < n:
                    gather_candidate(row_in[col], fx.Int32(col), s_vals, s_idxs)

        gpu.barrier()  # Phase 3 owns s_keys; s_scan is dead after phase 2.

        # Phase 3: cooperative bitonic sort.
        active = tid < fx.Int32(k)
        sort_active = tid < fx.Int32(sort_len)
        sort_tid = sort_active.select(tid, 0)
        if sort_active:
            if active:
                s_keys[sort_tid] = _make_key(s_vals[sort_tid], s_idxs[sort_tid], spec)
            else:
                s_keys[sort_tid] = fx.Int64(-(1 << 63))
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
                                s_keys[sort_tid] = partner_key
                        else:
                            if self_lt_partner:
                                s_keys[sort_tid] = partner_key
                    else:
                        if block_dir == 0:
                            if self_lt_partner:
                                s_keys[sort_tid] = partner_key
                        else:
                            if self_gt_partner:
                                s_keys[sort_tid] = partner_key
                gpu.barrier()

        # Phase 4: results to gmem.
        if tid < fx.Int32(k):
            val, idx = decode_topk_key(s_keys[tid])
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


def _build_register_topk_module(n: int, k: int, arch: str, spec: TopKSpec):
    threads_per_row = 32 if is_rdna_arch(arch) else 64
    block_threads = threads_per_row * _REGISTER_ROWS_PER_CTA
    vec = n // threads_per_row
    # Process elements group by group to reduce VGPR usage and preserve occupancy.
    # Groups are K elements, widened to the 16-byte load width. N >= 1024 is a
    # power of two, with threads_per_row = 32 or 64.
    group = max(k, spec.vec)
    if vec % group:
        raise AssertionError(f"vec={vec} must be divisible by group={group}")
    if group % spec.vec:
        raise AssertionError(f"group={group} must be divisible by vec={spec.vec}")
    group_loads = group // spec.vec
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
        row = bid * fx.Int32(_REGISTER_ROWS_PER_CTA) + row_local
        row_safe = (row < rows_m).select(row, 0)
        in_bounds = row < rows_m

        input_buf = fx.rocdl.make_buffer_tensor(input)
        values_buf = fx.rocdl.make_buffer_tensor(values)
        indices_buf = fx.rocdl.make_buffer_tensor(indices)
        row_input = fx.slice(input_buf, (row_safe, None))
        row_values = fx.slice(values_buf, (row_safe, None))
        row_indices = fx.slice(indices_buf, (row_safe, None))
        input_div = fx.logical_divide(row_input, fx.make_layout(spec.vec, 1))
        copy_atom_v = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), spec.width)

        def decode_topk_key(key):
            stored, idx = _split_key(key)
            val, narrow = spec.from_ord(stored)
            if spec.has_nan:
                if narrow == spec.int_ty(spec.nan_sentinel):
                    val = row_input[idx]
            return val, idx

        def compare_and_swap(arr, i: int, j: int, descending: bool):
            a = arr[i]
            b = arr[j]
            hi = maxsi(a, b)
            lo = minsi(a, b)
            arr[i] = hi if descending else lo
            arr[j] = lo if descending else hi

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
        # step has the wave reading a contiguous vec * wave span. Which
        # columns a lane ends up owning is irrelevant: both the fold below and
        # the butterfly merge after it only ever keep the top-K of a union.
        topk = fx.make_rmem_tensor(k, Int64)
        buf = fx.make_rmem_tensor(group, Int64)
        for g in range_constexpr(num_groups):
            for j in range_constexpr(group_loads):
                chunk = fx.Int32((g * group_loads + j) * threads_per_row) + lane
                rvals = _load_vec(spec, copy_atom_v, input_div, chunk)
                base = chunk * fx.Int32(spec.vec)
                for e in range_constexpr(spec.vec):
                    buf[j * spec.vec + e] = _make_key(
                        rvals[e], base + fx.Int32(e), spec
                    )
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
                val, idx = decode_topk_key(topk[i])
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
        num_blocks = (rows_m + _REGISTER_ROWS_PER_CTA - 1) // _REGISTER_ROWS_PER_CTA
        launcher = register_topk_kernel(input, values, indices, rows_m)
        launcher.launch(
            grid=(num_blocks, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_register_topk


@instrumented_flydsl_cache(
    "aten::topk",
    key_fn=lambda n, k, arch, spec, backend, device_index, *a, **kw: (
        f"register N={n} K={k} {spec.cache_token()} {arch} "
        f"backend={backend} device={device_index}"
    ),
)
def _compile_register_topk(
    n: int,
    k: int,
    arch: str,
    spec: TopKSpec,
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
    launch = _build_register_topk_module(n, k, arch, spec)
    return flyc.compile(
        launch,
        make_compile_arg(input_2d, read_only=True),
        make_compile_arg(values_2d),
        make_compile_arg(indices_2d),
        rows_m,
        stream,
    )


def topk_register_out(
    input_2d: torch.Tensor,
    k: int,
    values_2d: torch.Tensor,
    indices_2d: torch.Tensor,
    *,
    largest: bool = True,
) -> None:
    rows_m = input_2d.shape[0]
    n = input_2d.shape[1]
    device_index = input_2d.device.index
    arch: str = _resolve_rocm_arch(device_index)  # pyrefly: ignore[bad-assignment]
    spec = topk_spec(input_2d.dtype, largest)

    with torch.cuda.device(input_2d.device):
        stream = torch.cuda.current_stream(input_2d.device)
        compiled = _compile_register_topk(
            n,
            k,
            arch,
            spec,
            flyc.compile_backend_name(),
            device_index,
            compile_args=(input_2d, values_2d, indices_2d, rows_m, stream),
        )
        compiled(read_only_tensor(input_2d), values_2d, indices_2d, rows_m, stream)


def topk_register(
    input_2d: torch.Tensor, k: int, *, largest: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    rows_m = input_2d.shape[0]
    values_2d = torch.empty((rows_m, k), device=input_2d.device, dtype=input_2d.dtype)
    indices_2d = torch.empty((rows_m, k), device=input_2d.device, dtype=torch.int64)
    topk_register_out(input_2d, k, values_2d, indices_2d, largest=largest)
    return values_2d, indices_2d


@instrumented_flydsl_cache(
    "aten::topk",
    key_fn=lambda n, k, deterministic, arch, spec, backend, device_index, *a, **kw: (
        f"radix N={n} K={k} deterministic={deterministic} {spec.cache_token()} "
        f"{arch} backend={backend} device={device_index}"
    ),
)
def _compile_radix_select_topk(
    n: int,
    k: int,
    deterministic: bool,
    arch: str,
    spec: TopKSpec,
    backend: str,
    device_index: int,
    *,
    compile_args,
) -> flyc.CompiledFunction:
    del backend, device_index  # cache keys only; see _compile_register_topk
    input_2d, values_2d, indices_2d, rows_m, stream = compile_args
    launch = _build_radix_select_topk_module(n, k, deterministic, arch, spec)
    return flyc.compile(
        launch,
        make_compile_arg(input_2d, read_only=True),
        make_compile_arg(values_2d),
        make_compile_arg(indices_2d),
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
    largest: bool = True,
) -> None:
    rows_m = input_2d.shape[0]
    n = input_2d.shape[1]
    device_index = input_2d.device.index
    arch: str = _resolve_rocm_arch(device_index)  # pyrefly: ignore[bad-assignment]
    spec = topk_spec(input_2d.dtype, largest)

    with torch.cuda.device(input_2d.device):
        stream = torch.cuda.current_stream(input_2d.device)
        compiled = _compile_radix_select_topk(
            n,
            k,
            deterministic,
            arch,
            spec,
            flyc.compile_backend_name(),
            device_index,
            compile_args=(input_2d, values_2d, indices_2d, rows_m, stream),
        )
        compiled(read_only_tensor(input_2d), values_2d, indices_2d, rows_m, stream)


def topk_radix(
    input_2d: torch.Tensor,
    k: int,
    *,
    deterministic: bool = True,
    largest: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows_m = input_2d.shape[0]
    values_2d = torch.empty((rows_m, k), device=input_2d.device, dtype=input_2d.dtype)
    indices_2d = torch.empty((rows_m, k), device=input_2d.device, dtype=torch.int64)
    topk_radix_out(
        input_2d,
        k,
        values_2d,
        indices_2d,
        deterministic=deterministic,
        largest=largest,
    )
    return values_2d, indices_2d


_COMPILERS = {"register": _compile_register_topk, "radix": _compile_radix_select_topk}


def clear_topk_cache() -> None:
    for compiler in _COMPILERS.values():
        cast(CachedCompile, compiler).cache_clear()


def topk_cache_info(kernel: str | None = None) -> CacheInfo:
    """Compile-cache stats for one kernel, or summed over both.

    Pass a kernel name to pin which specialization a shape dispatched to: the
    sum cannot tell a register compile from a radix one, so an assertion on it
    alone still passes when the gate picks the wrong kernel.
    """
    if kernel is not None:
        return cast(CachedCompile, _COMPILERS[kernel]).cache_info()
    infos = [cast(CachedCompile, c).cache_info() for c in _COMPILERS.values()]
    return CacheInfo(
        sum(i.hits for i in infos),
        sum(i.misses for i in infos),
        sum(i.currsize for i in infos),
    )
