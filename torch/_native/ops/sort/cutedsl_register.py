"""Register-resident sort of rows of at most 2048 elements.

One warp per row: lane ``l`` holds ``VEC`` consecutive elements of the row
padded to the rung ``32 * VEC``, and a bitonic network runs in registers and
shuffles, with no shared memory. Each element is the unique word
``ordinal << IDX_BITS | index`` (64-bit keys keep the index beside it), so the
result is the stable order; 1-byte keys at rungs 64 to 256 sort two words per
register. The ordinal is the replaced stock leaf's: comparison order up to 128,
radix order above. Values are decoded from the ordinal, and keys it cannot
invert (zeros, NaN under the comparison order) are re-read from ``x``.
"""

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T

import torch
from torch._native.cutedsl.launch import compile_kernel, fake_compact, read_only, sym
from torch._native.instrumentation import instrumented_cutedsl_cache
from torch._native.utils.tensor import const_data_ptr

from .cutedsl_common import (
    INDEX_DTYPES,
    index_name,
    KEY_KINDS,
    radix_ordinal,
    RAW_CUTE,
    raw_view,
)
from .cutedsl_inplace import _cmp_ordinal
from .policy import (
    REGISTER_ROWS_PER_CTA,
    register_rung,
    register_store,
    vector_alignment,
    WARP_MERGE_MAX_NSORT,
)


# Inline PTX: cute.arch has no min/max.u16x2 (sm_90+, as the router requires),
# and cute.arch.prmt marks its asm side-effecting and takes the selector in a
# register.
@dsl_user_op
def _ptx(asm: str, *operands, loc=None, ip=None) -> Int32:
    constraint = ",".join(["=r"] + ["r"] * len(operands))
    values = [Int32(v).ir_value(loc=loc, ip=ip) for v in operands]
    return Int32(llvm.inline_asm(T.i32(), values, asm, constraint, loc=loc, ip=ip))


def _vmin2(a, b):
    return _ptx("min.u16x2 $0, $1, $2;", a, b)


def _vmax2(a, b):
    return _ptx("max.u16x2 $0, $1, $2;", a, b)


def _prmt(a, b, selector: int):
    return _ptx(f"prmt.b32 $0, $1, $2, {selector};", a, b)


def _at(regs: cute.Tensor | None, i: int):
    return None if regs is None else regs[i]


class RegisterSort:
    def __init__(
        self,
        key: str,
        index_dtype: type[cutlass.Numeric],
        descending: bool,
        rung: int,
        rows: int,
        store: int,
    ):
        self.key = key
        self.kind, self.key_bits = KEY_KINDS[getattr(torch, key)]
        self.raw_dtype = RAW_CUTE[self.key_bits]
        self.index_dtype = index_dtype
        self.descending = descending
        self.vec = rung // 32
        self.idx_bits = (rung - 1).bit_length()
        self.cmp = rung <= WARP_MERGE_MAX_NSORT
        self.rows = rows
        self.store = store

    def __repr__(self):
        # CuTeDSL names the kernel after this, so profiles show rung and width.
        return f"<{type(self).__module__}.RegisterSort rung{32 * self.vec} store{self.store}>"

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        stream: cuda.CUstream,
    ):
        rows = const_expr(self.rows)
        self.kernel(mX, mValues, mIndices).launch(
            # Overflow-free round-up: rows reach policy.MAX_ROWS.
            grid=[(mX.shape[0] - 1) // rows + 1, 1, 1],
            block=[32 * rows, 1, 1],
            stream=stream,
        )

    @cute.jit
    def ordinal(self, bits):
        if const_expr(self.cmp):
            return _cmp_ordinal(
                bits, self.key, self.kind, self.key_bits, self.descending
            )
        return radix_ordinal(bits, self.kind, self.key_bits, self.descending)

    @cute.jit
    def less(self, ka, ia, kb, ib):
        """Word order; 64-bit keys break ties by their separate index."""
        if const_expr(ia is None):
            return ka < kb
        return ka < kb or (ka == kb and ia < ib)

    @cute.jit
    def network(self, keys: cute.Tensor, index: cute.Tensor | None, lane: Int32):
        """Ascending bitonic sort without direction flags: the first step of
        merge ``size`` pairs e with e ^ (size - 1), later steps e with
        e ^ stride; the lower element of each pair keeps the minimum. ``index``
        holds the tie-breaking indices of 64-bit keys and is None otherwise."""
        VEC = const_expr(self.vec)
        LOG_VEC = const_expr(VEC.bit_length() - 1)
        wide = const_expr(self.key_bits == 64)
        for log_size in cutlass.range_constexpr(1, self.idx_bits + 1):
            size = 1 << log_size
            for log_stride in cutlass.range_constexpr(log_size - 1, -1, -1):
                stride = 1 << log_stride
                mask = size - 1 if const_expr(log_stride == log_size - 1) else stride
                if const_expr(stride < VEC):
                    for i in cutlass.range_constexpr(VEC):
                        p = i ^ mask
                        if const_expr(p > i):
                            if self.less(
                                keys[p], _at(index, p), keys[i], _at(index, i)
                            ):
                                keys[i], keys[p] = keys[p], keys[i]
                                if const_expr(wide):
                                    index[i], index[p] = index[p], index[i]
                else:
                    lane_mask = mask >> LOG_VEC
                    flip = const_expr((mask & (VEC - 1)) != 0)
                    lower = (lane & Int32(stride >> LOG_VEC)) == Int32(0)
                    for i in cutlass.range_constexpr(VEC):
                        p = i ^ (VEC - 1) if const_expr(flip) else i
                        if const_expr(p >= i):
                            ka = cute.arch.shuffle_sync_bfly(keys[p], offset=lane_mask)
                            kb = cute.arch.shuffle_sync_bfly(keys[i], offset=lane_mask)
                            ia = ib = None
                            if const_expr(wide):
                                ia = cute.arch.shuffle_sync_bfly(
                                    index[p], offset=lane_mask
                                )
                                ib = cute.arch.shuffle_sync_bfly(
                                    index[i], offset=lane_mask
                                )
                            if self.less(ka, ia, keys[i], _at(index, i)) == lower:
                                keys[i] = ka
                                if const_expr(wide):
                                    index[i] = ia
                            if const_expr(p > i):
                                if self.less(kb, ib, keys[p], _at(index, p)) == lower:
                                    keys[p] = kb
                                    if const_expr(wide):
                                        index[p] = ib

    @cute.jit
    def packed_network(self, keys: cute.Tensor, lane: Int32):
        """``network`` on 16-bit words, two per register: register t holds
        positions t (low half) and t + VEC / 2 (high half)."""
        VEC = const_expr(self.vec)
        HALF = const_expr(VEC // 2)
        LOG_VEC = const_expr(VEC.bit_length() - 1)
        packed = cute.make_rmem_tensor(HALF, Int32)
        for t in cutlass.range_constexpr(HALF):
            packed[t] = keys[t] | (keys[t + HALF] << Int32(16))
        for log_size in cutlass.range_constexpr(1, self.idx_bits + 1):
            size = 1 << log_size
            for log_stride in cutlass.range_constexpr(log_size - 1, -1, -1):
                stride = 1 << log_stride
                mask = size - 1 if const_expr(log_stride == log_size - 1) else stride
                reg_mask = mask & (VEC - 1)
                if const_expr(stride < VEC):
                    # Pairs (t, t ^ reg_mask); with the HALF bit set, the low
                    # half of t meets the high half of u = t ^ reg_mask ^ HALF.
                    cross = const_expr(reg_mask & HALF != 0)
                    for t in cutlass.range_constexpr(HALF):
                        u = t ^ (reg_mask & (HALF - 1))
                        if const_expr(not cross and u > t):
                            a, b = packed[t], packed[u]
                            packed[t], packed[u] = _vmin2(a, b), _vmax2(a, b)
                        elif const_expr(cross and u >= t):
                            a, b = packed[t], _prmt(packed[u], packed[u], 0x1032)
                            lo, hi = _vmin2(a, b), _vmax2(a, b)
                            packed[t] = _prmt(lo, hi, 0x7610)
                            if const_expr(u > t):
                                packed[u] = _prmt(lo, hi, 0x5432)
                else:
                    # On a flip, t needs the partner's HALF - 1 - t, halves
                    # swapped.
                    flip = const_expr(reg_mask != 0)
                    lower = (lane & Int32(stride >> LOG_VEC)) == Int32(0)
                    other = cute.make_rmem_tensor(HALF, Int32)
                    for t in cutlass.range_constexpr(HALF):
                        w = cute.arch.shuffle_sync_bfly(
                            packed[HALF - 1 - t if const_expr(flip) else t],
                            offset=mask >> LOG_VEC,
                        )
                        other[t] = _prmt(w, w, 0x1032) if const_expr(flip) else w
                    for t in cutlass.range_constexpr(HALF):
                        if lower:
                            packed[t] = _vmin2(packed[t], other[t])
                        else:
                            packed[t] = _vmax2(packed[t], other[t])
        for t in cutlass.range_constexpr(HALF):
            keys[t] = packed[t] & Int32(0xFFFF)
            keys[t + HALF] = (packed[t] >> Int32(16)) & Int32(0xFFFF)

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mValues: cute.Tensor, mIndices: cute.Tensor):
        VEC = const_expr(self.vec)
        IDX_BITS = const_expr(self.idx_bits)
        LOG_VEC = const_expr(VEC.bit_length() - 1)
        wide = const_expr(self.key_bits == 64)
        word_t = Int64 if const_expr(self.key_bits >= 32) else Int32
        bits_t = Int64 if const_expr(wide) else Int32
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % Int32(32)
        row = bidx * Int32(self.rows) + tidx // Int32(32)
        n = mX.shape[1]
        row_valid = row < mX.shape[0]

        # STORE (``policy.register_store``) divides the row and all three
        # pointers are aligned to it. With STORE > 1 lane l moves VEC * l ..
        # VEC * l + VEC - 1 in vectors of STORE elements, where the network
        # leaves positions; with STORE == 1 (rung 32, odd rows or misaligned
        # pointers) it loads element 32 * i + l into register i (coalesced).
        # Padding words get the maximal ordinal and their index, so they sort
        # last.
        STORE = const_expr(self.store)
        keys = cute.make_rmem_tensor(VEC, word_t)
        index = cute.make_rmem_tensor(VEC, Int32)
        for c in cutlass.range_constexpr(VEC // STORE):
            first = Int32(32 * c) + lane
            if const_expr(STORE > 1):
                first = lane * Int32(VEC) + Int32(c * STORE)
            raw = cute.make_rmem_tensor(STORE, self.raw_dtype)
            valid = row_valid and first < n
            if valid:
                offset = cute.assume(Int64(row) * Int64(n) + Int64(first), divby=STORE)
                src = cute.make_tensor(mX.iterator + offset, cute.make_layout(STORE))
                cute.autovec_copy(src, raw)
            for t in cutlass.range_constexpr(STORE):
                i = c * STORE + t
                j = first + Int32(t)
                if const_expr(wide):
                    keys[i] = Int64((1 << 63) - 1)
                    index[i] = j
                else:
                    keys[i] = word_t(((1 << self.key_bits) - 1) << IDX_BITS) | word_t(j)
                if valid:
                    ord_ = self.ordinal(bits_t(raw[t]))
                    if const_expr(wide):
                        keys[i] = ord_ ^ Int64(-(1 << 63))
                    elif const_expr(self.key_bits == 32):
                        ord64 = Int64(ord_) & Int64(0xFFFFFFFF)
                        keys[i] = (ord64 << IDX_BITS) | Int64(j)
                    else:
                        keys[i] = (ord_ << IDX_BITS) | j

        if const_expr(self.key_bits + IDX_BITS <= 16 and VEC >= 2):
            self.packed_network(keys, lane)
        else:
            self.network(keys, index if const_expr(wide) else None, lane)

        # Scalar stores: swap the low M register bits with the low M lane
        # bits, so that each store writes runs of 2^M consecutive positions.
        M = const_expr(min(VEC.bit_length() - 1, 5))
        if const_expr(STORE == 1):
            for k in cutlass.range_constexpr(M):
                upper = (lane & Int32(1 << k)) != Int32(0)
                for i in cutlass.range_constexpr(VEC):
                    if const_expr(i & (1 << k) == 0):
                        h = i | (1 << k)
                        for regs in (keys, index) if const_expr(wide) else (keys,):
                            send = regs[h]
                            if upper:
                                send = regs[i]
                            got = cute.arch.shuffle_sync_bfly(send, offset=1 << k)
                            if upper:
                                regs[i] = got
                            else:
                                regs[h] = got

        # Invert the ordinal; zeros and NaN (comparison order) are re-read.
        full = const_expr(self.key_bits >= 32)
        bits_mask = const_expr(-1 if full else (1 << self.key_bits) - 1)
        sign = const_expr(
            -(1 << (self.key_bits - 1)) if full else 1 << (self.key_bits - 1)
        )
        magnitude_mask = const_expr((1 << (self.key_bits - 1)) - 1)
        for c in cutlass.range_constexpr(VEC // STORE):
            first = lane * Int32(VEC) + Int32(c * STORE)
            if const_expr(STORE == 1):
                low = (1 << M) - 1
                first = (
                    (lane & Int32(low))
                    | Int32(((c >> M) << M) | ((c & low) << LOG_VEC))
                    | ((lane >> Int32(M)) << Int32(LOG_VEC + M))
                )
            if row_valid and first < n:
                values = cute.make_rmem_tensor(STORE, self.raw_dtype)
                indices = cute.make_rmem_tensor(STORE, self.index_dtype)
                for t in cutlass.range_constexpr(STORE):
                    i = c * STORE + t
                    if const_expr(wide):
                        src = index[i]
                        ord_ = keys[i] ^ Int64(-(1 << 63))
                    else:
                        src = Int32(keys[i] & word_t((1 << IDX_BITS) - 1))
                        ord_ = bits_t(keys[i] >> IDX_BITS)
                    if const_expr(self.descending):
                        ord_ = ~ord_
                    bits = ord_
                    if const_expr(self.kind == "signed"):
                        bits = ord_ ^ bits_t(sign)
                    elif const_expr(self.kind == "float"):
                        bits = ~ord_
                        if ord_ & bits_t(sign) != bits_t(0):
                            bits = ord_ ^ bits_t(sign)
                        magnitude = bits & bits_t(magnitude_mask)
                        if magnitude == bits_t(0) or (
                            const_expr(self.cmp) and magnitude == bits_t(magnitude_mask)
                        ):
                            bits = bits_t(mX[row, src])
                    values[t] = self.raw_dtype(bits & bits_t(bits_mask))
                    indices[t] = self.index_dtype(src)
                offset = cute.assume(Int64(row) * Int64(n) + Int64(first), divby=STORE)
                for out, frag in ((mValues, values), (mIndices, indices)):
                    dst = cute.make_tensor(
                        out.iterator + offset, cute.make_layout(STORE)
                    )
                    cute.autovec_copy(frag, dst)


def spec_name(spec: dict) -> str:
    return (
        f"sort_register{spec['rung']}_{spec['key']}_{spec['index']}"
        f"_{'desc' if spec['descending'] else 'asc'}_r{spec['rows']}"
        f"_v{spec['store']}" + (f"_sm{spec['arch']}" if "arch" in spec else "")
    )


def make_spec(
    arch: int,
    key_dtype: torch.dtype,
    index_dtype: torch.dtype,
    descending: bool,
    nsort: int,
    store: int,
) -> dict:
    return {
        "kernel": "register",
        "key": index_name(key_dtype),
        "index": index_name(index_dtype),
        "descending": descending,
        "rung": register_rung(nsort),
        "rows": REGISTER_ROWS_PER_CTA,
        "store": store,
        "arch": arch,
    }


def compile_args(spec: dict) -> tuple[RegisterSort, list]:
    """Kernel object and fake operands for one spec; shared by JIT and AOT."""
    raw = RAW_CUTE[KEY_KINDS[getattr(torch, spec["key"])][1]]
    index = INDEX_DTYPES[getattr(torch, spec["index"])]
    store = spec["store"]
    kernel = RegisterSort(
        spec["key"], index, spec["descending"], spec["rung"], spec["rows"], store
    )
    fake = [
        fake_compact(
            t,
            (sym(), sym()),
            stride_order=(1, 0),
            align=vector_alignment(store, t.width // 8),
        )
        for t in (raw, raw, index)
    ] + [
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    ]
    return kernel, fake


@instrumented_cutedsl_cache("aten::sort", key_fn=lambda spec: spec_name(dict(spec)))
def _compile(spec: tuple):
    kernel, fake = compile_args(dict(spec))
    return compile_kernel(kernel, *fake)


def register_sort(
    x: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
    descending: bool,
    arch: int,
) -> None:
    """Stable sort of each row of 2D row-major ``x`` into ``values`` and
    ``indices``; also a valid unstable sort.

    The caller checks eligibility: key and index dtypes supported, row length
    at most ``policy.REGISTER_MAX_NSORT`` and addressable by the index dtype,
    contiguous outputs.
    """
    if x.numel() == 0:
        return
    nsort = x.shape[1]
    operands = [(const_data_ptr(x), x.element_size())] + [
        (t.data_ptr(), t.element_size()) for t in (values, indices)
    ]
    store = register_store(nsort, operands)
    spec = make_spec(arch, x.dtype, indices.dtype, descending, nsort, store)
    _compile(tuple(sorted(spec.items())))(
        read_only(raw_view(x)), raw_view(values), indices
    )
