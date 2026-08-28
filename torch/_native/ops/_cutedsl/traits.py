# CuTeDSL trait library for native reductions: the trait protocol + cross-thread
# reduce helpers (warp_reduce / block_reduce). Shared machinery under _cutedsl/ so
# future pointwise ops can reuse the same abstractions. The kernels and the aten
# overrides live in ../reductions/.
#
# Input is 2D (M, N); reduce along the contiguous last axis, producing one
# output per row. One CTA handles one row; the whole reduction fits in a block
# via warp-shuffle butterfly + a shared-memory block reduce.
#
# THREE value methods, and the split between them is what lets any trait ride any fold order:
#   leaf(val, idx) -> one ELEMENT as a standalone accumulator (|x|**p for a norm, a 0/1 flag
#                     for all/any, (value, position) for argmax, (x, 0, 1) for Welford).
#   combine(a, b)  -> merge two accumulators. Associative, so a fold may associate freely.
#   reduce(acc, val, idx, valid) -> the SERIAL update, which fuses the two for the rolled
#                     folds and may use a cheaper online formula (Welford's is not combine's).
# A tree fold cannot use `reduce`: it needs each contribution on its own before it can pair
# them up. Anything that transforms an element therefore has to say so in `leaf`, or a tree
# order silently folds raw values (this was a real bug -- norm became a plain sum).
#
# ACCUMULATOR DTYPE is a PARAMETER (`acc`, default Float32). Every trait threads it
# through fdtypes + its init/reduce/combine/project literals so the SAME trait can
# accumulate in fp32 (fp16/bf16/fp32 inputs) or a wider type (fp64) -- and, later,
# integer/complex accumulators. The accumulator identity (0 / 1 / +-inf) is taken
# from the dtype via _zero/_one/_pos_id/_neg_id so it is correct for any acc dtype
# (e.g. ints have no inf; they will supply min/max int instead). The argmax/argmin
# INDEX field is Int32 by default (a position, not an accumuland) and Int64 when the
# reduced extent can exceed 2^31 -- both threaded via the trait's `idx` parameter.
#
# Discovered cute intrinsics (the productive ones):
#   scalar select  : Python ternary (a if pred else b) INSIDE a @cute.jit body;
#                    the jit preprocessor lowers it to a select. It does NOT
#                    lower inside plain (undecorated) callees, so every trait
#                    value-method is @cute.jit.
#   isnan(x)       : x != x  (Boolean SSA)
#   -inf           : -<acc>.inf
#   abs            : cute.math.absf
#   pow / sqrt     : cute.math.exp/log for x**p; cute.math.sqrt for sqrt
#   fmax           : cute.arch.fmax  (NaN-SUPPRESSING: returns the non-NaN arg,
#                    so argmax NaN handling is done explicitly via x != x)
#   butterfly shfl : cute.arch.shuffle_sync_bfly(value, offset=...)

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, const_expr, Float32, Int32, Int64


WARP = 32

# argmax/argmin "no winner yet" sentinel, per index dtype. Int32 is the default (an
# index is a position, not an accumuland, and the narrow field halves partial-buffer
# traffic + speeds the warp shuffle); the builder switches an index trait to Int64
# only when the reduced extent can exceed the Int32 range (see _idx_sentinel).
_INT32_MAX = (1 << 31) - 1
_INT64_MAX = (1 << 63) - 1


def _idx_sentinel(idx_dtype):
    return idx_dtype(_INT64_MAX if idx_dtype is Int64 else _INT32_MAX)


def _pos_id(acc):
    # "Largest" identity for a min-reduction's init / the value that loses every
    # max. Floats only here: an integer accumulator has no .inf, so the commit that
    # serves those dtypes extends this with explicit arms. Wrap in `acc(...)` so the
    # result
    # carries the accumulator dtype -- `acc.inf` is a bare Python float, which the
    # DSL treats as Float32, breaking the ifexp type-match in `_pick` for fp64.
    # max. +inf for floats; the max representable value for integer accumulators
    # (Int32/Int64 have no .inf). Wrap in `acc(...)` so the result carries the
    # accumulator dtype -- a bare Python number would be treated as Float32,
    # breaking the ifexp type-match in `_pick` for fp64.
    if acc is Int32:
        return acc(_INT32_MAX)
    if acc is Int64:
        return acc(_INT64_MAX)
    return acc(acc.inf)


def _neg_id(acc):
    # "Smallest" identity for a max-reduction's init: -inf for floats, the min
    # representable value for integer accumulators; typed via `acc(...)` as above.
    if acc is Int32:
        return acc(-_INT32_MAX - 1)
    if acc is Int64:
        return acc(-_INT64_MAX - 1)
    return acc(-acc.inf)


class SumOps:
    # acc = (sum,). Validates vs torch.sum(x, dim=-1).
    nfields = 1

    def __init__(self, acc=Float32):
        self.acc = acc
        self.fdtypes = (acc,)

    def init(self):
        return (self.acc(0.0),)

    @cute.jit
    def leaf(self, val, idx):
        # This element's contribution as a standalone accumulator: the element itself.
        return (self.acc(val),)

    @cute.jit
    def reduce(self, acc, val, idx, valid):
        add = val if valid else self.acc(0.0)
        return (acc[0] + add,)

    @cute.jit
    def combine(self, a, b):
        return (a[0] + b[0],)

    @cute.jit
    def shfl_down(self, acc, offset):
        return (cute.arch.shuffle_sync_bfly(acc[0], offset=offset),)

    @cute.jit
    def project(self, acc, n):
        return acc[0]


class NormOps:
    # acc = (sum of |x|**p,). project = sum**(1/p).
    # Validates vs torch.linalg.vector_norm(x, ord=p, dim=-1).
    nfields = 1

    def __init__(self, p, acc=Float32):
        self.p = float(p)
        self.acc = acc
        self.fdtypes = (acc,)

    @cute.jit
    def _absp(self, val):
        a = self.acc(cute.math.absf(val))
        if const_expr(self.p == 1.0):
            return a
        elif const_expr(self.p == 2.0):
            return a * a
        else:
            # |x|**p via exp(p*log|x|); log(0)=-inf so 0**p -> exp(-inf)=0 for p>0.
            return cute.math.exp(self.acc(self.p) * cute.math.log(a))

    def init(self):
        return (self.acc(0.0),)

    @cute.jit
    def leaf(self, val, idx):
        # This element's contribution as a standalone accumulator: |x|**p.
        return (self._absp(val),)

    @cute.jit
    def reduce(self, acc, val, idx, valid):
        contrib = self._absp(val) if valid else self.acc(0.0)
        return (acc[0] + contrib,)

    @cute.jit
    def combine(self, a, b):
        return (a[0] + b[0],)

    @cute.jit
    def shfl_down(self, acc, offset):
        return (cute.arch.shuffle_sync_bfly(acc[0], offset=offset),)

    @cute.jit
    def project(self, acc, n):
        s = acc[0]
        if const_expr(self.p == 1.0):
            return s
        elif const_expr(self.p == 2.0):
            return cute.math.sqrt(s)
        else:
            return cute.math.exp(cute.math.log(s) / self.acc(self.p))


@cute.jit
def _welford_denom(acc_dtype, nf, correction):
    # var/std divisor, CLAMPED AT ZERO like aten: `correction >= n` must divide by 0
    # (-> +inf, which is what aten returns and what the numpy-reference tests expect
    # after their inf->nan mapping), NOT by a negative number. Unclamped, a
    # correction larger than the reduced extent returned a NEGATIVE variance.
    # `nf` is a runtime value, so this is a select, not a python max().
    d = nf - acc_dtype(correction)
    z = acc_dtype(0.0)
    return d if d > z else z  # noqa: FURB136 -- builtin max()/min() do not lower


class WelfordOps:
    # acc = (mean, m2, nf) all in the accumulator dtype.
    #   reduce  = ONLINE (Welford) update of a single element.
    #   combine = PARALLEL (Chan) merge of two partial accumulators.
    # These two formulas are deliberately different. project computes
    #   var = m2 / max(nf - correction, 0), optionally sqrt, optionally mean.
    # Validates vs torch.var / torch.std (dim=-1, correction).
    nfields = 3

    def __init__(self, correction=1, take_sqrt=False, return_mean=False, acc=Float32):
        self.correction = float(correction)
        self.take_sqrt = bool(take_sqrt)
        self.return_mean = bool(return_mean)
        self.acc = acc
        self.fdtypes = (acc, acc, acc)

    def init(self):
        z = self.acc(0.0)
        return (z, z, z)

    @cute.jit
    def leaf(self, val, idx):
        # This element's contribution as a standalone accumulator: a one-element accumulator (mean = x, m2 = 0, count = 1).
        return (self.acc(val), self.acc(0.0), self.acc(1.0))

    @cute.jit
    def reduce(self, acc, val, idx, valid):
        mean, m2, nf = acc
        new_nf = nf + self.acc(1.0)
        delta = val - mean
        new_mean = mean + delta / new_nf
        new_m2 = m2 + delta * (val - new_mean)
        # OOB element: keep accumulator unchanged (do not advance the count).
        out_mean = new_mean if valid else mean
        out_m2 = new_m2 if valid else m2
        out_nf = new_nf if valid else nf
        return (out_mean, out_m2, out_nf)

    @cute.jit
    def combine(self, a, b):
        ma, m2a, na = a
        mb, m2b, nb = b
        nn = na + nb
        nb_over_n = (nb / nn) if (nn > self.acc(0.0)) else self.acc(0.0)
        delta = mb - ma
        mean = ma + delta * nb_over_n
        m2 = m2a + m2b + delta * delta * na * nb_over_n
        return (mean, m2, nn)

    @cute.jit
    def shfl_down(self, acc, offset):
        return (
            cute.arch.shuffle_sync_bfly(acc[0], offset=offset),
            cute.arch.shuffle_sync_bfly(acc[1], offset=offset),
            cute.arch.shuffle_sync_bfly(acc[2], offset=offset),
        )

    @cute.jit
    def project(self, acc, n):
        mean, m2, nf = acc
        if const_expr(self.return_mean):
            return mean
        var = m2 / _welford_denom(self.acc, nf, self.correction)
        if const_expr(self.take_sqrt):
            return cute.math.sqrt(var)
        return var


class ArgMaxOps:
    # acc = (best_val: acc dtype, best_idx: idx dtype, default Int32).
    # GreaterOrNan winner: NaN beats everything; exact tie -> LOWER index wins;
    # otherwise larger value wins. Matches torch.argmax (first NaN, first max).
    # has_index: the accumulator carries a per-element INDEX. The index dtype is a
    # parameter: Int32 by default (cheaper partials + shuffle), Int64 when the reduced
    # extent can exceed 2^31 so the winning position never overflows -- the builder
    # picks it from N (this is what lets the cross-CTA split serve huge-N argmax; the
    # stage-1 fold's chunk-global column must be computed in the same width, see
    # tile.TileMap.col_base).
    nfields = 2
    has_index = True

    def __init__(self, acc=Float32, idx=Int32):
        self.acc = acc
        self.idx = idx
        self.fdtypes = (acc, idx)

    def init(self):
        return (_neg_id(self.acc), _idx_sentinel(self.idx))

    @cute.jit
    def _pick(self, bv, bi, cv, ci):
        # Does candidate (cv, ci) beat current best (bv, bi)?
        cand_nan = cv != cv
        best_nan = bv != bv
        repl = (
            ((ci < bi) if best_nan else Boolean(True))
            if cand_nan
            else (
                Boolean(False) if best_nan else ((ci < bi) if (cv == bv) else (cv > bv))
            )
        )
        nv = cv if repl else bv
        ni = ci if repl else bi
        return (nv, ni)

    @cute.jit
    def leaf(self, val, idx):
        # This element's contribution as a standalone accumulator: the (value, position) pair; `combine` does the picking.
        return (self.acc(val), self.idx(idx))

    @cute.jit
    def reduce(self, acc, val, idx, valid):
        nv, ni = self._pick(acc[0], acc[1], val, self.idx(idx))
        out_v = nv if valid else acc[0]
        out_i = ni if valid else acc[1]
        return (out_v, out_i)

    @cute.jit
    def combine(self, a, b):
        return self._pick(a[0], a[1], b[0], b[1])

    @cute.jit
    def shfl_down(self, acc, offset):
        return (
            cute.arch.shuffle_sync_bfly(acc[0], offset=offset),
            cute.arch.shuffle_sync_bfly(acc[1], offset=offset),
        )

    @cute.jit
    def project(self, acc, n):
        return acc[1]


class ProdOps:
    # acc = (product,). Validates vs torch.prod(x, dim=-1).
    nfields = 1

    def __init__(self, acc=Float32):
        self.acc = acc
        self.fdtypes = (acc,)

    def init(self):
        return (self.acc(1.0),)

    @cute.jit
    def leaf(self, val, idx):
        # This element's contribution as a standalone accumulator: the element itself.
        return (self.acc(val),)

    @cute.jit
    def reduce(self, acc, val, idx, valid):
        mul = val if valid else self.acc(1.0)
        return (acc[0] * mul,)

    @cute.jit
    def combine(self, a, b):
        return (a[0] * b[0],)

    @cute.jit
    def shfl_down(self, acc, offset):
        return (cute.arch.shuffle_sync_bfly(acc[0], offset=offset),)

    @cute.jit
    def project(self, acc, n):
        return acc[0]


class MeanOps:
    # acc = (sum,); factor applied in project. Validates vs torch.mean(x, dim=-1).
    nfields = 1

    def __init__(self, acc=Float32):
        self.acc = acc
        self.fdtypes = (acc,)

    def init(self):
        return (self.acc(0.0),)

    @cute.jit
    def leaf(self, val, idx):
        # This element's contribution as a standalone accumulator: the element itself.
        return (self.acc(val),)

    @cute.jit
    def reduce(self, acc, val, idx, valid):
        add = val if valid else self.acc(0.0)
        return (acc[0] + add,)

    @cute.jit
    def combine(self, a, b):
        return (a[0] + b[0],)

    @cute.jit
    def shfl_down(self, acc, offset):
        return (cute.arch.shuffle_sync_bfly(acc[0], offset=offset),)

    @cute.jit
    def project(self, acc, n):
        return acc[0] / n


class NanSumOps:
    # acc = (sum,); NaN inputs map to 0. Validates vs torch.nansum(x, dim=-1).
    nfields = 1

    def __init__(self, acc=Float32):
        self.acc = acc
        self.fdtypes = (acc,)

    def init(self):
        return (self.acc(0.0),)

    @cute.jit
    def leaf(self, val, idx):
        # This element's contribution as a standalone accumulator: 0 for a NaN, the element otherwise.
        return (self.acc(val) if (val == val) else self.acc(0.0),)

    @cute.jit
    def reduce(self, acc, val, idx, valid):
        clean = val if (val == val) else self.acc(0.0)
        add = clean if valid else self.acc(0.0)
        return (acc[0] + add,)

    @cute.jit
    def combine(self, a, b):
        return (a[0] + b[0],)

    @cute.jit
    def shfl_down(self, acc, offset):
        return (cute.arch.shuffle_sync_bfly(acc[0], offset=offset),)

    @cute.jit
    def project(self, acc, n):
        return acc[0]


class AllOps:
    # acc = (1.0 while all-true,). x != 0 is True (NaN is truthy, matches torch).
    # AND via product of 0/1 flags. Validates vs torch.all(x, dim=-1).
    nfields = 1

    def __init__(self, acc=Float32):
        self.acc = acc
        self.fdtypes = (acc,)

    def init(self):
        return (self.acc(1.0),)

    @cute.jit
    def leaf(self, val, idx):
        # This element's contribution as a standalone accumulator: the 0/1 truth flag.
        return (self.acc(1.0) if (val != self.acc(0.0)) else self.acc(0.0),)

    @cute.jit
    def reduce(self, acc, val, idx, valid):
        flag = self.acc(1.0) if (val != self.acc(0.0)) else self.acc(0.0)
        keep = flag if valid else self.acc(1.0)
        return (acc[0] * keep,)

    @cute.jit
    def combine(self, a, b):
        return (a[0] * b[0],)

    @cute.jit
    def shfl_down(self, acc, offset):
        return (cute.arch.shuffle_sync_bfly(acc[0], offset=offset),)

    @cute.jit
    def project(self, acc, n):
        return acc[0]


class AnyOps:
    # acc = (1.0 if any-true,). OR via max of 0/1 flags. Validates vs torch.any.
    nfields = 1

    def __init__(self, acc=Float32):
        self.acc = acc
        self.fdtypes = (acc,)

    def init(self):
        return (self.acc(0.0),)

    @cute.jit
    def leaf(self, val, idx):
        # This element's contribution as a standalone accumulator: the 0/1 truth flag.
        return (self.acc(1.0) if (val != self.acc(0.0)) else self.acc(0.0),)

    @cute.jit
    def reduce(self, acc, val, idx, valid):
        flag = self.acc(1.0) if (val != self.acc(0.0)) else self.acc(0.0)
        keep = flag if valid else self.acc(0.0)
        return (max(keep, acc[0]),)

    @cute.jit
    def combine(self, a, b):
        return (max(b[0], a[0]),)

    @cute.jit
    def shfl_down(self, acc, offset):
        return (cute.arch.shuffle_sync_bfly(acc[0], offset=offset),)

    @cute.jit
    def project(self, acc, n):
        return acc[0]


class CountNonzeroOps:
    # acc = (count,). Also serves p=0 norm. Validates vs torch.count_nonzero and
    # torch.linalg.vector_norm(x, ord=0, dim=-1). NaN counts as nonzero.
    nfields = 1

    def __init__(self, acc=Float32):
        self.acc = acc
        self.fdtypes = (acc,)

    def init(self):
        return (self.acc(0.0),)

    @cute.jit
    def leaf(self, val, idx):
        # This element's contribution as a standalone accumulator: the 0/1 truth flag.
        return (self.acc(1.0) if (val != self.acc(0.0)) else self.acc(0.0),)

    @cute.jit
    def reduce(self, acc, val, idx, valid):
        flag = self.acc(1.0) if (val != self.acc(0.0)) else self.acc(0.0)
        add = flag if valid else self.acc(0.0)
        return (acc[0] + add,)

    @cute.jit
    def combine(self, a, b):
        return (a[0] + b[0],)

    @cute.jit
    def shfl_down(self, acc, offset):
        return (cute.arch.shuffle_sync_bfly(acc[0], offset=offset),)

    @cute.jit
    def project(self, acc, n):
        return acc[0]


class AbsMaxOps:
    # acc = (max|x|,). p=inf norm. Validates vs vector_norm(x, ord=inf, dim=-1).
    nfields = 1

    def __init__(self, acc=Float32):
        self.acc = acc
        self.fdtypes = (acc,)

    def init(self):
        return (self.acc(0.0),)

    @cute.jit
    def leaf(self, val, idx):
        # This element's contribution as a standalone accumulator: |x|.
        return (self.acc(cute.math.absf(val)),)

    @cute.jit
    def reduce(self, acc, val, idx, valid):
        a = self.acc(cute.math.absf(val))
        m = max(acc[0], a)
        return (m if valid else acc[0],)

    @cute.jit
    def combine(self, a, b):
        return (max(b[0], a[0]),)

    @cute.jit
    def shfl_down(self, acc, offset):
        return (cute.arch.shuffle_sync_bfly(acc[0], offset=offset),)

    @cute.jit
    def project(self, acc, n):
        return acc[0]


class AbsMinOps:
    # acc = (min|x|,). p=-inf norm. Validates vs vector_norm(x, ord=-inf, dim=-1).
    nfields = 1

    def __init__(self, acc=Float32):
        self.acc = acc
        self.fdtypes = (acc,)

    def init(self):
        return (_pos_id(self.acc),)

    @cute.jit
    def leaf(self, val, idx):
        # This element's contribution as a standalone accumulator: |x|.
        return (self.acc(cute.math.absf(val)),)

    @cute.jit
    def reduce(self, acc, val, idx, valid):
        a = self.acc(cute.math.absf(val))
        m = min(acc[0], a)
        return (m if valid else acc[0],)

    @cute.jit
    def combine(self, a, b):
        return (min(b[0], a[0]),)

    @cute.jit
    def shfl_down(self, acc, offset):
        return (cute.arch.shuffle_sync_bfly(acc[0], offset=offset),)

    @cute.jit
    def project(self, acc, n):
        return acc[0]


class ArgMinOps:
    # acc = (best_val, best_idx). LessOrNan: NaN beats everything; tie -> lower
    # index; else smaller value wins. Matches torch.argmin.
    nfields = 2
    has_index = (
        True  # index dtype parametric (Int32 default / Int64 huge-N); see ArgMaxOps
    )

    def __init__(self, acc=Float32, idx=Int32):
        self.acc = acc
        self.idx = idx
        self.fdtypes = (acc, idx)

    def init(self):
        return (_pos_id(self.acc), _idx_sentinel(self.idx))

    @cute.jit
    def _pick(self, bv, bi, cv, ci):
        cand_nan = cv != cv
        best_nan = bv != bv
        repl = (
            ((ci < bi) if best_nan else Boolean(True))
            if cand_nan
            else (
                Boolean(False) if best_nan else ((ci < bi) if (cv == bv) else (cv < bv))
            )
        )
        nv = cv if repl else bv
        ni = ci if repl else bi
        return (nv, ni)

    @cute.jit
    def leaf(self, val, idx):
        # This element's contribution as a standalone accumulator: the (value, position) pair; `combine` does the picking.
        return (self.acc(val), self.idx(idx))

    @cute.jit
    def reduce(self, acc, val, idx, valid):
        nv, ni = self._pick(acc[0], acc[1], val, self.idx(idx))
        out_v = nv if valid else acc[0]
        out_i = ni if valid else acc[1]
        return (out_v, out_i)

    @cute.jit
    def combine(self, a, b):
        return self._pick(a[0], a[1], b[0], b[1])

    @cute.jit
    def shfl_down(self, acc, offset):
        return (
            cute.arch.shuffle_sync_bfly(acc[0], offset=offset),
            cute.arch.shuffle_sync_bfly(acc[1], offset=offset),
        )

    @cute.jit
    def project(self, acc, n):
        return acc[1]


class AMaxOps:
    # acc = (max,). Pure single-field NaN-propagating max -- amax returns only the
    # value, so it does NOT carry the index accumulator argmax needs. Keeping it
    # 1-field (vs subclassing ArgMaxOps) halves the warp-shuffle / smem traffic and,
    # because it is not has_index, lets the dispatcher take the cross-CTA fast path
    # at huge N. NaN propagates: if either operand is NaN the result is NaN (matches
    # torch.amax). Validates vs torch.amax(x, dim=-1).
    nfields = 1

    def __init__(self, acc=Float32):
        self.acc = acc
        self.fdtypes = (acc,)

    def init(self):
        return (_neg_id(self.acc),)

    @cute.jit
    def _maxnan(self, a, b):
        # NaN-propagating max: b if (b > a or b is NaN) else a. b != b detects NaN.
        return b if ((b > a) or (b != b)) else a

    @cute.jit
    def leaf(self, val, idx):
        # This element's contribution as a standalone accumulator: the element itself.
        return (self.acc(val),)

    @cute.jit
    def reduce(self, acc, val, idx, valid):
        m = self._maxnan(acc[0], val)
        return (m if valid else acc[0],)

    @cute.jit
    def combine(self, a, b):
        return (self._maxnan(a[0], b[0]),)

    @cute.jit
    def shfl_down(self, acc, offset):
        return (cute.arch.shuffle_sync_bfly(acc[0], offset=offset),)

    @cute.jit
    def project(self, acc, n):
        return acc[0]


class AMinOps:
    # acc = (min,). Pure single-field NaN-propagating min (the AMaxOps mirror).
    # Validates vs torch.amin(x, dim=-1).
    nfields = 1

    def __init__(self, acc=Float32):
        self.acc = acc
        self.fdtypes = (acc,)

    def init(self):
        return (_pos_id(self.acc),)

    @cute.jit
    def _minnan(self, a, b):
        return b if ((b < a) or (b != b)) else a

    @cute.jit
    def leaf(self, val, idx):
        # This element's contribution as a standalone accumulator: the element itself.
        return (self.acc(val),)

    @cute.jit
    def reduce(self, acc, val, idx, valid):
        m = self._minnan(acc[0], val)
        return (m if valid else acc[0],)

    @cute.jit
    def combine(self, a, b):
        return (self._minnan(a[0], b[0]),)

    @cute.jit
    def shfl_down(self, acc, offset):
        return (cute.arch.shuffle_sync_bfly(acc[0], offset=offset),)

    @cute.jit
    def project(self, acc, n):
        return acc[0]


def _offsets(threads_per_row, ascending: bool = False):
    # Decreasing butterfly offsets: matches the PyTorch/Triton fp reduction order.
    # ASCENDING (1, 2, 4, ...) is ATen's WarpReduceDirection::ASCENDING, which the tile
    # datapath's lane merge uses. Same result mathematically, different add order ->
    # different fp bits, so the direction is part of a kernel's numerics contract.
    n = min(threads_per_row, WARP)
    offs = []
    if ascending:
        o = 1
        while o < n:
            offs.append(o)
            o = o * 2
    else:
        o = n // 2
        while o > 0:
            offs.append(o)
            o = o // 2
    return offs


@cute.jit
def warp_reduce(
    trait,
    acc,
    threads_per_row: cutlass.Constexpr,
    ascending: cutlass.Constexpr = False,
):
    for offset in _offsets(threads_per_row, ascending):
        acc = trait.combine(acc, trait.shfl_down(acc, offset))
    return acc


@cute.jit
def block_reduce(
    trait,
    acc,
    bufs,
    warps_per_row: cutlass.Constexpr,
    rows_per_block: cutlass.Constexpr = 1,
):
    # Cross-warp reduction WITHIN each row's warp group. A block may hold
    # rows_per_block rows, each spanning warps_per_row warps; warp w belongs to
    # row (w // warps_per_row) at group position (w % warps_per_row). Each row
    # reduces its own group independently -- mixing groups (the old flat version)
    # corrupted multi-row blocks. Smem bufs are (rows_per_block, warps_per_row)
    # per field; lane 0 of each warp writes its group slot, then the group's
    # position-0 warp re-reduces its row's warps_per_row partials.
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    row_g = warp // warps_per_row
    col_g = warp % warps_per_row
    if lane == 0:
        for f in cutlass.range_constexpr(trait.nfields):
            bufs[f][row_g * const_expr(warps_per_row) + col_g] = acc[f]
    cute.arch.barrier()
    # Every warp re-loads its OWN row's group (so all lanes of all warps get the
    # reduced value, matching the broadcast the row kernel's lane-0 store expects).
    out = trait.init()
    if lane < warps_per_row:
        out = tuple(
            bufs[f][row_g * const_expr(warps_per_row) + lane]
            for f in range(trait.nfields)
        )
    out = warp_reduce(trait, out, warps_per_row)
    return out


# --- Two-output traits (var_mean / std_mean, max.dim / min.dim, aminmax). They
# reuse the single-output accumulators above and only change project() to return
# a tuple of `nouts` values; nouts (1 or 2) tells the kernel how many outputs to
# store. ---


class VarMeanOps(WelfordOps):
    # Reuse the validated Welford accumulator (mean, m2, nf); project BOTH the
    # variance/std AND the mean. correction and take_sqrt behave as in the base.
    # Validates vs torch.var_mean / torch.std_mean (dim=-1, correction).
    nouts = 2

    def __init__(self, correction=1, take_sqrt=False, acc=Float32):
        super().__init__(correction=correction, take_sqrt=take_sqrt, acc=acc)

    @cute.jit
    def project(self, acc, n):
        mean, m2, nf = acc
        var = m2 / _welford_denom(self.acc, nf, self.correction)
        result = cute.math.sqrt(var) if const_expr(self.take_sqrt) else var
        return (result, mean)


class MaxDimOps(ArgMaxOps):
    # Reuse the GreaterOrNan winner logic (NaN propagates, lowest index on tie);
    # project BOTH the winning value and its index. Validates vs torch.max(dim).
    nouts = 2

    @cute.jit
    def project(self, acc, n):
        return (acc[0], acc[1])


class MinDimOps(ArgMinOps):
    # LessOrNan winner; project (value, index). Validates vs torch.min(dim).
    nouts = 2

    @cute.jit
    def project(self, acc, n):
        return (acc[0], acc[1])


class AMinMaxOps:
    # acc = (min_val, max_val), both the accumulator dtype. reduce/combine track
    # both extremes; project returns (min, max). NaN-propagating like torch.aminmax
    # (any NaN in the row makes both outputs NaN). Validates vs torch.aminmax.
    nfields = 2
    nouts = 2

    def __init__(self, acc=Float32):
        self.acc = acc
        self.fdtypes = (acc, acc)

    def init(self):
        return (_pos_id(self.acc), _neg_id(self.acc))

    @cute.jit
    def _fmin(self, a, b):
        # NaN-propagating min: if either is NaN, result is NaN. Keep the explicit
        # ternary -- in a @cute.jit body it lowers to a cute select over SSA
        # scalars; Python's min()/max() (RUFF FURB136) would NOT lower the same way.
        return (a if a != a else (a if a < b else b)) if b == b else b  # noqa: FURB136

    @cute.jit
    def _fmax(self, a, b):
        return (a if a != a else (a if a > b else b)) if b == b else b  # noqa: FURB136

    @cute.jit
    def leaf(self, val, idx):
        # This element's contribution as a standalone accumulator: the element as both extremes.
        return (self.acc(val), self.acc(val))

    @cute.jit
    def reduce(self, acc, val, idx, valid):
        lo = self._fmin(acc[0], val)
        hi = self._fmax(acc[1], val)
        out_lo = lo if valid else acc[0]
        out_hi = hi if valid else acc[1]
        return (out_lo, out_hi)

    @cute.jit
    def combine(self, a, b):
        return (self._fmin(a[0], b[0]), self._fmax(a[1], b[1]))

    @cute.jit
    def shfl_down(self, acc, offset):
        return (
            cute.arch.shuffle_sync_bfly(acc[0], offset=offset),
            cute.arch.shuffle_sync_bfly(acc[1], offset=offset),
        )

    @cute.jit
    def project(self, acc, n):
        return (acc[0], acc[1])
