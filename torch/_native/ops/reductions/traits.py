# Reusable CuteDSL reduction traits and cross-thread helpers. leaf creates a one-element
# accumulator, combine merges accumulators, and reduce performs a serial update. Tree folds
# use leaf/combine, so leaf must include every element transform. Other names mirror
# SharedReduceOps.h. Accumulator dtypes provide their identities. Scalar conditionals lower
# only inside cute.jit methods; cute.arch.fmax suppresses NaNs, so traits handle them.

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, const_expr, Float32, Int32, Int64


WARP = 32

# Arg-reduction "no winner" sentinel. Use Int32 to reduce partial traffic unless the
# extent requires Int64.
_INT32_MAX = (1 << 31) - 1
_INT64_MAX = (1 << 63) - 1


def _idx_sentinel(idx_dtype):
    return idx_dtype(_INT64_MAX if idx_dtype is Int64 else _INT32_MAX)


def _pos_id(acc):
    # A typed +inf or integer maximum; bare Python values become Float32 and break fp64 ifexp.
    if acc is Int32:
        return acc(_INT32_MAX)
    if acc is Int64:
        return acc(_INT64_MAX)
    return acc(acc.inf)


def _neg_id(acc):
    # Typed -inf or integer minimum for max-reduction initialization.
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
    # acc = (sum(|x|**p),); project raises it to 1/p.
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
    # Clamp ATen's var/std divisor at zero: correction >= n yields +inf, not negative
    # variance. nf is runtime, so use a select rather than Python max().
    d = nf - acc_dtype(correction)
    z = acc_dtype(0.0)
    return d if d > z else z  # noqa: FURB136 -- see the note above _maxnan


class WelfordOps:
    # acc = (mean, m2, nf). `reduce` is the ONLINE Welford update and `combine` the PARALLEL Chan
    # merge -- deliberately different formulas. project divides m2 by the clamped dof.
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
        return (self.acc(val), self.acc(0.0), self.acc(1.0))

    @cute.jit
    def reduce(self, acc, val, idx, valid):
        mean, m2, nf = acc
        new_nf = nf + self.acc(1.0)
        delta = val - mean
        new_mean = mean + delta / new_nf
        new_m2 = m2 + delta * (val - new_mean)
        out_mean = new_mean if valid else mean
        out_m2 = new_m2 if valid else m2
        out_nf = new_nf if valid else nf
        return (out_mean, out_m2, out_nf)

    @cute.jit
    def combine(self, a, b):
        ma, m2a, na = a
        mb, m2b, nb = b
        zero = self.acc(0.0)
        mean, m2, nn = ma, m2a, na
        if na == zero:
            mean, m2, nn = mb, m2b, nb
        elif nb != zero:
            nn = na + nb
            nb_over_n = nb / nn
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
    # acc = (value, index). NaN wins, then larger value, with lower-index tie-breaking.
    # Indices use Int32 unless the extent can exceed its range.
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
    # acc is the product of 0/1 truth flags; NaN is truthy, matching torch.all.
    nfields = 1

    def __init__(self, acc=Float32):
        self.acc = acc
        self.fdtypes = (acc,)

    def init(self):
        return (self.acc(1.0),)

    @cute.jit
    def leaf(self, val, idx):
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
    # acc = (nonzero count,); also serves p=0 norm. NaN counts as nonzero.
    nfields = 1

    def __init__(self, acc=Float32):
        self.acc = acc
        self.fdtypes = (acc,)

    def init(self):
        return (self.acc(0.0),)

    @cute.jit
    def leaf(self, val, idx):
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
    # acc = (value, index). NaN wins, then smaller value, with lower-index tie-breaking.
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
    # acc = (max,). Avoiding an unused index halves shuffle/smem traffic and enables the
    # value-only cross-CTA path.
    nfields = 1

    def __init__(self, acc=Float32):
        self.acc = acc
        self.fdtypes = (acc,)

    def init(self):
        return (_neg_id(self.acc),)

    @cute.jit
    def _maxnan(self, a, b):
        # Keep the explicit NaN-propagating max: an autofix to max() changed emitted bits.
        return b if ((b > a) or (b != b)) else a

    @cute.jit
    def leaf(self, val, idx):
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
    # acc = (min,), the single-field NaN-propagating AMaxOps mirror.
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
    # Decreasing matches PyTorch/Triton; ascending matches ATen and changes association.
    n = min(threads_per_row, WARP)
    if n <= 0 or n & (n - 1):
        raise ValueError(f"butterfly width must be a positive power of two, got {n}")
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
    # Reduce within each row's warp group; mixing groups corrupts multi-row blocks.
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    row_g = warp // warps_per_row
    col_g = warp % warps_per_row
    if lane == 0:
        for f in cutlass.range_constexpr(trait.nfields):
            bufs[f][row_g * const_expr(warps_per_row) + col_g] = acc[f]
    cute.arch.barrier()
    # Each warp reloads its row's group, broadcasting the result to all lanes.
    out = trait.init()
    if lane < warps_per_row:
        out = tuple(
            bufs[f][row_g * const_expr(warps_per_row) + lane]
            for f in range(trait.nfields)
        )
    out = warp_reduce(trait, out, warps_per_row)
    return out


# Two-output traits reuse accumulators and return tuples from project(); nouts sizes stores.


class VarMeanOps(WelfordOps):
    # Reuse Welford and project variance/std plus mean; correction and take_sqrt are unchanged.
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
    # Reuse GreaterOrNan and project value plus index.
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
    # acc = (min, max); like torch.aminmax, any NaN makes both outputs NaN.
    nfields = 2
    nouts = 2

    def __init__(self, acc=Float32):
        self.acc = acc
        self.fdtypes = (acc, acc)

    def init(self):
        return (_pos_id(self.acc), _neg_id(self.acc))

    @cute.jit
    def _fmin(self, a, b):
        # Keep the explicit NaN truth table; see AMaxOps._maxnan.
        return (a if a != a else (a if a < b else b)) if b == b else b  # noqa: FURB136

    @cute.jit
    def _fmax(self, a, b):
        return (a if a != a else (a if a > b else b)) if b == b else b  # noqa: FURB136

    @cute.jit
    def leaf(self, val, idx):
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
