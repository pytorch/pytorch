"""Correctness checks for the custom CUDA LDL kernel behind
torch.linalg.ldl_factor(x, hermitian=True).

The kernel is a right-looking blocked LDL^H: panels of MAX_LDL_NB=32 columns are
factored by a single-block kernel, and the trailing matrix update is deferred to
a GEMM at the panel boundary. What breaks it is therefore input that crosses
panel boundaries *and* forces interchanges, so the emphasis here is on indefinite
and structurally awkward matrices at sizes straddling multiples of 32.

Two things shape how correctness is measured:

  * Never compare factors or pivots against CPU LAPACK. Bunch-Kaufman decisions
    sit on thresholds, so a different summation order legitimately picks a
    different (equally valid) pivot sequence, after which L and D differ
    wholesale. Validate with the solve residual instead.
  * The solve runs on CPU. torch.linalg.ldl_solve rejects complex inputs with
    hermitian=True on CUDA -- exactly the combination the custom kernel exists
    for -- so CPU is the only available vehicle for checking it.

Reaching the kernel at all takes two things. It sits under
LinalgBackend::Cusolver and is gated on hermitian, so the tests force the
cusolver backend in setUp and pass hermitian=True. Under the default backend
complex+hermitian goes to MAGMA and everything else to cuSOLVER, so the custom
kernel is never called and the whole file silently tests something else. The
gate is on hermitian only -- real dtypes reach it just as complex ones do.
"""

import functools
import os
import re

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    largeTensorTest,
    onlyCUDA,
)
from torch.testing._internal.common_dtype import floating_and_complex_types
from torch.testing._internal.common_utils import (
    make_tensor,
    parametrize,
    random_hermitian_pd_matrix,
    run_tests,
    TestCase,
)


# The kernel's panel width. Sizes either side of a multiple of this are what
# exercise the deferred trailing update and out-of-panel pivots.
NB = 32

_HERE = os.path.dirname(os.path.abspath(__file__))
_STANDALONE_CU = os.path.join(_HERE, "ldl_panel_standalone.cu")
_ORIGINAL_CU = os.path.join(_HERE, "aten/src/ATen/native/cuda/linalg/LURecBlas3Kernel.cu")


@functools.cache
def _standalone_ext():
    from torch.utils.cpp_extension import load

    return load("ldl_panel_standalone", [_STANDALONE_CU], verbose=False)


_GRID_VARIANTS = [(True, b, 256) for b in (1, 2, 3, 7)] + [(False, 1, 256), (False, 1, 1024)]


def _factor_panels(A, nb, panels, cooperative, nblocks, nthreads, hermitian):
    """Runs the first `panels` panels (all of them if None) of the standalone kernel
    on a column-major copy of A, with the trailing update between them as the
    library does it."""
    n = A.size(-1)
    LD = A.mT.contiguous().mT.clone()
    ipiv = torch.zeros(n, dtype=torch.int32, device=A.device)
    info = torch.zeros(1, dtype=torch.int32, device=A.device)
    step = done = 0
    while step < n and (panels is None or done < panels):
        width = min(nb, n - step)
        new_step = _standalone_ext().panel(
            LD, ipiv, info, width, step, cooperative, nblocks, nthreads, hermitian
        )
        if new_step == step:
            raise RuntimeError(f"the panel at step {step} made no progress")
        end = step + width
        if end < n:
            LD[end:, end:] -= LD[end:, step:new_step] @ LD[step:new_step, end:]
        step = new_step
        done += 1
    return LD, ipiv, info


# The threshold LAPACK's own test drivers use for its scaled residuals. Those run at
# small n; past n ~ 2000 the residual of any backward-stable solver on these inputs
# exceeds it, and how fast it grows with n depends on the machine: on GB200/aarch64
# it is about 2x what it was on H100/x86, on CPU LAPACK, cuSOLVER LU and the custom
# kernel alike. So large n is bounded relative to an LU solve of the same system,
# against which the custom kernel and CPU LAPACK's LDL measured at most 1.2x, and
# 1.3x on integer_valued, where Bunch-Kaufman is simply less accurate than LU.
LAPACK_RESID_THRESH = 30.0
REF_RESID_FACTOR = 1.5


def _resid_threshold(A, B):
    ref = _lapack_residual(A, torch.linalg.solve(A, B), B)
    return max(LAPACK_RESID_THRESH, REF_RESID_FACTOR * ref)


def _lapack_residual(A, X, B):
    """LAPACK's scaled residual for a linear solve (as in xPOT02/xSYT02):
    ||B - A X||_1 / (||A||_1 ||X||_1 eps), with eps the relative machine precision
    (half of finfo.eps, as LAPACK's dlamch('E') counts rounding, not truncation).
    Worst over the batch."""
    eps = torch.finfo(A.dtype).eps / 2
    one_norm = lambda M: M.abs().sum(-2).amax(-1)
    return (one_norm(B - A @ X) / (one_norm(A) * one_norm(X) * eps)).amax().item()


def _residual_after_solve(A, LD, piv, hermitian=True, nrhs=3):
    """Solves on CPU (complex+Hermitian is not supported on CUDA) with the
    factorization of A, returning the LAPACK residual and its bound. The reference
    solve for the bound runs on A's device: LU on CPU is slow at large n."""
    A = symmetrize(A, hermitian)
    B = make_tensor((*A.shape[:-1], nrhs), dtype=A.dtype, device="cpu")
    X = torch.linalg.ldl_solve(LD.cpu(), piv.cpu(), B, hermitian=hermitian)
    return _lapack_residual(A.cpu(), X, B), _resid_threshold(A, B.to(A.device))


# Sizes for test_huge's blocks_per_sm: ~47-50 GiB per copy of the matrix, so that 3.2
# copies fit on a 184 GiB GB200. Cooperative blocks in the first panel: ceil(n / 256)
# = 450, 313, 313, 219, against caps of 760, 760, 608, 456 there.
_HUGE_N = {torch.float32: 115000, torch.float64: 80000, torch.complex64: 80000, torch.complex128: 56000}


def _huge_n(dtype, regime):
    return 46341 if regime == "int32_offsets" else _HUGE_N[dtype]


def _huge_bytes(dtype, regime, copies):
    return int(copies * _huge_n(dtype, regime) ** 2 * dtype.itemsize)


def _equal_with_nonfinite(a, b):
    """Bitwise equality that tolerates NaNs, whose payload is not specified."""
    return torch.equal(a.isnan(), b.isnan()) and torch.equal(a.nan_to_num(), b.nan_to_num())


def _kernel_source(path, begin, end):
    with open(path) as f:
        text = f.read()
    body = text[text.index(begin) : text.index(end)]
    return re.sub(r"\s+", " ", body).strip()


def _real_dtype(dtype):
    return torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32


def _hermitian(a):
    return a + a.mH


def _spectrum(n, batch, s, dtype, device):
    """Hermitian matrix with prescribed eigenvalues s via Q diag(s) Q^H."""
    x = torch.randn(*batch, n, n, dtype=dtype, device=device)
    q, _ = torch.linalg.qr(x)
    return (q * s.unsqueeze(-2).to(dtype)) @ q.mH


def exact_last_zero_matrix(n, dtype, device, seed, swaps=()):
    """Hermitian matrix whose last pivot is exactly zero, in exact arithmetic and in
    floating point alike: an identity-like diagonal d_i = +-2^k with one coupling
    v_i = 2^j * unit to the last index (|v_i| <= |d_i|, so no interchange), and
    A[-1, -1] = sum |v_i|^2 / d_i. Everything is a small dyadic number, so the
    updates reach exactly zero in whatever order they are applied, unlike a general
    singular matrix, where rounding decides. The last entry is only ever touched
    by the updates, never factored until the very end.

    Each (a, b) in swaps (a < b, all distinct, below n - 1) plants an interchange:
    d_a = 1/2, d_b = 4 and A[a, b] = 1 fail the 1x1 test for column a and make
    Bunch-Kaufman take b as a 1x1 pivot instead. The pair's couplings v_a = v_b / 4
    cancel once b is eliminated, so no further interchange follows, and the Schur
    complement of the pair is added to A[-1, -1] exactly."""
    g = torch.Generator().manual_seed(seed)
    j = torch.randint(0, 2, (n - 1,), generator=g)
    k = j + torch.randint(0, 2, (n - 1,), generator=g)
    sign = torch.randint(0, 2, (n - 1,), generator=g) * 2 - 1
    units = [1, 1j, -1, -1j] if dtype.is_complex else [1, -1]
    u = torch.tensor([units[i] for i in torch.randint(0, len(units), (n - 1,), generator=g).tolist()], dtype=dtype)
    v = (2.0 ** j).to(dtype) * u
    d = (sign * 2.0 ** k).to(dtype)
    A = torch.zeros(n, n, dtype=dtype)
    last = torch.zeros((), dtype=dtype)
    paired = set()
    for a, b in swaps:
        paired |= {a, b}
        d[a], d[b] = 0.5, 4.0
        v[b] = u[b]
        v[a] = 0.25 * u[b]
        A[a, b] = A[b, a] = 1.0
        v2 = torch.stack([v[a], v[b]])
        Minv = torch.tensor([[4.0, -1.0], [-1.0, 0.5]], dtype=dtype)
        last = last + v2.conj() @ Minv @ v2
    for i in range(n - 1):
        if i not in paired:
            last = last + v[i].abs() ** 2 / d[i].real
    A.diagonal()[:-1] = d
    A[:-1, -1] = v
    A[-1, :-1] = v.conj()
    A[-1, -1] = last.real if dtype.is_complex else last
    return A.to(device)


# ---------------------------------------------------------------- structures
# Each returns a Hermitian matrix of shape (*batch, n, n).

def indefinite(n, batch, dtype, device):
    """Mixed-sign spectrum, cond ~ 3. The baseline hard case: indefiniteness is
    what forces 2x2 pivots and interchanges in the first place."""
    s = 1.0 + 2.0 * torch.rand(*batch, n, dtype=_real_dtype(dtype), device=device)
    s[..., ::2] *= -1
    return _hermitian(_spectrum(n, batch, s, dtype, device))


def positive_definite(n, batch, dtype, device):
    """Never interchanges, so it passes even with the interchange path broken.
    Kept as a control that separates the two failure modes."""
    return random_hermitian_pd_matrix(n, *batch, dtype=dtype, device=device)


def negative_definite(n, batch, dtype, device):
    return -positive_definite(n, batch, dtype, device)


def zero_diagonal(n, batch, dtype, device):
    """Diagonal identically zero, so no 1x1 pivot is ever admissible and
    Bunch-Kaufman must take a 2x2 block at every step."""
    a = _hermitian(torch.randn(*batch, n, n, dtype=dtype, device=device))
    a.diagonal(dim1=-2, dim2=-1).zero_()
    return a


def tiny_diagonal(n, batch, dtype, device):
    """Diagonal small relative to the off-diagonal: pushes the alpha tests right
    up against their thresholds, where 1x1 vs 2x2 is nearly a coin flip."""
    a = _hermitian(torch.randn(*batch, n, n, dtype=dtype, device=device))
    a.diagonal(dim1=-2, dim2=-1).mul_(1e-3)
    return a


def out_of_panel_heavy(n, batch, dtype, device):
    """Mass concentrated in rows/cols >= NB, so the pivot search for early
    columns keeps selecting candidates outside the current panel.

    This is the direct stress for the deferred-block replay: the candidate row
    and column live in the region the panel has not yet updated.
    """
    a = _hermitian(torch.randn(*batch, n, n, dtype=dtype, device=device))
    scale = torch.ones(n, dtype=_real_dtype(dtype), device=device)
    scale[NB:] = 50.0
    # D A D keeps the matrix Hermitian
    a = a * scale.unsqueeze(-1).to(dtype) * scale.unsqueeze(-2).to(dtype)
    a.diagonal(dim1=-2, dim2=-1).mul_(1e-2)
    return a


def ill_conditioned(n, batch, dtype, device):
    """Spectrum spanning many orders of magnitude, mixed sign."""
    rdt = _real_dtype(dtype)
    lo = -8.0 if rdt == torch.float64 else -4.0
    e = torch.linspace(lo, 0.0, n, dtype=rdt, device=device).expand(*batch, n).clone()
    s = torch.pow(torch.tensor(10.0, dtype=rdt, device=device), e)
    s[..., ::2] *= -1
    return _hermitian(_spectrum(n, batch, s, dtype, device))


def clustered(n, batch, dtype, device):
    """Eigenvalues all +-1: maximal tying in the pivot search, which exercises
    the argmax tie-break (lowest index wins)."""
    s = torch.ones(*batch, n, dtype=_real_dtype(dtype), device=device)
    s[..., ::2] *= -1
    return _hermitian(_spectrum(n, batch, s, dtype, device))


def integer_valued(n, batch, dtype, device):
    """Small integer entries produce exact ties in the pivot search, unlike
    random floats which essentially never tie."""
    a = torch.randint(-3, 4, (*batch, n, n), device=device).to(dtype)
    return _hermitian(a)


def tridiagonal(n, batch, dtype, device):
    a = _hermitian(torch.randn(*batch, n, n, dtype=dtype, device=device))
    mask = torch.ones(n, n, dtype=torch.bool, device=device).tril(1).triu(-1)
    return a * mask


def banded(n, batch, dtype, device):
    a = _hermitian(torch.randn(*batch, n, n, dtype=dtype, device=device))
    bw = max(1, NB // 4)
    mask = torch.ones(n, n, dtype=torch.bool, device=device).tril(bw).triu(-bw)
    return a * mask


def arrow(n, batch, dtype, device):
    """Diagonal plus a dense last row/column -- the pivot search for every
    column has its off-diagonal maximum at the very last index."""
    a = torch.zeros(*batch, n, n, dtype=dtype, device=device)
    a.diagonal(dim1=-2, dim2=-1).copy_(
        1.0 + torch.rand(*batch, n, dtype=_real_dtype(dtype), device=device)
    )
    border = torch.randn(*batch, n, dtype=dtype, device=device)
    a[..., -1, :] = border
    a[..., :, -1] = border.conj()
    a[..., -1, -1] = a[..., -1, -1].real.to(dtype)
    return a


def diagonal_only(n, batch, dtype, device):
    d = 1.0 + torch.rand(*batch, n, dtype=_real_dtype(dtype), device=device)
    d[..., ::2] *= -1
    return torch.diag_embed(d).to(dtype)


def complex_real_valued(n, batch, dtype, device):
    """Complex dtype carrying exactly real data. D is then real by construction,
    which is the case the kernel used to force explicitly and no longer does."""
    a = indefinite(n, batch, _real_dtype(dtype), device)
    return a.to(dtype)


def imaginary_offdiag(n, batch, dtype, device):
    """Real diagonal, purely imaginary off-diagonal: still Hermitian, and it
    drives the |re| + |im| pivot magnitude entirely through the imaginary part."""
    if not dtype.is_complex:
        return indefinite(n, batch, dtype, device)
    a = torch.randn(*batch, n, n, dtype=_real_dtype(dtype), device=device)
    a = (a - a.mT).to(dtype) * 1j
    d = 1.0 + torch.rand(*batch, n, dtype=_real_dtype(dtype), device=device)
    d[..., ::2] *= -1
    return a + torch.diag_embed(d).to(dtype) * 1e-2


def tiny_scale(n, batch, dtype, device):
    scale = 1e-30 if _real_dtype(dtype) == torch.float64 else 1e-15
    return indefinite(n, batch, dtype, device) * scale


def huge_scale(n, batch, dtype, device):
    scale = 1e30 if _real_dtype(dtype) == torch.float64 else 1e15
    return indefinite(n, batch, dtype, device) * scale


STRUCTURES = {
    "indefinite": indefinite,
    "positive_definite": positive_definite,
    "negative_definite": negative_definite,
    "zero_diagonal": zero_diagonal,
    "tiny_diagonal": tiny_diagonal,
    "out_of_panel_heavy": out_of_panel_heavy,
    "ill_conditioned": ill_conditioned,
    "clustered": clustered,
    "integer_valued": integer_valued,
    "tridiagonal": tridiagonal,
    "banded": banded,
    "arrow": arrow,
    "diagonal_only": diagonal_only,
    "complex_real_valued": complex_real_valued,
    "imaginary_offdiag": imaginary_offdiag,
    "tiny_scale": tiny_scale,
    "huge_scale": huge_scale,
}


def symmetrize(a, hermitian):
    return a.tril() + a.tril(-1).mH if hermitian else a.tril() + a.tril(-1).mT


# ------------------------------------------------------- forced pivot patterns
# A pattern is a tuple of block sizes summing to n, e.g. (1, 2, 2, 1). Building a
# block-diagonal matrix from it pins Bunch-Kaufman's choice exactly: a dominant
# diagonal entry always yields a 1x1 pivot (lambda is 0, so |a_kk| >= alpha * 0
# holds), and a 2x2 block with a zero diagonal fails every 1x1 test in turn and
# must be taken as a 2x2. Neither case interchanges, so the achieved pattern is
# the requested one and the test can assert it.

def pivot_patterns(n):
    """Every composition of n into parts of 1 and 2."""
    if n == 0:
        yield ()
        return
    for rest in pivot_patterns(n - 1):
        yield (1,) + rest
    if n >= 2:
        for rest in pivot_patterns(n - 2):
            yield (2,) + rest


def pattern_from_pivots(pivots):
    """Recover the block structure from LAPACK's compact pivot encoding."""
    out, k, p = [], 0, pivots.tolist()
    while k < len(p):
        if p[k] < 0:
            out.append(2)
            k += 2
        else:
            out.append(1)
            k += 1
    return tuple(out)


def _straddling_pattern(n, period):
    """1x1 blocks, with a 2x2 across every boundary at a multiple of period."""
    pattern, pos = [], 0
    while pos < n:
        if (pos + 1) % period == 0 and pos + 2 <= n:
            pattern.append(2)
            pos += 2
        else:
            pattern.append(1)
            pos += 1
    return tuple(pattern)


def matrix_for_pattern(pattern, dtype, device, coupling=0.0):
    n = sum(pattern)
    a = torch.zeros(n, n, dtype=dtype, device=device)
    rdt = _real_dtype(dtype)
    i = 0
    for blk in pattern:
        if blk == 1:
            # magnitude >= 1 so the 1x1 pivot is never degenerate
            d = 1.0 + torch.rand((), dtype=rdt, device=device)
            a[i, i] = (d if i % 4 < 2 else -d).to(dtype)
        else:
            # zero diagonal, nonzero off-diagonal -> forced 2x2, det = -|b|^2
            b = torch.randn((), dtype=dtype, device=device) + 1.0
            a[i, i + 1] = b
            a[i + 1, i] = b.conj()
        i += blk
    if coupling:
        noise = _hermitian(torch.randn(n, n, dtype=dtype, device=device)) * coupling
        noise.diagonal().mul_(0)  # keep the forced diagonal structure intact
        a = a + noise
    return a


def mostly_2x2_pattern(n, frac2, seed):
    import random

    rng = random.Random(seed)
    pat, tot = [], 0
    while tot < n:
        if n - tot == 1 or rng.random() >= frac2:
            pat.append(1)
            tot += 1
        else:
            pat.append(2)
            tot += 2
    return tuple(pat)


class TestLDLCustomKernel(TestCase):
    def setUp(self):
        super().setUp()
        self._prev_backend = None
        if torch.cuda.is_available():
            # Without this the custom kernel is unreachable -- see module docstring.
            self._prev_backend = torch.backends.cuda.preferred_linalg_library()
            torch.backends.cuda.preferred_linalg_library("cusolver")

    def tearDown(self):
        if self._prev_backend is not None:
            torch.backends.cuda.preferred_linalg_library(self._prev_backend)
        super().tearDown()

    def _check(self, A, hermitian, nrhs=3):
        """Factor A on its own device, verify via the solve residual on CPU."""
        LD, pivots, info = torch.linalg.ldl_factor_ex(A, hermitian=hermitian)

        self.assertEqual(info, torch.zeros_like(info))

        # LAPACK's compact pivot encoding: 1-based, sign marks a 2x2 block, so
        # every entry must be a usable index. Catches a panel that never wrote
        # part of the vector, which reads back as uninitialized memory.
        n = A.shape[-1]
        self.assertTrue((pivots.abs() >= 1).all(), f"pivot with |p| < 1: {pivots}")
        self.assertTrue((pivots.abs() <= n).all(), f"pivot out of range: {pivots}")

        # LAPACK's scaled solve residual. The per-dtype defaults of assertEqual are far too
        # tight here: they reject valid float32 results from n=33 up, on CPU as
        # readily as on CUDA.
        resid, thresh = _residual_after_solve(A, LD, pivots, hermitian, nrhs)
        self.assertLess(resid, thresh)

    # ------------------------------------------------------------------ sweeps

    @parametrize("structure", list(STRUCTURES))
    @parametrize("n", [2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 34, 63, 64, 65, 66, 96, 127, 128, 129])
    @dtypes(*floating_and_complex_types())
    def test_structures(self, device, dtype, n, structure):
        if structure == "zero_diagonal" and n < 2:
            self.skipTest("zero_diagonal is the zero matrix for n=1")
        torch.manual_seed(n)
        A = STRUCTURES[structure](n, (), dtype, device)
        self._check(A, hermitian=True)

    @parametrize("structure", ["indefinite", "zero_diagonal", "out_of_panel_heavy", "arrow"])
    @parametrize("n", [150, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 4096, 4097])
    @dtypes(torch.float64, torch.complex128)
    def test_large(self, device, dtype, n, structure):
        """Sizes past 512, where the panel kernel switches from 256 to 1024
        threads, and with many panels for the deferred update to accumulate."""
        torch.manual_seed(n)
        A = STRUCTURES[structure](n, (), dtype, device)
        self._check(A, hermitian=True)

    # Panels with more than COOP_MIN_ROWS = 2048 remaining rows are factored by a
    # cooperative grid with one row per thread, so the pivot search is a grid-wide
    # argmax. For n > 2048 the first panels take that path and the last ones fall
    # back to a single block within the same factorization.
    @onlyCUDA
    @parametrize("structure", ["indefinite", "zero_diagonal", "out_of_panel_heavy",
                               "arrow", "integer_valued", "clustered"])
    @parametrize("n", [2048, 2049, 2050, 2304, 2305, 2560, 3000, 4100, 8193])
    @dtypes(torch.float64, torch.complex128)
    def test_cooperative_panel(self, device, dtype, n, structure):
        """arrow puts every column's maximum in the last row, i.e. the last block;
        integer_valued and clustered tie across blocks, so the lowest index has to
        win the grid-wide reduction as it does within a block."""
        torch.manual_seed(n)
        A = STRUCTURES[structure](n, (), dtype, device)
        self._check(A, hermitian=True)

    @onlyCUDA
    @parametrize("n", [2049, 4100])
    @dtypes(torch.float32, torch.complex64)
    def test_cooperative_panel_single_precision(self, device, dtype, n):
        torch.manual_seed(n)
        self._check(indefinite(n, (), dtype, device), hermitian=True)

    @onlyCUDA
    @parametrize("n", [2049, 3000])
    @dtypes(torch.float64, torch.complex128)
    def test_cooperative_panel_batched(self, device, dtype, n):
        torch.manual_seed(n)
        self._check(indefinite(n, (2,), dtype, device), hermitian=True)

    @onlyCUDA
    @parametrize("n", [2049, 4100])
    @dtypes(torch.float64, torch.complex128)
    def test_cooperative_panel_determinism(self, device, dtype, n):
        """The grid-wide reduction must not depend on block scheduling."""
        torch.manual_seed(n)
        A = integer_valued(n, (), dtype, device)
        LD0, piv0, _ = torch.linalg.ldl_factor_ex(A, hermitian=True)
        for _ in range(5):
            LD, piv, _ = torch.linalg.ldl_factor_ex(A, hermitian=True)
            self.assertTrue(torch.equal(piv, piv0))
            self.assertTrue(torch.equal(LD, LD0))

    @onlyCUDA
    @largeTensorTest("60GB", "cuda")
    @dtypes(torch.float64)
    def test_very_large(self, device, dtype):
        """Many cooperative panels with a grid of 157 blocks at the start."""
        n = 40000
        torch.manual_seed(n)
        self._check(indefinite(n, (), dtype, device), hermitian=True)

    # Two regimes the sizes above never reach, both needing ~3 copies of the matrix
    # on the device (the input, LD, and the temporary in ldl_factor_blas3_kernel):
    #  - int32_offsets: n^2 > 2^31, so the offsets of the last columns overflow int32.
    #  - blocks_per_sm: grids of 1.4 to 3 cooperative blocks per SM on GB200 (152 SMs),
    #    so the grid-wide argmax and sync run with blocks sharing SMs. n = 40000
    #    above is ~1 block per SM there.
    # zero_diagonal forces a 2x2 pivot with an interchange at every step, from anywhere
    # in the column, i.e. from any block of the grid.
    @onlyCUDA
    @largeTensorTest(lambda self, device, dtype, regime: _huge_bytes(dtype, regime, 3.2))
    @largeTensorTest(lambda self, device, dtype, regime: _huge_bytes(dtype, regime, 3), device="cpu")
    @parametrize("regime", ["int32_offsets", "blocks_per_sm"])
    @dtypes(*floating_and_complex_types())
    def test_huge(self, device, dtype, regime):
        n = _huge_n(dtype, regime)
        torch.manual_seed(n)
        A = zero_diagonal(n, (), dtype, device)
        A_cpu = A.cpu()
        LD, piv, info = torch.linalg.ldl_factor_ex(A, hermitian=True)
        self.assertEqual(info.item(), 0)
        # Off the device before the reference solve, which needs two more copies.
        LD, piv = LD.cpu(), piv.cpu()
        B = make_tensor((n, 3), dtype=dtype, device="cpu")
        resid = _lapack_residual(A_cpu, torch.linalg.ldl_solve(LD, piv, B, hermitian=True), B)
        del LD, A_cpu
        self.assertLess(resid, _resid_threshold(A, B.to(device)))

    @onlyCUDA
    def test_standalone_kernel_matches_original(self, device):
        """ldl_panel_standalone.cu holds a copy of the panel kernel, so guard it
        against drifting from the original."""
        original = _kernel_source(
            _ORIGINAL_CU,
            "template <typename scalar_t, int BS, bool hermitian, bool cooperative>",
            "// Upper bound on the grid of the cooperative",
        )
        copy = _kernel_source(_STANDALONE_CU, "// BEGIN COPY", "// END COPY")
        self.assertEqual(copy.replace("// BEGIN COPY", "").strip(), original)

    # The cooperative grid is capped by max_cooperative_panel_blocks() (660 on H100
    # for double, i.e. n > ~169k), after which every thread covers several rows.
    # A grid smaller than the row count is forced here on the standalone copy of
    # the kernel, whose results must not depend on the grid size at all.
    @onlyCUDA
    @parametrize("nb", [32, 64, 128])
    @parametrize("structure", ["indefinite", "arrow", "integer_valued", "zero_diagonal"])
    @parametrize("n", [300, 4100])
    @dtypes(*floating_and_complex_types())
    def test_panel_grid_size_invariance(self, device, dtype, n, structure, nb):
        torch.manual_seed(n)
        A = STRUCTURES[structure](n, (), dtype, device)
        self._check_grid_invariance(A, nb, hermitian=True)

    # The runs above stop after a few panels. These go to the end, where the last column
    # is handled by its own launch (or the panel that ends on it), with every grid.
    @onlyCUDA
    @parametrize("nb", [16, 32, 48])
    @parametrize("n", [70, 300, 2100])
    @dtypes(*floating_and_complex_types())
    def test_panel_grid_size_invariance_to_completion(self, device, dtype, n, nb):
        torch.manual_seed(n)
        A = indefinite(n, (), dtype, device)
        self._check_grid_invariance(A, nb, hermitian=True, panels=None)

    @onlyCUDA
    @parametrize("nb", [16, 32, 48])
    @parametrize("n", [70, 300, 2100])
    @dtypes(*floating_and_complex_types())
    def test_panel_last_pivot_to_completion(self, device, dtype, n, nb):
        A = exact_last_zero_matrix(n, dtype, device, seed=n)
        for LD, ipiv, info in self._grid_variants(A, nb, hermitian=True, panels=None):
            self.assertEqual(info.item(), n)
            self.assertEqual(ipiv[-1].item(), n)
            self.assertEqual(LD.diagonal().imag.abs().max().item() if dtype.is_complex else 0, 0)

    # The library picks the panel width per architecture (e.g. 16, 24, 48, 96 on GB200
    # against 32 to 128 on H100), and ends panels on 256-byte boundaries. Whatever the
    # width, the whole factorization has to be right, so run the standalone kernel with
    # all of them, 2x2 blocks straddling the panel ends included.
    @onlyCUDA
    @parametrize("structure", ["indefinite", "zero_diagonal", "out_of_panel_heavy"])
    @parametrize("nb", [8, 16, 24, 48, 64, 96, 128, 200])
    @parametrize("n", [100, 300, 700])
    @dtypes(*floating_and_complex_types())
    def test_panel_widths_full_factorization(self, device, dtype, n, nb, structure):
        torch.manual_seed(n)
        A = STRUCTURES[structure](n, (), dtype, device)
        for cooperative, nblocks, nthreads in ((True, -(-n // 256), 256), (False, 1, 1024)):
            LD, ipiv, info = _factor_panels(A, nb, None, cooperative, nblocks, nthreads, True)
            self.assertEqual(info.item(), 0)
            torch.manual_seed(0)
            self.assertLess(*_residual_after_solve(A, LD, ipiv))

    @onlyCUDA
    @parametrize("n", [300, 4100])
    @dtypes(torch.complex64, torch.complex128)
    def test_panel_grid_size_invariance_symmetric(self, device, dtype, n):
        """Complex symmetric (not Hermitian): the other instantiation of the kernel."""
        torch.manual_seed(n)
        A = symmetrize(torch.randn(n, n, dtype=dtype, device=device), hermitian=False)
        self._check_grid_invariance(A, 32, hermitian=False)

    # Column 0 is planted with off-diagonal entries of equal magnitude in the given
    # rows, over a near-zero background and a zero diagonal. Every 1x1 test then
    # fails, and so does the one on the candidate's diagonal, so the pivot is a 2x2
    # block with the candidate as its second row: the swap runs between rows living
    # in different blocks. Ties must go to the lowest row, as inside a block.
    _PLANTED_ROWS = {
        "block_boundary": (255, 256),
        "far_blocks": (700, 3000, 4099),
        "same_block_then_far": (1000, 1010, 4000),
        "last_row": (4099,),
        **{f"block_{b}": (256 * b + 17,) for b in range(16)},
    }

    @onlyCUDA
    @parametrize("placement", list(_PLANTED_ROWS))
    @dtypes(*floating_and_complex_types())
    def test_pivot_across_blocks(self, device, dtype, placement):
        n, rows = 4100, self._PLANTED_ROWS[placement]
        torch.manual_seed(0)
        A = symmetrize(1e-3 * torch.randn(n, n, dtype=dtype, device=device), hermitian=True)
        A.diagonal().zero_()
        A[:, 0] = A[0, :].conj() if dtype.is_complex else A[0, :]
        for i, r in enumerate(rows):
            # |re| + |im| ties between 1 and 1j, which a Euclidean abs would not see
            A[r, 0] = 1j if (dtype.is_complex and i % 2) else 1.0
            A[0, r] = A[r, 0].conj()
        want = -(min(rows) + 1)
        for LD, ipiv, info in self._grid_variants(A, 32, hermitian=True):
            self.assertEqual(ipiv[:2].tolist(), [want, want])

    @onlyCUDA
    @parametrize("bad", [float("nan"), float("inf")])
    @dtypes(*floating_and_complex_types())
    def test_panel_nonfinite_column(self, device, dtype, bad):
        """A column of NaN/Inf makes the grid-wide argmax see no (or an infinite)
        candidate. Nothing may crash, and every grid must agree."""
        n = 4100
        torch.manual_seed(0)
        A = indefinite(n, (), dtype, device)
        A[1:, 0] = bad
        A[0, 1:] = bad
        self._check_grid_invariance(A, 32, hermitian=True, panels=2)

    @onlyCUDA
    @dtypes(torch.float64, torch.complex128)
    def test_panel_race_stress(self, device, dtype):
        """The partial results of the grid-wide argmax are double-buffered because a
        fast block may move on while a slow one still reads: perturb the block
        scheduling with concurrent GPU work and repeat."""
        n = 4100
        torch.manual_seed(0)
        A = integer_valued(n, (), dtype, device)
        ref = _factor_panels(A, 32, 2, True, -(-n // 256), 256, True)
        busy = torch.randn(4096, 4096, device=device)
        side = torch.cuda.Stream()
        for nblocks in (2, 3, 17):
            for _ in range(40):
                with torch.cuda.stream(side):
                    for _ in range(8):
                        busy @ busy
                got = _factor_panels(A, 32, 2, True, nblocks, 256, True)
                torch.cuda.synchronize()
                for g, w in zip(got, ref):
                    self.assertTrue(torch.equal(g, w))

    def _grid_variants(self, A, nb, hermitian, panels=4):
        """Results of the full-grid reference and of every forced configuration,
        reference first."""
        n = A.size(-1)
        yield _factor_panels(A, nb, panels, True, -(-n // 256), 256, hermitian)
        for cooperative, nblocks, nthreads in _GRID_VARIANTS:
            yield _factor_panels(A, nb, panels, cooperative, nblocks, nthreads, hermitian)

    def _check_grid_invariance(self, A, nb, hermitian, panels=4):
        variants = self._grid_variants(A, nb, hermitian, panels)
        ref = next(variants)
        for (cooperative, nblocks, nthreads), got in zip(_GRID_VARIANTS, variants):
            with self.subTest(cooperative=cooperative, nblocks=nblocks, nthreads=nthreads):
                for g, w in zip(got, ref):
                    self.assertTrue(_equal_with_nonfinite(g, w))

    @parametrize("structure", ["indefinite", "zero_diagonal", "out_of_panel_heavy",
                               "clustered", "ill_conditioned"])
    @parametrize("batch", [(1,), (2,), (3,), (5,), (2, 2), (1, 3), (3, 1), (2, 2, 2)])
    @parametrize("n", [2, 3, 5, 16, 32, 33, 64, 65, 150])
    @dtypes(torch.float64, torch.complex128)
    def test_batched(self, device, dtype, n, batch, structure):
        """The kernel factors one matrix per launch, so the batch is driven by a
        loop in the kernel's host wrapper. Multi-dimensional batch shapes matter:
        they are flattened before the loop, so a wrong flattening shows up as
        untouched trailing elements rather than as a wrong first element.
        """
        torch.manual_seed(n)
        A = STRUCTURES[structure](n, batch, dtype, device)
        self._check(A, hermitian=True)

    @parametrize("batch", [(2,), (4,), (2, 3)])
    @parametrize("n", [5, 33, 64])
    @dtypes(torch.float64, torch.complex128)
    def test_batched_elements_are_independent(self, device, dtype, n, batch):
        """Each batch element must be factored from its own data.

        Factoring the batch must agree element-for-element with factoring each
        matrix on its own -- the failure mode when only element 0 is processed is
        that the rest come back as uninitialized memory, which this catches
        directly rather than through a residual.
        """
        torch.manual_seed(n)
        A = STRUCTURES["indefinite"](n, batch, dtype, device)
        LD, piv, info = torch.linalg.ldl_factor_ex(A, hermitian=True)
        flat = A.reshape(-1, n, n)
        for i in range(flat.shape[0]):
            eLD, epiv, einfo = torch.linalg.ldl_factor_ex(flat[i], hermitian=True)
            self.assertTrue(torch.equal(piv.reshape(-1, n)[i], epiv),
                            f"batch element {i} pivots differ from standalone")
            self.assertEqual(LD.reshape(-1, n, n)[i], eLD)
            self.assertEqual(info.reshape(-1)[i], einfo)

    @parametrize("n", [2, 5, 33, 65, 150])
    @dtypes(torch.float64, torch.complex128)
    def test_symmetric_not_hermitian(self, device, dtype, n):
        """hermitian=False reads the lower triangle as a symmetric (not
        conjugate-symmetric) matrix -- a separate dispatch path."""
        torch.manual_seed(n)
        A = indefinite(n, (), dtype, device)
        self._check(A, hermitian=False)

    @parametrize("n", [33, 65])
    @dtypes(torch.float64, torch.complex128)
    def test_noncontiguous(self, device, dtype, n):
        """The kernel takes lda from the input stride, so a batch slice with a
        non-trivial layout is a distinct path."""
        torch.manual_seed(n)
        big = indefinite(n, (3,), dtype, device)
        self._check(big[1], hermitian=True)
        self._check(big.mH[2], hermitian=True)

    @parametrize("nrhs", [1, 7, 16])
    @parametrize("n", [33, 64])
    @dtypes(torch.float64, torch.complex128)
    def test_nrhs(self, device, dtype, n, nrhs):
        torch.manual_seed(n)
        A = indefinite(n, (), dtype, device)
        self._check(A, hermitian=True, nrhs=nrhs)

    @parametrize("n", list(range(1, 11)))
    @dtypes(*floating_and_complex_types())
    def test_all_pivot_patterns(self, device, dtype, n):
        """Every 1x1/2x2 block layout for small n.

        The matrix is built so Bunch-Kaufman is forced into the requested
        layout, which lets the achieved pattern be asserted rather than hoped
        for. n <= 10 fits in a single panel, so this isolates the block handling
        -- D construction, inv(D) for rank 2, L21 scaling -- from the blocking.
        """
        for pattern in pivot_patterns(n):
            torch.manual_seed(n)
            A = matrix_for_pattern(pattern, dtype, device)
            with self.subTest(pattern=pattern):
                _, piv, info = torch.linalg.ldl_factor_ex(A, hermitian=True)
                self.assertEqual(info, torch.zeros_like(info))
                self.assertEqual(pattern_from_pivots(piv.cpu()), pattern)
                self._check(A, hermitian=True)

    @parametrize("coupling", [0.0, 0.05])
    @parametrize("n", [150, 151, 256, 513])
    @dtypes(torch.float64, torch.complex128)
    def test_mostly_2x2(self, device, dtype, n, coupling):
        """Sizes spanning several panels with ~90% 2x2 pivots.

        Uncoupled the layout is exact and is asserted; the coupled variant
        perturbs it enough to draw in interchanges and out-of-panel pivots, so
        only the residual is checked there.
        """
        for seed in range(3):
            pattern = mostly_2x2_pattern(n, frac2=0.9, seed=seed)
            torch.manual_seed(seed)
            A = matrix_for_pattern(pattern, dtype, device, coupling=coupling)
            with self.subTest(seed=seed, blocks2=sum(1 for b in pattern if b == 2)):
                self._check(A, hermitian=True)
                if coupling == 0.0:
                    _, piv, _ = torch.linalg.ldl_factor_ex(A, hermitian=True)
                    self.assertEqual(pattern_from_pivots(piv.cpu()), pattern)

    # -------------------------------------------------------------- CUDA-only

    @onlyCUDA
    @parametrize("structure", list(STRUCTURES))
    @parametrize("n", list(range(1, 41)))
    @dtypes(torch.float64, torch.complex128)
    def test_pivots_match_cpu(self, device, dtype, n, structure):
        """CUDA and CPU Bunch-Kaufman must choose identical pivots for small n.

        This is the one place the suite compares pivots against CPU. Up to
        NB=32 the panel kernel runs the whole factorization with no inter-panel
        GEMM, so it follows LAPACK's sequential zhetf2 closely enough for the
        choices to coincide exactly; 33..40 add one boundary with a trailing
        block of at most 8 columns.

        Agreement was measured to hold well past this range -- every structure
        here matches CPU through n=512, including the all-2x2 cases by both
        routes (zero_diagonal via the search, block-diagonal patterns by
        construction). The bound is 40 because integer_valued, whose small
        integer entries produce exact ties in the pivot search, first diverges
        at n=100: with two candidates of equal magnitude the two backends can
        pick either, and both factorizations are valid (residuals 1e-14 and
        4e-13 at n=100 and 512). Widening the range means excluding that
        structure, not relaxing the assertion.

        Everything else in this file validates by residual instead, because at
        large n the pivot sequence is genuinely allowed to differ.
        """
        if structure == "zero_diagonal" and n < 2:
            self.skipTest("zero_diagonal is the zero matrix for n=1")
        for seed in range(10):
            torch.manual_seed(seed)
            A = STRUCTURES[structure](n, (), dtype, device)
            _, piv_cuda, _ = torch.linalg.ldl_factor_ex(A, hermitian=True)
            _, piv_cpu, _ = torch.linalg.ldl_factor_ex(A.cpu(), hermitian=True)
            with self.subTest(seed=seed):
                self.assertTrue(
                    torch.equal(piv_cuda.cpu(), piv_cpu),
                    f"n={n} seed={seed}: cuda={piv_cuda.cpu().tolist()} cpu={piv_cpu.tolist()}",
                )

    @onlyCUDA
    @parametrize("structure", ["indefinite", "zero_diagonal", "out_of_panel_heavy"])
    @dtypes(torch.float64, torch.complex128)
    def test_many_seeds(self, device, dtype, structure):
        """Whether the pivot search reaches outside the current panel depends on
        the draw, so a single seed per size covers very little."""
        for seed in range(30):
            torch.manual_seed(seed)
            n = 33 + (seed % 48)
            A = STRUCTURES[structure](n, (), dtype, device)
            with self.subTest(seed=seed, n=n):
                self._check(A, hermitian=True)

    @onlyCUDA
    @parametrize("structure", ["indefinite", "zero_diagonal", "out_of_panel_heavy", "clustered"])
    @parametrize("n", [127, 511, 512, 513, 1023, 4097])
    @dtypes(torch.float64, torch.complex128)
    def test_determinism(self, device, dtype, n, structure):
        """The same input must factor to bitwise identical output every time.

        The panel kernel drives a single thread block through shared-memory
        reductions and hand-placed barriers, so a missing sync surfaces as
        run-to-run variation well before it becomes a wrong answer. Sizes bracket
        512, where the panel launches switch from 256 to 1024 threads and the
        reduction tree changes shape; the structures are the ones that pivot and
        interchange most, since a quiescent pivot path races against nothing.
        """
        torch.manual_seed(n)
        A = STRUCTURES[structure](n, (), dtype, device)
        ref_LD, ref_piv, ref_info = torch.linalg.ldl_factor_ex(A, hermitian=True)
        for rep in range(12):
            LD, piv, info = torch.linalg.ldl_factor_ex(A, hermitian=True)
            self.assertTrue(torch.equal(piv, ref_piv), f"pivots differ on repeat {rep}")
            self.assertTrue(torch.equal(info, ref_info), f"info differs on repeat {rep}")
            self.assertTrue(torch.equal(LD, ref_LD), f"LD differs on repeat {rep}")

    @onlyCUDA
    @parametrize("n", [257, 513])
    @dtypes(torch.complex128)
    def test_determinism_under_load(self, device, dtype, n):
        """Same check with the GPU kept busy on another stream.

        Contention changes how warps of the panel block get scheduled, which is
        what decides whether an unsynchronised read wins or loses its race. A
        quiet GPU tends to schedule warps uniformly and can hide such a bug.
        """
        torch.manual_seed(n)
        A = STRUCTURES["out_of_panel_heavy"](n, (), dtype, device)
        ref_LD, ref_piv, _ = torch.linalg.ldl_factor_ex(A, hermitian=True)

        noise = torch.randn(2048, 2048, device=device)
        side = torch.cuda.Stream()
        for rep in range(10):
            with torch.cuda.stream(side):
                for _ in range(4):
                    noise = (noise @ noise.mT).clamp_(-1e3, 1e3)
            LD, piv, _ = torch.linalg.ldl_factor_ex(A, hermitian=True)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(piv, ref_piv), f"pivots differ under load, repeat {rep}")
            self.assertTrue(torch.equal(LD, ref_LD), f"LD differs under load, repeat {rep}")

    @parametrize("n", [33, 65])
    @dtypes(torch.float64, torch.complex128)
    def test_leading_dimension(self, device, dtype, n):
        """A sub-block of a larger matrix has a row stride wider than n, which is
        a different leading dimension from the contiguous case."""
        torch.manual_seed(n)
        big = indefinite(n + 47, (), dtype, device)
        view = big[:n, :n]
        self.assertFalse(view.is_contiguous())
        LD_v, piv_v, _ = torch.linalg.ldl_factor_ex(view, hermitian=True)
        LD_c, piv_c, _ = torch.linalg.ldl_factor_ex(view.contiguous(), hermitian=True)
        self.assertTrue(torch.equal(piv_v, piv_c))
        self.assertEqual(LD_v, LD_c)

    @parametrize("hermitian", [True, False])
    @parametrize("batch", [(), (2,)])
    @parametrize("n", [2, 5, 33, 150, 300])
    @dtypes(*floating_and_complex_types())
    def test_only_lower_triangle_is_read(self, device, dtype, n, batch, hermitian):
        """The strictly upper triangle is never read, and neither is the imaginary
        part of a Hermitian diagonal (LAPACK ignores it). The kernel mirrors the
        lower triangle into the upper one by adding to it, so junk there would leak
        into the result unless it is cleared first."""
        torch.manual_seed(n)
        A = symmetrize(indefinite(n, batch, dtype, device), hermitian)
        junk = make_tensor((*batch, n, n), dtype=dtype, device=device)
        dirty = A.tril() + junk.triu(1)
        if hermitian and dtype.is_complex:
            dirty.diagonal(dim1=-2, dim2=-1).add_(1j * junk.diagonal(dim1=-2, dim2=-1).real)
        clean_LD, clean_piv, clean_info = torch.linalg.ldl_factor_ex(A.tril(), hermitian=hermitian)
        LD, piv, info = torch.linalg.ldl_factor_ex(dirty, hermitian=hermitian)
        self.assertTrue(torch.equal(piv, clean_piv))
        self.assertTrue(torch.equal(info, clean_info))
        self.assertEqual(LD.tril(), clean_LD.tril())

    @parametrize("bad", [float("nan"), float("inf"), -float("inf")])
    @parametrize("batch", [(), (3,)])
    @parametrize("n", [1, 2, 33, 65, 300, 4100])
    @dtypes(torch.complex64, torch.complex128)
    def test_nonfinite_imag_diagonal_is_ignored(self, device, dtype, n, batch, bad):
        """The imaginary part of a Hermitian diagonal is ignored even when it is not
        finite: it must reach neither the real part (e.g. via beta * C = (1 + 0i) * C
        in the trailing GEMM, as 0 * inf is NaN) nor the pivot search."""
        torch.manual_seed(n)
        A = indefinite(n, batch, dtype, device)
        dirty = A.clone()
        dirty.diagonal(dim1=-2, dim2=-1).imag.fill_(bad)
        clean_LD, clean_piv, clean_info = torch.linalg.ldl_factor_ex(A, hermitian=True)
        LD, piv, info = torch.linalg.ldl_factor_ex(dirty, hermitian=True)
        self.assertTrue(torch.equal(piv, clean_piv))
        self.assertTrue(torch.equal(info, clean_info))
        self.assertEqual(LD.tril(), clean_LD.tril())

    @parametrize("n", [33, 65])
    @dtypes(torch.float64, torch.complex128)
    def test_batched_leading_dimension(self, device, dtype, n):
        torch.manual_seed(n)
        big = indefinite(n + 47, (3,), dtype, device)
        view = big[:, :n, :n]
        LD_v, piv_v, _ = torch.linalg.ldl_factor_ex(view, hermitian=True)
        LD_c, piv_c, _ = torch.linalg.ldl_factor_ex(view.contiguous(), hermitian=True)
        self.assertTrue(torch.equal(piv_v, piv_c))
        self.assertEqual(LD_v, LD_c)

    @parametrize("n", [33, 150])
    @dtypes(torch.float64, torch.complex128)
    def test_noncontiguous_batch(self, device, dtype, n):
        """Batch dims that are strided or permuted, not just the matrix dims."""
        torch.manual_seed(n)
        A = indefinite(n, (4, 6), dtype, device)
        for name, view in (("strided", A[::2]), ("transposed", A.transpose(0, 1)), ("both", A[::2, ::3])):
            with self.subTest(view=name):
                self.assertFalse(view.is_contiguous())
                LD_v, piv_v, info_v = torch.linalg.ldl_factor_ex(view, hermitian=True)
                LD_c, piv_c, info_c = torch.linalg.ldl_factor_ex(view.contiguous(), hermitian=True)
                self.assertTrue(torch.equal(piv_v, piv_c))
                self.assertTrue(torch.equal(info_v, info_c))
                self.assertEqual(LD_v, LD_c)

    @parametrize("kind", ["column", "single", "all"])
    @parametrize("bad", [float("nan"), float("inf")])
    @parametrize("n", [5, 40, 300, 4100])
    @dtypes(*floating_and_complex_types())
    def test_nonfinite_input(self, device, dtype, n, bad, kind):
        """NaN/Inf in the input must neither crash nor leave pivots unwritten. An
        all-NaN column once made the pivot search return index -1, which was used
        as a column index and read off the front of the buffer."""
        torch.manual_seed(n)
        A = indefinite(n, (), dtype, device)
        if kind == "column":
            A[1:, 0] = bad
            A[0, 1:] = bad
        elif kind == "single":
            A[n // 2, 1] = bad
            A[1, n // 2] = bad
        else:
            A.fill_(bad)
        _, piv, info = torch.linalg.ldl_factor_ex(A, hermitian=True)
        piv, info = piv.cpu(), info.item()
        self.assertTrue((piv.abs() >= 1).all(), f"pivot with |p| < 1: {piv}")
        self.assertTrue((piv.abs() <= n).all(), f"pivot out of range: {piv}")
        self.assertTrue(0 <= info <= n)

    @onlyCUDA
    @parametrize("n", [33, 4100])
    @dtypes(torch.float64, torch.complex128)
    def test_non_default_stream(self, device, dtype, n):
        torch.manual_seed(n)
        A = indefinite(n, (), dtype, device)
        ref_LD, ref_piv, _ = torch.linalg.ldl_factor_ex(A, hermitian=True)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            LD, piv, _ = torch.linalg.ldl_factor_ex(A, hermitian=True)
        side.synchronize()
        self.assertTrue(torch.equal(piv, ref_piv))
        self.assertTrue(torch.equal(LD, ref_LD))

    @onlyCUDA
    @parametrize("n", [33, 4100])
    @dtypes(torch.float64, torch.complex128)
    def test_other_device(self, device, dtype, n):
        """The input lives on another GPU than the current one."""
        if torch.cuda.device_count() < 2:
            self.skipTest("needs two GPUs")
        torch.manual_seed(n)
        A = indefinite(n, (), dtype, "cuda:0")
        ref_LD, ref_piv, _ = torch.linalg.ldl_factor_ex(A, hermitian=True)
        with torch.cuda.device(0):
            LD, piv, _ = torch.linalg.ldl_factor_ex(A.to("cuda:1"), hermitian=True)
        self.assertTrue(torch.equal(piv.cpu(), ref_piv.cpu()))
        self.assertTrue(torch.equal(LD.cpu(), ref_LD.cpu()))

    @onlyCUDA
    @parametrize("n", [33, 4100])
    @dtypes(torch.float64)
    def test_cuda_graph_capture(self, device, dtype, n):
        """The panel loop reads the step back to the host, so capture is not
        expected to work. It has to fail cleanly, leaving the device usable, or
        else reproduce the eager result."""
        torch.manual_seed(n)
        A = indefinite(n, (), dtype, device)
        ref_LD, ref_piv, _ = torch.linalg.ldl_factor_ex(A, hermitian=True)
        static = A.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            torch.linalg.ldl_factor_ex(static, hermitian=True)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                out = torch.linalg.ldl_factor_ex(static, hermitian=True)
        except RuntimeError:
            out = None
        torch.cuda.synchronize()
        if out is not None:
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(out.pivots, ref_piv))
            self.assertTrue(torch.equal(out.LD, ref_LD))
        LD, piv, _ = torch.linalg.ldl_factor_ex(A, hermitian=True)
        self.assertTrue(torch.equal(piv, ref_piv))
        self.assertTrue(torch.equal(LD, ref_LD))

    @parametrize("batch", [(16,), (64,), (4, 8)])
    @dtypes(torch.complex128)
    def test_large_batch(self, device, dtype, batch):
        torch.manual_seed(0)
        A = indefinite(33, batch, dtype, device)
        self._check(A, hermitian=True)

    @parametrize("shape", [(0, 0), (3, 0, 0), (0, 5, 5)])
    @dtypes(torch.float64, torch.complex128)
    def test_empty(self, device, dtype, shape):
        A = torch.zeros(*shape, dtype=dtype, device=device)
        LD, piv, info = torch.linalg.ldl_factor_ex(A, hermitian=True)
        self.assertEqual(LD.shape, A.shape)
        self.assertEqual(piv.shape, A.shape[:-1])
        self.assertEqual(info.shape, A.shape[:-2])

    @parametrize("batch", [(), (3,), (2, 3), (0,), (2, 0), (0, 3)])
    @dtypes(*floating_and_complex_types())
    def test_n1(self, device, dtype, batch):
        """n == 1 is a shortcut: pivot is 1 and info flags an exactly zero entry. As
        in LAPACK (the CPU instance) and MAGMA, the imaginary part of a Hermitian
        diagonal is ignored, so a purely imaginary entry is singular too."""
        A = make_tensor((*batch, 1, 1), dtype=dtype, device=device)
        A.view(-1)[::2] = 0
        if dtype.is_complex:
            A.view(-1)[1::4] = 1j
        LD, piv, info = torch.linalg.ldl_factor_ex(A, hermitian=True)
        expected = A.real.to(dtype) if dtype.is_complex else A
        self.assertEqual(LD, expected)
        self.assertEqual(piv, torch.ones_like(piv))
        self.assertEqual(info, (expected == 0).squeeze(-1).squeeze(-1).to(torch.int32))

    @parametrize("batch", [(), (3,), (2, 3), (0,), (2, 0), (0, 3)])
    @dtypes(*floating_and_complex_types())
    def test_n0(self, device, dtype, batch):
        A = torch.empty(*batch, 0, 0, dtype=dtype, device=device)
        LD, piv, info = torch.linalg.ldl_factor_ex(A, hermitian=True)
        self.assertEqual(LD.shape, A.shape)
        self.assertEqual(piv.shape, (*batch, 0))
        self.assertEqual(info, torch.zeros(batch, dtype=torch.int32, device=device))

    @parametrize("n", [33, 64])
    @dtypes(torch.float64, torch.complex128)
    def test_out_variant(self, device, dtype, n):
        torch.manual_seed(n)
        A = indefinite(n, (), dtype, device)
        LD = torch.empty_like(A)
        piv = torch.empty(A.shape[:-1], dtype=torch.int32, device=device)
        torch.linalg.ldl_factor(A, hermitian=True, out=(LD, piv))
        eLD, epiv = torch.linalg.ldl_factor(A, hermitian=True)
        self.assertEqual(LD, eLD)
        self.assertTrue(torch.equal(piv, epiv))

    @onlyCUDA
    @parametrize("n", [257, 513, 1023])
    @dtypes(torch.float32, torch.complex64)
    def test_large_single_precision(self, device, dtype, n):
        """Single precision at sizes where element growth has room to bite."""
        torch.manual_seed(n)
        A = indefinite(n, (), dtype, device)
        self._check(A, hermitian=True)

    @onlyCUDA
    @parametrize("n", [33, 64, 127])
    @dtypes(torch.complex128)
    def test_inverse_via_solve(self, device, dtype, n):
        """Solving against A itself must return the identity."""
        torch.manual_seed(n)
        A = indefinite(n, (), dtype, device)
        LD, piv, _ = torch.linalg.ldl_factor_ex(A, hermitian=True)
        X = torch.linalg.ldl_solve(LD.cpu(), piv.cpu(), A.cpu(), hermitian=True)
        tol = 200 * n * torch.finfo(dtype).eps
        self.assertEqual(X, torch.eye(n, dtype=dtype), rtol=tol, atol=tol)

    @parametrize("k", [0, 1, 5, 31, 32, 33, 60])
    @parametrize("n", [64, 150])
    @dtypes(*floating_and_complex_types())
    def test_zero_row_col_completes(self, device, dtype, n, k):
        """A zero row/column makes the matrix singular without stopping the work.

        The factorization must run to completion -- every pivot slot written,
        the panel loop reaching the end -- and report an info in range. The base
        matrix is indefinite so 2x2 blocks are in play, which is the point: the
        info bookkeeping sits in the panel kernel beside the 2x2 handling.

        info is NOT required to be nonzero. A zero row/column does not
        necessarily produce an exactly zero pivot: the singularity can surface
        a hundred columns later through accumulation, where CPU may cancel to
        exactly 0.0 while CUDA lands on 1e-17. Both honour the same LAPACK
        contract -- info is set only on an exactly zero pivot -- so requiring it
        here would be demanding exact cancellation that neither backend
        guarantees. test_exact_zero_pivot_sets_info covers the deterministic case.
        """
        torch.manual_seed(n)
        A = indefinite(n, (), dtype, device)
        _, base_piv, _ = torch.linalg.ldl_factor_ex(A, hermitian=True)
        self.assertGreater(sum(1 for v in base_piv.cpu().tolist() if v < 0), 0)

        A[k, :] = 0
        A[:, k] = 0
        _, piv, info = torch.linalg.ldl_factor_ex(A, hermitian=True)

        # completed: every pivot slot holds a usable 1-based index
        self.assertTrue((piv.abs() >= 1).all(), f"pivot with |p| < 1: {piv}")
        self.assertTrue((piv.abs() <= n).all(), f"pivot out of range: {piv}")
        self.assertGreaterEqual(info.item(), 0)
        self.assertLessEqual(info.item(), n)

    # (n, planted interchanges): adjacent, far, several, across panel boundaries,
    # and out of panel on the cooperative path
    _SWAP_LAYOUTS = {
        "adjacent": (8, [(0, 1)]),
        "far": (70, [(0, 40)]),
        "two": (70, [(10, 60), (20, 50)]),
        "panel_boundary": (300, [(31, 33), (5, 200), (100, 150)]),
        "cooperative": (2100, [(0, 2000), (31, 2050)]),
    }

    @parametrize("layout", list(_SWAP_LAYOUTS))
    @dtypes(*floating_and_complex_types())
    def test_zero_last_pivot_after_interchanges(self, device, dtype, layout):
        """Same as test_zero_last_pivot_from_updates, but the updates reach the
        last entry through columns that were interchanged first."""
        n, swaps = self._SWAP_LAYOUTS[layout]
        for seed in range(2):
            A = exact_last_zero_matrix(n, dtype, device, seed, swaps)
            _, piv, info = torch.linalg.ldl_factor_ex(A, hermitian=True)
            self.assertEqual([piv[a].item() for a, _ in swaps], [b + 1 for _, b in swaps])
            self.assertEqual(info.item(), n)
            A[-1, -1] += 0.5
            _, _, info = torch.linalg.ldl_factor_ex(A, hermitian=True)
            self.assertEqual(info.item(), 0)

    @onlyCUDA
    @parametrize("hermitian", [True, False])
    @parametrize("n", [5, 33, 150, 600, 2049])
    @dtypes(torch.float32, torch.float64)
    def test_solve_on_cuda(self, device, dtype, n, hermitian):
        """ldl_solve runs on CUDA for real dtypes. Its input is what the custom
        kernel produced, so the pivot encoding and the layout of LD have to be the
        ones cuSOLVER's solver expects. (Complex+Hermitian has no CUDA solver.)"""
        torch.manual_seed(n)
        A = indefinite(n, (), dtype, device)
        B = make_tensor((n, 3), dtype=dtype, device=device)
        LD, piv, _ = torch.linalg.ldl_factor_ex(A, hermitian=hermitian)
        X = torch.linalg.ldl_solve(LD, piv, B, hermitian=hermitian)
        self.assertLess(_lapack_residual(A, X, B), _resid_threshold(A, B))

    @dtypes(torch.float16, torch.bfloat16)
    def test_unsupported_dtype(self, device, dtype):
        with self.assertRaises(NotImplementedError):
            torch.linalg.ldl_factor_ex(torch.eye(4, dtype=dtype, device=device), hermitian=True)

    @onlyCUDA
    @parametrize("structure", ["indefinite", "zero_diagonal", "out_of_panel_heavy", "clustered",
                               "tiny_diagonal", "integer_valued", "complex_real_valued", "ill_conditioned"])
    @parametrize("n", [33, 127, 513, 1025, 2049])
    @dtypes(*floating_and_complex_types())
    def test_lapack_residual(self, device, dtype, n, structure):
        """The scaled residual stays under LAPACK's own test threshold, or close to
        that of an LU solve where both exceed it (see _resid_threshold), so a loss of
        a few digits would show."""
        torch.manual_seed(n)
        A = STRUCTURES[structure](n, (), dtype, device)
        torch.manual_seed(0)
        self.assertLess(*_residual_after_solve(A, *torch.linalg.ldl_factor_ex(A, hermitian=True)[:2]))

    # huge_scale and tiny_scale stay far inside the range where a product of two
    # entries is finite and normal, i.e. sqrt(finfo.tiny) to sqrt(finfo.max).
    @parametrize("structure", ["indefinite", "zero_diagonal"])
    @parametrize("n", [2, 64, 300])
    @dtypes(*floating_and_complex_types())
    def test_extreme_scale(self, device, dtype, n, structure):
        """Neither the pivot test nor the inverse of a 2x2 block may form a product
        of two entries: past sqrt(finfo.max) it overflows into finite garbage with
        info == 0, below sqrt(finfo.tiny) it underflows into a false info > 0."""
        exponents = (19, 21, 23, 30) if _real_dtype(dtype) == torch.float32 else (153, 160, 165, 250)
        torch.manual_seed(n)
        A = STRUCTURES[structure](n, (), dtype, device)
        for e in exponents:
            for scale in (10.0**e, 10.0**-e):
                with self.subTest(scale=scale):
                    self._check(A * scale, hermitian=True)

    @parametrize("n", [2, 5, 33, 34, 65, 150, 2049, 4100])
    @dtypes(*floating_and_complex_types())
    def test_zero_last_pivot_from_updates(self, device, dtype, n):
        """The last pivot is exactly zero only after every earlier column has
        updated it, so it is stale when the last panel ends, and the check has to
        run on the value after the final GEMM."""
        for seed in range(3):
            A = exact_last_zero_matrix(n, dtype, device, seed)
            _, piv, info = torch.linalg.ldl_factor_ex(A, hermitian=True)
            self.assertEqual(info.item(), n)
            self.assertEqual(piv.tolist(), list(range(1, n + 1)))
            # nudging the last entry by an exactly representable amount must clear it
            A[-1, -1] += 0.5
            _, _, info = torch.linalg.ldl_factor_ex(A, hermitian=True)
            self.assertEqual(info.item(), 0)

    @parametrize("n", [2, 3, 5, 32, 33, 34, 40, 65, 150, 2049, 4100])
    @dtypes(*floating_and_complex_types())
    def test_zero_last_pivot_sets_info(self, device, dtype, n):
        """The last pivot has no column of its own to be factored, so singularity
        there has to be checked separately. info must name it, as LAPACK does."""
        A = torch.diag(torch.arange(1, n + 1, device=device).to(dtype))
        A[-1, -1] = 0
        _, piv, info = torch.linalg.ldl_factor_ex(A, hermitian=True)
        self.assertEqual(info.item(), n)
        self.assertEqual(piv.tolist(), list(range(1, n + 1)))

    # The last column is handled by a launch of its own, or by the panel that ends on
    # it, depending on where the panel boundaries and 2x2 blocks fall. These layouts
    # cover the last pivot being a 1x1 or a 2x2 block, and a 2x2 block straddling a
    # panel boundary (NB = 32) right before it.
    _LAST_COLUMN_PATTERNS = {
        "all_1x1": lambda n: (1,) * n,
        "ends_2x2": lambda n: (1,) * (n - 2) + (2,),
        "straddle_then_1x1": lambda n: (1,) * (NB - 1) + (2,) + (1,) * (n - NB - 1),
        "straddle_then_2x2": lambda n: (1,) * (NB - 1) + (2,) + (1,) * (n - NB - 3) + (2,),
        "2x2_before_last": lambda n: (1,) * (n - 3) + (2,) + (1,),
        # a 2x2 block across every multiple of the period, which is where the panels of the
        # various dtypes end (256 bytes worth of columns: 64, 32, 32 and 16)
        "straddle_every_16": lambda n: _straddling_pattern(n, 16),
        "straddle_every_32": lambda n: _straddling_pattern(n, 32),
        "straddle_every_64": lambda n: _straddling_pattern(n, 64),
    }

    @parametrize("layout", list(_LAST_COLUMN_PATTERNS))
    @parametrize("n", [34, 35, 36, 64, 65, 66, 97, 129, 2049, 2050, 4100])
    @dtypes(*floating_and_complex_types())
    def test_last_column_layouts(self, device, dtype, n, layout):
        pattern = self._LAST_COLUMN_PATTERNS[layout](n)
        if sum(pattern) != n or min(pattern) < 1:
            self.skipTest(f"{layout} does not fit n={n}")
        torch.manual_seed(n)
        A = matrix_for_pattern(pattern, dtype, device)
        _, piv, info = torch.linalg.ldl_factor_ex(A, hermitian=True)
        self.assertEqual(info.item(), 0)
        self.assertEqual(pattern_from_pivots(piv.cpu()), pattern)
        self.assertEqual(piv[-1].item(), n if pattern[-1] == 1 else -n)
        self._check(A, hermitian=True)

        if pattern[-1] == 1:
            # same layout with the last pivot exactly zero
            A[-1, -1] = 0
            _, piv, info = torch.linalg.ldl_factor_ex(A, hermitian=True)
            self.assertEqual(info.item(), n)
            self.assertEqual(pattern_from_pivots(piv.cpu()), pattern)

    @parametrize("n", [33, 65, 150])
    @dtypes(torch.float64, torch.complex128)
    def test_zero_last_pivot_keeps_first_info(self, device, dtype, n):
        """info names the first singular pivot, so a zero last pivot must not
        overwrite an earlier one."""
        A = torch.diag(torch.arange(1, n + 1, device=device).to(dtype))
        A[n // 2, n // 2] = 0
        A[-1, -1] = 0
        _, _, info = torch.linalg.ldl_factor_ex(A, hermitian=True)
        self.assertEqual(info.item(), n // 2 + 1)

    @parametrize("n", [33, 65])
    @dtypes(torch.float64, torch.complex128)
    def test_tiny_last_pivot_is_not_singular(self, device, dtype, n):
        A = torch.diag(torch.arange(1, n + 1, device=device).to(dtype))
        A[-1, -1] = torch.finfo(torch.float64).tiny
        _, _, info = torch.linalg.ldl_factor_ex(A, hermitian=True)
        self.assertEqual(info.item(), 0)

    @parametrize("n", [2, 33, 65])
    @dtypes(torch.float64, torch.complex128)
    def test_zero_last_pivot_batched(self, device, dtype, n):
        """info is per batch element: only the flagged ones may report n."""
        A = torch.diag_embed(torch.arange(1, n + 1, device=device).to(dtype).expand(4, n)).clone()
        for i in (1, 3):
            A[i, -1, -1] = 0
        _, piv, info = torch.linalg.ldl_factor_ex(A, hermitian=True)
        self.assertEqual(info.tolist(), [0, n, 0, n])
        self.assertEqual(piv.tolist(), [list(range(1, n + 1))] * 4)

    @parametrize("batch", [(), (3,)])
    @parametrize("n", [2, 33, 64, 65, 150, 257])
    @dtypes(torch.complex64, torch.complex128)
    def test_hermitian_D_is_real(self, device, dtype, n, batch):
        """A Hermitian D has a real diagonal, as LAPACK returns it, including the
        last entry, which only the trailing GEMM touches. The input carries an
        imaginary part on its diagonal, which LAPACK ignores. Holds for the diagonal
        of 2x2 blocks too."""
        torch.manual_seed(n)
        A = indefinite(n, batch, dtype, device)
        A.diagonal(dim1=-2, dim2=-1).add_(0.5j * torch.rand(*batch, n, device=device))
        LD, _, _ = torch.linalg.ldl_factor_ex(A, hermitian=True)
        self.assertEqual(LD.diagonal(dim1=-2, dim2=-1).imag, torch.zeros(*batch, n, device=device, dtype=LD.diagonal().imag.dtype))

    @parametrize("n", [16, 64, 150])
    @dtypes(*floating_and_complex_types())
    def test_exact_zero_pivot_sets_info(self, device, dtype, n):
        """When the zero pivot is exact by construction, info must report it.

        Zeroing row/column 0 puts an exact zero at the very first pivot, before
        any arithmetic has had a chance to perturb it, so there is no dependence
        on cancellation and the answer is backend-independent.
        """
        torch.manual_seed(n)
        A = indefinite(n, (), dtype, device)
        A[0, :] = 0
        A[:, 0] = 0
        _, piv, info = torch.linalg.ldl_factor_ex(A, hermitian=True)
        self.assertEqual(info.item(), 1)
        self.assertTrue((piv.abs() >= 1).all())
        self.assertTrue((piv.abs() <= n).all())

    @parametrize("n", [64, 150])
    @dtypes(torch.float64, torch.complex128)
    def test_zero_row_col_batched(self, device, dtype, n):
        """info is reported per batch element, independently.

        Uses row/column 0 so each flagged element has an exact zero pivot -- see
        test_exact_zero_pivot_sets_info for why later positions are unreliable.
        """
        torch.manual_seed(n)
        A = indefinite(n, (3,), dtype, device)
        for i in (0, 2):
            A[i, 0, :] = 0
            A[i, :, 0] = 0
        _, piv, info = torch.linalg.ldl_factor_ex(A, hermitian=True)
        self.assertEqual([v.item() > 0 for v in info.cpu()], [True, False, True])
        self.assertTrue((piv.abs() >= 1).all())

    @parametrize("n", [64, 150])
    @dtypes(torch.float64, torch.complex128)
    def test_info_zero_when_nonsingular(self, device, dtype, n):
        """Merely near-singular is not reported: the test is an exact zero pivot,
        so a tiny-but-nonzero eigenvalue must still give info == 0."""
        torch.manual_seed(n)
        A = ill_conditioned(n, (), dtype, device)
        _, _, info = torch.linalg.ldl_factor_ex(A, hermitian=True)
        self.assertEqual(info.item(), 0)

    @onlyCUDA
    @dtypes(torch.float64, torch.complex128)
    def test_singular_reports_info(self, device, dtype):
        """An exactly singular input must be reported through info rather than
        silently producing a factorization."""
        n = 40
        A = torch.zeros(n, n, dtype=dtype, device=device)
        _, _, info = torch.linalg.ldl_factor_ex(A, hermitian=True)
        self.assertGreater(info.item(), 0)
        # NOTE: check_errors=True currently raises an INTERNAL ASSERT rather than
        # a clean error for this info code -- reproduces on CPU too, so it is a
        # pre-existing gap in the error mapping, not a property of this kernel.


instantiate_device_type_tests(TestLDLCustomKernel, globals())

if __name__ == "__main__":
    run_tests()
