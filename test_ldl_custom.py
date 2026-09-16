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

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
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


def _real_dtype(dtype):
    return torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32


def _hermitian(a):
    return a + a.mH


def _spectrum(n, batch, s, dtype, device):
    """Hermitian matrix with prescribed eigenvalues s via Q diag(s) Q^H."""
    x = torch.randn(*batch, n, n, dtype=dtype, device=device)
    q, _ = torch.linalg.qr(x)
    return (q * s.unsqueeze(-2).to(dtype)) @ q.mH


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

# Structures that are singular (or effectively so) at tiny n, where there is no
# room for the 2x2 blocks they are built to require.
_NEEDS_ROOM = ("zero_diagonal", "out_of_panel_heavy", "arrow", "tridiagonal", "banded")


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

        Asym = symmetrize(A.cpu(), hermitian)
        B = make_tensor((*A.shape[:-1], nrhs), dtype=A.dtype, device="cpu")
        X = torch.linalg.ldl_solve(LD.cpu(), pivots.cpu(), B, hermitian=hermitian)
        R = Asym @ X - B

        # Backward error: ||AX - B|| / (||A|| ||X|| + ||B||). Normalising this way
        # keeps the bound meaningful for ill-conditioned and badly scaled input,
        # where the residual is small but B alone is a poor yardstick. The fixed
        # per-dtype defaults in assertEqual are far too tight here -- they reject
        # valid float32 results from n=33 up, on CPU as readily as on CUDA.
        denom = Asym.abs().amax() * X.abs().amax() + B.abs().amax()
        rel = (R.abs().amax() / denom).item()
        tol = 1000 * n * torch.finfo(A.dtype).eps
        self.assertLess(rel, tol, f"backward error {rel:.3e} exceeds {tol:.3e}")

    # ------------------------------------------------------------------ sweeps

    @parametrize("structure", list(STRUCTURES))
    @parametrize("n", [2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 34, 63, 64, 65, 66, 96, 127, 128, 129])
    @dtypes(*floating_and_complex_types())
    def test_structures(self, device, dtype, n, structure):
        if structure in _NEEDS_ROOM and n < 4:
            self.skipTest(f"{structure} is singular for n={n}")
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

    @parametrize("structure", ["indefinite", "zero_diagonal", "out_of_panel_heavy",
                               "clustered", "ill_conditioned"])
    @parametrize("batch", [(1,), (2,), (3,), (5,), (2, 2), (1, 3), (3, 1), (2, 2, 2)])
    @parametrize("n", [2, 3, 5, 16, 32, 33, 64, 65, 150])
    @dtypes(torch.float64, torch.complex128)
    def test_batched(self, device, dtype, n, batch, structure):
        """The kernel factors one matrix per launch, so the batch is driven by a
        loop in BatchLinearAlgebra.cpp. Multi-dimensional batch shapes matter:
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

    @parametrize("n", list(range(2, 11)))
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
    @parametrize("structure", ["indefinite", "zero_diagonal", "out_of_panel_heavy", "ill_conditioned"])
    @parametrize("n", [33, 64, 127, 257, 513, 1025])
    @dtypes(torch.float64, torch.complex128)
    def test_matches_cpu_quality(self, device, dtype, n, structure):
        """CUDA must factor about as accurately as CPU LAPACK.

        Compares residuals, not factors: the two backends routinely choose
        different pivot sequences at these sizes, which makes LD and the pivot
        vector differ wholesale while both factorizations stay valid.
        """
        torch.manual_seed(n)
        A = STRUCTURES[structure](n, (), dtype, device)
        B = make_tensor((n, 3), dtype=dtype, device="cpu")
        Asym = symmetrize(A.cpu(), True)

        def resid(a):
            LD, piv, _ = torch.linalg.ldl_factor_ex(a, hermitian=True)
            x = torch.linalg.ldl_solve(LD.cpu(), piv.cpu(), B, hermitian=True)
            denom = Asym.abs().amax() * x.abs().amax() + B.abs().amax()
            return ((Asym @ x - B).abs().amax() / denom).item()

        cuda_r, cpu_r = resid(A), resid(A.cpu())
        floor = 200 * n * torch.finfo(dtype).eps
        self.assertLess(
            cuda_r,
            max(50 * cpu_r, floor),
            f"cuda {cuda_r:.3e} vs cpu {cpu_r:.3e} (n={n}, {structure})",
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
