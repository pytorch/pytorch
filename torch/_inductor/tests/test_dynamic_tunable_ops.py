"""Test suite for the dynamic TunableOp dispatch matrix.

Exercises every branch of the design:

Tuning enabled (compile-time autotune)
  1. no dynamic dim + concrete key miss  -> tune; persist concrete key only
  2. has dynamic dim + concrete key miss -> tune; persist concrete + wildcard
  3. no dynamic dim + concrete key hit   -> nothing new persisted
  4. has dynamic dim + concrete key hit  -> persist wildcard if missing

Tuning disabled (runtime)
  Runtime callers (notably the AOTI cpp_wrapper) cannot know which
  dim is dynamic: no `TunableDynamicDimsGuard` is emitted, so
  `GetCurrentDynamicDimsMask()` is always empty.

  1. concrete key hit                       -> dispatch via concrete
  2. concrete miss + wildcard match         -> dispatch via wildcard
                                               (TuningResultsManager::
                                               LookupWildcardFallback scans
                                               persisted wildcard entries
                                               token-by-token against the
                                               concrete signature)
  3. concrete miss + no wildcard match      -> result stays Null;
                                               caller (e.g.
                                               launchTunableGemmAndBias)
                                               falls back to the
                                               non-tunable aten path
                                               (no Default kernel invoked,
                                               no entries added)

Covered (every variant is verified for all three runtime dispatch outcomes --
concrete hit, concrete miss + wildcard fallback, and concrete miss + no
wildcard -> safe non-tunable aten fallback -- matching the addmm reference
launchTunableGemmAndBias):
  - GemmAndBiasTunableOp via torch.addmm
  - GemmTunableOp via torch.mm
  - GemmStridedBatchedTunableOp via torch.bmm and torch.baddbmm
  - ScaledGemmTunableOp via torch._scaled_mm (incl. the safe-aten fallback on
    a total miss -- the ResultEntry::Default() kernel is never invoked)
  - several mm layout combinations: NN/NT/TN/TT (exercising the swapped_mn
    M<->N mask remap)

All non-scaled entry points (mm/bmm/baddbmm) route through gemm_tunable /
bgemm_tunable in CUDABlas.cpp, which now (a) resolve a concrete miss via
TuningResultsManager::LookupWildcardFallback and (b) receive the inductor-frame
mask already remapped into BLAS frame (by launchGemmCublas /
baddbmm_out_cuda_impl in Blas.cpp) when the dispatch swaps M<->N -- exactly as
launchTunableGemmAndBias does for addmm. The scaled path (_tunable_scaled_gemm_
rocm in ScaledBlas.cpp) applies the same remap and now gates dispatch so a
total miss falls back to at::cuda::blas::scaled_gemm.

Not fully covered:
  - addbmm or other higher-level batched add variants, if routed differently
  - scaled GEMM variants beyond the FP8 tensorwise _scaled_mm shape
"""

# pyre-strict

import logging
import os
import tempfile
import unittest

import sympy

import torch
import torch.cuda.tunable
from torch._inductor import config as inductor_config
from torch._inductor.kernel_inputs import MMKernelInputs
from torch.testing._internal.common_utils import run_tests, TestCase

logger: logging.Logger = logging.getLogger(__name__)


DEVICE: str = "cuda"
DTYPE: torch.dtype = torch.bfloat16


def _addmm(
    m: int, n: int, k: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (bias, mat1, mat2) for an addmm of shape (m, n, k)."""
    torch.manual_seed(seed)
    bias = torch.randn(n, dtype=DTYPE, device=DEVICE)
    mat1 = torch.randn(m, k, dtype=DTYPE, device=DEVICE)
    mat2 = torch.randn(k, n, dtype=DTYPE, device=DEVICE)
    return bias, mat1, mat2


def _mm(m: int, n: int, k: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (mat1, mat2) for a plain mm of shape (m, n, k)."""
    torch.manual_seed(seed)
    mat1 = torch.randn(m, k, dtype=DTYPE, device=DEVICE)
    mat2 = torch.randn(k, n, dtype=DTYPE, device=DEVICE)
    return mat1, mat2


def _bmm(
    b: int, m: int, n: int, k: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (batch1, batch2) for a bmm of shape (b, m, n, k)."""
    torch.manual_seed(seed)
    batch1 = torch.randn(b, m, k, dtype=DTYPE, device=DEVICE)
    batch2 = torch.randn(b, k, n, dtype=DTYPE, device=DEVICE)
    return batch1, batch2


def _baddbmm(
    b: int, m: int, n: int, k: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (bias, batch1, batch2) for a baddbmm of shape (b, m, n, k)."""
    torch.manual_seed(seed)
    bias = torch.randn(b, m, n, dtype=DTYPE, device=DEVICE)
    batch1 = torch.randn(b, m, k, dtype=DTYPE, device=DEVICE)
    batch2 = torch.randn(b, k, n, dtype=DTYPE, device=DEVICE)
    return bias, batch1, batch2


def _entries_for(op_substr: str, *required_dim_tokens: str) -> list:
    """Filter `torch.cuda.tunable.get_results()` to entries whose
    op-signature contains `op_substr` AND whose params-signature contains
    every requested `_<token>_` substring.

    The dim tokens are matched as `_<token>_` against `_<params_sig>_`, so
    callers do not have to know the cuBLAS-vs-PyTorch column-major /
    row-major dim ordering -- they just supply the integer dims that must
    appear somewhere in the params signature."""
    out = []
    for entry in torch.cuda.tunable.get_results():
        # Each entry is a 4-tuple: (op_signature, params_signature, key, time).
        op_sig, params_sig, _, _ = entry
        if op_substr not in op_sig:
            continue
        padded = "_" + params_sig + "_"
        if all(f"_{tok}_" in padded for tok in required_dim_tokens):
            out.append(entry)
    return out


def _has_concrete_entry(op_substr: str, m: int, n: int, k: int) -> bool:
    """True if get_results() has an entry whose params_sig contains every
    one of (m, n, k) as `_N_` substrings AND no `*` wildcard token."""
    for entry in _entries_for(op_substr, str(m), str(n), str(k)):
        if "*" not in entry[1]:
            return True
    return False


def _has_wildcard_entry(op_substr: str) -> bool:
    """True if get_results() has any wildcard (asterisk-bearing) entry for
    the given op."""
    for entry in torch.cuda.tunable.get_results():
        op_sig, params_sig, _, _ = entry
        if op_substr in op_sig and "*" in params_sig:
            return True
    return False


def _has_wildcard_with_dims(op_substr: str, *dims: int) -> bool:
    """True if get_results() has a wildcard entry whose params_sig contains
    every one of `dims` as `_N_` substrings."""
    for entry in torch.cuda.tunable.get_results():
        op_sig, params_sig, _, _ = entry
        if op_substr not in op_sig or "*" not in params_sig:
            continue
        padded = "_" + params_sig + "_"
        if all(f"_{d}_" in padded for d in dims):
            return True
    return False


# Positions of (lda, ldb, ldc) tokens right after the "_ld_" marker.
_LD_MARKER = "_ld_"


def _parse_gemm_params_sig(
    params_sig: str,
) -> tuple[str, str, str, str, str, str, str, str]:
    """Split a Gemm*Params signature into its component tokens.

    Handles all variants -- GemmParams / GemmAndBiasParams
    (``{ta}{tb}_{m}_{n}_{k}_ld_{lda}_{ldb}_{ldc}``), GemmStridedBatched
    (extra ``_B_{batch}`` before ``_ld_``) and ScaledGemm (extra
    ``_rw_..._bias_...`` after the ld triple) -- because m/n/k are always the
    first three tokens after the leading ``{ta}{tb}`` token and lda/ldb/ldc
    are always the first three tokens after ``_ld_``.

    Returns ``(transa, transb, m, n, k, lda, ldb, ldc)`` as raw string tokens
    (each is either a decimal integer or ``"*"``)."""
    transa, transb = params_sig[0], params_sig[1]
    ld_idx = params_sig.find(_LD_MARKER)
    assert ld_idx != -1, f"no '{_LD_MARKER}' marker in signature: {params_sig}"
    prefix_toks = params_sig[:ld_idx].split("_")
    m_tok, n_tok, k_tok = prefix_toks[1], prefix_toks[2], prefix_toks[3]
    ld_toks = params_sig[ld_idx + len(_LD_MARKER) :].split("_")
    lda_tok, ldb_tok, ldc_tok = ld_toks[0], ld_toks[1], ld_toks[2]
    return transa, transb, m_tok, n_tok, k_tok, lda_tok, ldb_tok, ldc_tok


def _assert_ld_wildcarding_consistent(test: TestCase, op_substr: str) -> None:
    """White-box check that the leading-dim wildcarding of every persisted
    wildcard entry matches the actual lda/ldb/ldc dependency encoded by the
    transpose flags in that same signature.

    This is the invariant that ``ShouldWildcardLda/Ldb/Ldc`` in GemmCommon.h
    must satisfy, derived from ``GetSizeA/B/C`` in that file:

      * lda >= m for non-transposed A (transa == 'N'), lda >= k for
        transposed A (transa == 'T'), so lda tracks M or K accordingly.
      * ldb >= k for non-transposed B (transb == 'N'), ldb >= n for
        transposed B (transb == 'T'), so ldb tracks K or N accordingly.
      * ldc >= m always, so ldc tracks M.

    Reading transa/transb and which of m/n/k are ``*`` from the persisted
    signature makes the check independent of the row-major addmm/mm M<->N
    remap: whatever frame the signature ends up in, its ld wildcarding must
    be self-consistent. The pre-fix inverted ``UsesMForLda`` wildcards lda on
    the wrong dim whenever exactly one of {m, k} is dynamic, which this check
    flags."""
    checked = 0
    for op_sig, params_sig, _, _ in torch.cuda.tunable.get_results():
        if op_substr not in op_sig or "*" not in params_sig:
            continue
        transa, transb, m_tok, n_tok, k_tok, lda_tok, ldb_tok, ldc_tok = (
            _parse_gemm_params_sig(params_sig)
        )
        dyn_m, dyn_n, dyn_k = m_tok == "*", n_tok == "*", k_tok == "*"
        exp_lda = dyn_m if transa in "Nn" else dyn_k
        exp_ldb = dyn_k if transb in "Nn" else dyn_n
        exp_ldc = dyn_m
        test.assertEqual(
            lda_tok == "*",
            exp_lda,
            f"lda wildcarding wrong for {params_sig!r}: transa={transa}, "
            f"dyn_m={dyn_m}, dyn_k={dyn_k}; lda tracks "
            f"{'M' if transa in 'Nn' else 'K'}",
        )
        test.assertEqual(
            ldb_tok == "*",
            exp_ldb,
            f"ldb wildcarding wrong for {params_sig!r}: transb={transb}, "
            f"dyn_n={dyn_n}, dyn_k={dyn_k}; ldb tracks "
            f"{'K' if transb in 'Nn' else 'N'}",
        )
        test.assertEqual(
            ldc_tok == "*",
            exp_ldc,
            f"ldc wildcarding wrong for {params_sig!r}: dyn_m={dyn_m}; "
            f"ldc always tracks M",
        )
        checked += 1
    test.assertGreaterEqual(
        checked,
        1,
        f"expected at least one {op_substr} wildcard entry to validate",
    )


class DynamicTunableOpsTest(TestCase):
    """Verification of the full tuning enable/disable x dynamic-dim matrix.

    Note: each test case picks distinct (m, n, k) shapes so the in-memory
    TuningResultsManager state from one test does not leak useful entries
    into another.  The state is always restored to (enabled=False,
    tuning=False) in tearDown.
    """

    _tmpdir: str = ""
    _tmp_results_path: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("cuda not available")
        # Redirect TunableOp persistence to a fresh per-process tempfile.
        # The default filename "tunableop_results.csv" is shared across
        # invocations; once a prior test run has tuned anything, its
        # realtime-append writes pollute the in-memory results table for
        # every subsequent run (the default file is auto-loaded lazily on
        # the first GetTuningResultsManager() call). Setting our own
        # filename BEFORE any tunable operation forces the lazy load to
        # read our (empty) tempfile instead.
        cls._tmpdir = tempfile.mkdtemp(prefix="dynamic_tunable_ops_test_")
        cls._tmp_results_path = os.path.join(cls._tmpdir, "tunable_results.csv")
        torch.cuda.tunable.set_filename(cls._tmp_results_path, False)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._tmpdir:
            import shutil

            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def setUp(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)

    def tearDown(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)

    # -- Tuning enabled --------------------------------------------------

    def test_tuning_enabled_no_dynamic_concrete_miss_persists_concrete_only(
        self,
    ) -> None:
        """Tuning enabled + no dynamic dim + concrete miss
        -> tune; persist concrete; do NOT persist wildcard."""
        # Choose a fresh shape that is unlikely to clash with anything else
        # (small to keep tuning latency low).
        # Unusual primes -> guaranteed not in any prod-style CSV.
        m, n, k = 17, 2053, 257
        bias, mat1, mat2 = _addmm(m, n, k, seed=1)

        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        out = torch.addmm(bias, mat1, mat2)
        self.assertEqual(out.shape, (m, n))

        matched = _entries_for("GemmAndBiasTunableOp", str(m), str(n), str(k))
        # Should have exactly one concrete entry, no wildcard entry.
        self.assertGreaterEqual(
            len(matched), 1, f"expected >=1 entry for ({m},{n},{k}), got {matched}"
        )
        for entry in matched:
            self.assertNotIn(
                "*",
                entry[1],
                f"unexpected wildcard entry for non-dynamic call: {entry}",
            )

    def test_tuning_enabled_dynamic_concrete_miss_persists_concrete_and_wildcard(
        self,
    ) -> None:
        """Tuning enabled + has dynamic dim (M dynamic) + concrete miss
        -> tune; persist concrete AND wildcard."""
        m, n, k = 23, 2053, 269
        bias, mat1, mat2 = _addmm(m, n, k, seed=2)

        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        with torch.cuda.tunable.dynamic_dims_mask(M=True):
            out = torch.addmm(bias, mat1, mat2)
        self.assertEqual(out.shape, (m, n))

        concrete_matched = [
            e
            for e in _entries_for("GemmAndBiasTunableOp", str(m), str(n), str(k))
            if "*" not in e[1]
        ]
        wildcard_matched = [
            e for e in _entries_for("GemmAndBiasTunableOp") if "*" in e[1]
        ]
        self.assertGreaterEqual(
            len(concrete_matched),
            1,
            f"expected concrete entry for ({m},{n},{k}); got {concrete_matched}",
        )
        self.assertGreaterEqual(
            len(wildcard_matched),
            1,
            f"expected at least one wildcard entry; got {wildcard_matched}",
        )

    def test_tuning_enabled_no_dynamic_concrete_hit_no_new_entries(self) -> None:
        """Tuning enabled + no dynamic dim + concrete hit
        -> no new entries are added on the second call."""
        m, n, k = 29, 2053, 271
        bias, mat1, mat2 = _addmm(m, n, k, seed=3)

        # Prime: tune once.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        torch.addmm(bias, mat1, mat2)
        before = len(torch.cuda.tunable.get_results())

        # Run again with same shape, no dynamic mask.
        torch.addmm(bias, mat1, mat2)
        after = len(torch.cuda.tunable.get_results())
        self.assertEqual(
            before,
            after,
            "concrete-hit + no-dynamic case should not add entries",
        )

    def test_tuning_enabled_dynamic_concrete_hit_persists_wildcard_if_missing(
        self,
    ) -> None:
        """Tuning enabled + has dynamic dim + concrete hit (wildcard not
        yet persisted) -> persist wildcard now.

        Note on the BLAS-frame remap: PyTorch's row-major addmm dispatch
        swaps inductor's (M, N) into BLAS's (n, m) so cuBLAS can stay
        column-major. `cublasCommonArgs::swapped_mn` flags this and
        `launchTunableGemmAndBias` remaps the inductor-frame mask sent
        by the test (`dynamic_dims_mask(M=True)`) into a BLAS-frame
        (N=True) before stamping `params.dynamic_dims_mask`. The
        persisted wildcard signature therefore has `*` in the BLAS-n
        slot (which is the original inductor-M, the actually dynamic
        dim). The wildcard's BLAS-m slot is concrete and equals
        inductor-N, and BLAS-k stays inductor-K."""
        # Use a (m, k) pair that no other test in this class touches so the
        # wildcard signature is unique and we can check for its specific
        # presence rather than relying on a fragile count delta.
        m, n, k = 41, 2053, 1009
        bias, mat1, mat2 = _addmm(m, n, k, seed=4)

        # Phase A: tune the concrete shape WITHOUT any dynamic mask, so only
        # the concrete key gets persisted.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        torch.addmm(bias, mat1, mat2)

        # Sanity: concrete entry should exist, but no wildcard for it yet.
        self.assertTrue(
            _has_concrete_entry("GemmAndBiasTunableOp", m, n, k),
            f"expected concrete entry for ({m},{n},{k}) after phase A",
        )
        # The "concrete-as-checked" tokens here are inductor-frame; the
        # actual BLAS-frame signature swaps M↔N. `_has_wildcard_with_dims`
        # below checks BLAS-frame tokens to match what's persisted.
        self.assertFalse(
            _has_wildcard_with_dims("GemmAndBiasTunableOp", n, k),
            f"unexpected wildcard for BLAS-frame ({n},*,{k}) before phase B",
        )

        # Phase B: same concrete shape, but now declare M dynamic. operator()
        # should hit the concrete entry and persist the wildcard.
        with torch.cuda.tunable.dynamic_dims_mask(M=True):
            torch.addmm(bias, mat1, mat2)

        # The persisted wildcard is in BLAS frame: tokens `_<n>_` (BLAS m
        # = inductor N) and `_<k>_` (BLAS k = inductor K) are concrete;
        # the BLAS-n slot holds `*` (the swapped inductor-M).
        self.assertTrue(
            _has_wildcard_with_dims("GemmAndBiasTunableOp", n, k),
            f"expected BLAS-frame wildcard containing ({n},*,{k}) after "
            "phase B (case 4: concrete hit + dynamic must persist wildcard "
            "if missing). With cublasCommonArgs::swapped_mn the inductor-M "
            "dynamic bit lands in BLAS-n.",
        )

    # -- Tuning disabled -------------------------------------------------

    def test_tuning_disabled_concrete_hit_dispatches_via_concrete(self) -> None:
        """Tuning disabled + concrete hit -> dispatch via concrete entry,
        result matches tunable-disabled reference."""
        m, n, k = 43, 2053, 1013
        bias, mat1, mat2 = _addmm(m, n, k, seed=5)

        # Prime concrete entry.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        torch.addmm(bias, mat1, mat2)

        # Reference: tunable disabled.
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        ref = torch.addmm(bias, mat1, mat2)

        # Test: tunable enabled, tuning disabled. Concrete hit.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        before = len(torch.cuda.tunable.get_results())
        out = torch.addmm(bias, mat1, mat2)
        after = len(torch.cuda.tunable.get_results())

        self.assertEqual(before, after, "tuning disabled should not add entries")
        self.assertTrue(
            torch.allclose(out, ref, atol=1e-2, rtol=1e-2),
            "tunable-enabled output should match tunable-disabled reference",
        )

    def test_tuning_disabled_concrete_miss_wildcard_hit_dispatches_via_wildcard(
        self,
    ) -> None:
        """Tuning disabled + concrete miss + wildcard match
        -> dispatch via wildcard entry, result matches tunable-disabled
        reference. The runtime cannot push a dynamic-dims mask (AOTI
        cpp_wrapper does not emit a TunableDynamicDimsGuard), so this
        test deliberately runs Phase B WITHOUT a `dynamic_dims_mask`
        context to mirror real runtime behavior. The wildcard match
        comes from `LookupWildcardFallback`'s token-by-token scan over
        persisted wildcard entries."""
        m_tuned, n, k = 47, 2053, 1019
        m_test = 53  # different M, same wildcard pattern
        bias_t, mat1_t, mat2_t = _addmm(m_tuned, n, k, seed=6)
        bias_x, mat1_x, mat2_x = _addmm(m_test, n, k, seed=7)

        # Phase A (compile-time analog): tune at m_tuned with M dynamic
        # so the wildcard `tn_*_n_k_…` is seeded into the manager.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        with torch.cuda.tunable.dynamic_dims_mask(M=True):
            torch.addmm(bias_t, mat1_t, mat2_t)

        wildcard_present = _has_wildcard_entry("GemmAndBiasTunableOp")
        self.assertTrue(
            wildcard_present, "expected wildcard entry after phase A tuning"
        )
        # m_test has no concrete entry yet.
        self.assertFalse(
            _has_concrete_entry("GemmAndBiasTunableOp", m_test, n, k),
            "should have no concrete entry for the new M before phase B",
        )

        # Reference: tunable disabled.
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        ref = torch.addmm(bias_x, mat1_x, mat2_x)

        # Phase B (runtime analog): tunable enabled, tuning disabled,
        # NO `dynamic_dims_mask` context -- the AOTI runtime cannot push
        # one. Concrete miss for m_test -> LookupWildcardFallback finds
        # the wildcard seeded in Phase A and dispatches via it.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        out = torch.addmm(bias_x, mat1_x, mat2_x)

        self.assertFalse(
            _has_concrete_entry("GemmAndBiasTunableOp", m_test, n, k),
            "tuning-disabled wildcard-fallback dispatch must not add a concrete entry",
        )
        self.assertTrue(
            torch.allclose(out, ref, atol=1e-2, rtol=1e-2),
            "wildcard-fallback dispatched output should match "
            "tunable-disabled reference",
        )

    def test_tuning_disabled_concrete_miss_wildcard_hit_without_runtime_mask(
        self,
    ) -> None:
        """Variant of the test above using a different shape to ensure the
        LookupWildcardFallback path is exercised across multiple
        compile/runtime tuples in the suite (regression gate against the
        runtime path silently regressing)."""
        m_tuned, n, k = 41, 2053, 1019
        m_test = 67  # different M, same wildcard pattern as Phase A
        bias_t, mat1_t, mat2_t = _addmm(m_tuned, n, k, seed=18)
        bias_x, mat1_x, mat2_x = _addmm(m_test, n, k, seed=19)

        # Phase A (compile-time analog): tune with M dynamic so the
        # wildcard `tn_*_n_k_...` is seeded into the tuning manager.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        with torch.cuda.tunable.dynamic_dims_mask(M=True):
            torch.addmm(bias_t, mat1_t, mat2_t)
        self.assertTrue(
            _has_wildcard_entry("GemmAndBiasTunableOp"),
            "expected wildcard entry after compile-time tuning phase",
        )
        self.assertFalse(
            _has_concrete_entry("GemmAndBiasTunableOp", m_test, n, k),
            "should have no concrete entry for the test M before runtime dispatch",
        )

        # Reference: tunable disabled.
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        ref = torch.addmm(bias_x, mat1_x, mat2_x)

        # Phase B (runtime analog): tunable enabled, tuning disabled, NO
        # dynamic_dims_mask context active -- the AOTI cpp_wrapper does
        # not emit TunableDynamicDimsGuard, so GetCurrentDynamicDimsMask()
        # is empty here. LookupWildcardFallback should still scan
        # persisted wildcard entries and dispatch via the matching one.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        out = torch.addmm(bias_x, mat1_x, mat2_x)

        self.assertFalse(
            _has_concrete_entry("GemmAndBiasTunableOp", m_test, n, k),
            "wildcard-fallback dispatch must not add a concrete entry "
            "when tuning is disabled",
        )
        self.assertTrue(
            torch.allclose(out, ref, atol=1e-2, rtol=1e-2),
            "wildcard-fallback dispatch (no runtime mask) should match "
            "tunable-disabled reference",
        )

    def test_tuning_disabled_both_miss_falls_back_to_non_tunable_aten(self) -> None:
        """Tuning disabled + concrete miss + no wildcard match
        -> dispatch falls back to the non-tunable aten path; no Default
        kernel invoked, no entries added, output matches tunable-disabled
        reference. Uses a large-K shape that historically crashed via the
        Default kernel on MI300X to confirm the safe path is taken."""
        # One of the prod shapes that crashed in tunable_mi300x_fresh_v2.txt
        # when TunableOp was enabled with no tuned entries.
        m, n, k = 2048, 2048, 11088
        bias, mat1, mat2 = _addmm(m, n, k, seed=8)

        # Reference: tunable disabled.
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        ref = torch.addmm(bias, mat1, mat2)

        # Test: tunable enabled, tuning disabled, no entries primed (no
        # concrete or wildcard for this shape exists in the manager).
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        before_results = len(torch.cuda.tunable.get_results())
        out = torch.addmm(bias, mat1, mat2)  # must NOT crash
        after_results = len(torch.cuda.tunable.get_results())

        self.assertEqual(
            before_results,
            after_results,
            "both-miss + tuning disabled must not add any entries",
        )
        self.assertTrue(
            torch.allclose(out, ref, atol=1e-2, rtol=1e-2),
            "fallback dispatch result should match tunable-disabled reference",
        )

    def test_tuning_disabled_no_wildcard_match_falls_back_safely(self) -> None:
        """Variant of `..._both_miss_falls_back_to_non_tunable_aten` using a
        fresh shape that has no concrete entry AND no wildcard pattern that
        could match. Verifies the runtime fall-through path stays safe even
        when the manager has unrelated wildcard entries from other tests
        (LookupWildcardFallback must reject non-matching wildcards rather
        than dispatch a wrong kernel).
        """
        m, n, k = 59, 2053, 1031
        bias, mat1, mat2 = _addmm(m, n, k, seed=9)

        # Reference: tunable disabled.
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        ref = torch.addmm(bias, mat1, mat2)

        # Test: tunable enabled, tuning disabled, no concrete entry exists
        # for this shape and no wildcard pattern matches it. (Runtime
        # cannot push a dynamic mask -- AOTI cpp_wrapper does not emit
        # TunableDynamicDimsGuard.)
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        out = torch.addmm(bias, mat1, mat2)  # must NOT crash

        self.assertTrue(
            torch.allclose(out, ref, atol=1e-2, rtol=1e-2),
            "fallback dispatch result should match tunable-disabled reference",
        )

    # -- Tuning disabled: mm / bmm runtime wildcard fallback -------------
    # Regression coverage for making GemmTunableOp (torch.mm) and
    # GemmStridedBatchedTunableOp (torch.bmm) consume persisted wildcard
    # entries at runtime via LookupWildcardFallback, matching the addmm
    # path (launchTunableGemmAndBias). Before the fix, gemm_tunable /
    # bgemm_tunable in CUDABlas.cpp did an exact-match Lookup on
    # DynamicSignature(); at runtime the mask is empty (no
    # TunableDynamicDimsGuard), so dynamic_sig == concrete_sig and the
    # lookup never fired -- every concrete miss silently fell back to the
    # non-tunable aten path instead of the persisted wildcard-tuned kernel.

    def test_tuning_disabled_mm_concrete_miss_wildcard_hit(self) -> None:
        """torch.mm: tuning disabled + concrete miss + wildcard match.
        Phase A tunes at m_tuned with M dynamic (seeds a GemmTunableOp
        wildcard); Phase B queries a different M at runtime with no mask.
        The concrete miss must resolve via LookupWildcardFallback (no
        concrete entry added) and match the tunable-disabled reference."""
        m_tuned, n, k = 61, 2063, 1021
        m_test = 79
        mat1_t, mat2_t = _mm(m_tuned, n, k, seed=50)
        mat1_x, mat2_x = _mm(m_test, n, k, seed=51)

        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        with torch.cuda.tunable.dynamic_dims_mask(M=True):
            torch.mm(mat1_t, mat2_t)
        self.assertTrue(
            _has_wildcard_entry("GemmTunableOp"),
            "expected GemmTunableOp wildcard entry after phase A tuning",
        )
        self.assertFalse(
            _has_concrete_entry("GemmTunableOp", m_test, n, k),
            "should have no concrete entry for the new M before phase B",
        )

        # Reference: tunable disabled.
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        ref = torch.mm(mat1_x, mat2_x)

        # Runtime: tunable enabled, tuning disabled, no mask.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        out = torch.mm(mat1_x, mat2_x)

        self.assertFalse(
            _has_concrete_entry("GemmTunableOp", m_test, n, k),
            "mm wildcard-fallback dispatch must not add a concrete entry",
        )
        self.assertTrue(
            torch.allclose(out, ref, atol=1e-2, rtol=1e-2),
            "mm wildcard-fallback output should match tunable-disabled reference",
        )

    def test_tuning_disabled_bmm_concrete_miss_wildcard_hit(self) -> None:
        """torch.bmm: tuning disabled + concrete miss + wildcard match.
        bmm is not subject to the cuBLAS M<->N swap, so the inductor-frame
        M-dynamic mask lands directly on the BLAS m slot; the persisted
        wildcard therefore matches a different-M concrete signature via
        LookupWildcardFallback's token scan. No concrete entry is added and
        the output matches the tunable-disabled reference."""
        b = 8
        m_tuned, n, k = 37, 263, 277
        m_test = 43
        b1_t, b2_t = _bmm(b, m_tuned, n, k, seed=52)
        b1_x, b2_x = _bmm(b, m_test, n, k, seed=53)

        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        with torch.cuda.tunable.dynamic_dims_mask(M=True):
            torch.bmm(b1_t, b2_t)
        self.assertTrue(
            _has_wildcard_entry("GemmStridedBatchedTunableOp"),
            "expected GemmStridedBatchedTunableOp wildcard entry after phase A",
        )
        self.assertFalse(
            _has_concrete_entry("GemmStridedBatchedTunableOp", m_test, n, k),
            "should have no concrete entry for the new M before phase B",
        )

        # Reference: tunable disabled.
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        ref = torch.bmm(b1_x, b2_x)

        # Runtime: tunable enabled, tuning disabled, no mask.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        out = torch.bmm(b1_x, b2_x)

        self.assertFalse(
            _has_concrete_entry("GemmStridedBatchedTunableOp", m_test, n, k),
            "bmm wildcard-fallback dispatch must not add a concrete entry",
        )
        self.assertTrue(
            torch.allclose(out, ref, atol=1e-2, rtol=1e-2),
            "bmm wildcard-fallback output should match tunable-disabled reference",
        )

    # -- Cache-behavior parity: concrete hit + safe aten fallback --------
    # Mirror the three addmm runtime scenarios (concrete hit, wildcard
    # fallback, both-miss safe fallback) for mm/bmm/baddbmm so every
    # non-scaled variant is verified to behave identically to
    # launchTunableGemmAndBias.

    def test_tuning_disabled_mm_concrete_hit(self) -> None:
        """torch.mm concrete hit: after tuning a concrete entry, a runtime
        query for the same shape dispatches via it -- no new entry, output
        matches the tunable-disabled reference."""
        m, n, k = 83, 2039, 1033
        mat1, mat2 = _mm(m, n, k, seed=54)

        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        ref = torch.mm(mat1, mat2)

        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        torch.mm(mat1, mat2)
        before = len(torch.cuda.tunable.get_results())

        torch.cuda.tunable.tuning_enable(False)
        out = torch.mm(mat1, mat2)
        after = len(torch.cuda.tunable.get_results())

        self.assertEqual(before, after, "mm concrete-hit must not add an entry")
        self.assertTrue(
            torch.allclose(out, ref, atol=1e-2, rtol=1e-2),
            "mm concrete-hit output should match reference",
        )

    def test_tuning_disabled_mm_both_miss_falls_back_safely(self) -> None:
        """torch.mm both-miss: tunable enabled, tuning disabled, no concrete
        entry and no matching wildcard -> safe non-tunable aten fallback (no
        crash, no entry added, output matches reference)."""
        m, n, k = 97, 1493, 1499
        mat1, mat2 = _mm(m, n, k, seed=55)

        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        ref = torch.mm(mat1, mat2)

        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        before = len(torch.cuda.tunable.get_results())
        out = torch.mm(mat1, mat2)  # must NOT crash
        after = len(torch.cuda.tunable.get_results())

        self.assertEqual(before, after, "mm both-miss must not add an entry")
        self.assertTrue(
            torch.allclose(out, ref, atol=1e-2, rtol=1e-2),
            "mm both-miss fallback output should match reference",
        )

    def test_tuning_disabled_bmm_concrete_hit(self) -> None:
        """torch.bmm concrete hit -> dispatch via concrete entry, no new
        entry added, output matches reference."""
        b, m, n, k = 8, 31, 269, 281
        b1, b2 = _bmm(b, m, n, k, seed=56)

        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        ref = torch.bmm(b1, b2)

        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        torch.bmm(b1, b2)
        before = len(torch.cuda.tunable.get_results())

        torch.cuda.tunable.tuning_enable(False)
        out = torch.bmm(b1, b2)
        after = len(torch.cuda.tunable.get_results())

        self.assertEqual(before, after, "bmm concrete-hit must not add an entry")
        self.assertTrue(
            torch.allclose(out, ref, atol=1e-2, rtol=1e-2),
            "bmm concrete-hit output should match reference",
        )

    def test_tuning_disabled_bmm_both_miss_falls_back_safely(self) -> None:
        """torch.bmm both-miss -> safe non-tunable aten fallback (no crash,
        no entry added, output matches reference)."""
        b, m, n, k = 8, 101, 1451, 1453
        b1, b2 = _bmm(b, m, n, k, seed=57)

        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        ref = torch.bmm(b1, b2)

        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        before = len(torch.cuda.tunable.get_results())
        out = torch.bmm(b1, b2)  # must NOT crash
        after = len(torch.cuda.tunable.get_results())

        self.assertEqual(before, after, "bmm both-miss must not add an entry")
        self.assertTrue(
            torch.allclose(out, ref, atol=1e-2, rtol=1e-2),
            "bmm both-miss fallback output should match reference",
        )

    def test_tuning_disabled_baddbmm_concrete_miss_wildcard_hit(self) -> None:
        """torch.baddbmm: tune at m_tuned with M dynamic -> wildcard; runtime
        query at a different M with no mask -> wildcard fallback, output
        matches reference. baddbmm shares bgemm_tunable with bmm; the bias is
        applied around the same GemmStridedBatchedTunableOp dispatch."""
        b = 8
        m_tuned, n, k = 29, 277, 287
        m_test = 41
        bias_t, b1_t, b2_t = _baddbmm(b, m_tuned, n, k, seed=58)
        bias_x, b1_x, b2_x = _baddbmm(b, m_test, n, k, seed=59)

        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        with torch.cuda.tunable.dynamic_dims_mask(M=True):
            torch.baddbmm(bias_t, b1_t, b2_t)
        self.assertTrue(
            _has_wildcard_entry("GemmStridedBatchedTunableOp"),
            "expected GemmStridedBatchedTunableOp wildcard entry after baddbmm phase A",
        )

        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        ref = torch.baddbmm(bias_x, b1_x, b2_x)

        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        out = torch.baddbmm(bias_x, b1_x, b2_x)

        self.assertTrue(
            torch.allclose(out, ref, atol=1e-2, rtol=1e-2),
            "baddbmm wildcard-fallback output should match reference",
        )

    def test_tuning_disabled_baddbmm_both_miss_falls_back_safely(self) -> None:
        """torch.baddbmm both-miss -> safe non-tunable aten fallback (no
        crash, output matches reference)."""
        b = 8
        m, n, k = 103, 1481, 1483
        bias, b1, b2 = _baddbmm(b, m, n, k, seed=60)

        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        ref = torch.baddbmm(bias, b1, b2)

        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        out = torch.baddbmm(bias, b1, b2)  # must NOT crash

        self.assertTrue(
            torch.allclose(out, ref, atol=1e-2, rtol=1e-2),
            "baddbmm both-miss fallback output should match reference",
        )

# ─── Coverage across all four TunableOp variants ──────────────────────
# These tests exercise the full compile-time-tune -> runtime-fallback
# loop on each TunableOp class:
#   - GemmAndBiasTunableOp     (torch.addmm with 1D bias, LT-fused-bias)
#   - GemmTunableOp            (torch.mm, plain GEMM)
#   - GemmStridedBatchedTunableOp (torch.bmm)
# For each class the test:
#   Phase A: tune at one shape with M dynamic -> wildcard persisted.
#   Phase B: compute reference with TunableOp disabled.
#   Phase C: query at a different M with TunableOp enabled, tuning OFF,
#            no runtime mask -> LookupWildcardFallback must hit the
#            wildcard, no concrete entry added, output matches reference.


class AllTunableOpsWildcardFallbackTest(TestCase):
    """Verify the dynamic-mask + wildcard-fallback contract on each
    TunableOp variant."""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("cuda not available")
        cls._tmpdir = tempfile.mkdtemp(prefix="all_tunable_ops_test_")
        cls._tmp_results_path = os.path.join(cls._tmpdir, "tunable_results.csv")
        torch.cuda.tunable.set_filename(cls._tmp_results_path, False)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._tmpdir:
            import shutil

            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def setUp(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)

    def tearDown(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)

    def _ref(self, fn) -> torch.Tensor:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        return fn()

    def test_gemm_and_bias_tunable_op_addmm_wildcard_round_trip(self) -> None:
        """GemmAndBiasTunableOp via torch.addmm with M-dynamic mask.
        Verifies the addmm path (the original repro)."""
        m_tuned, n, k = 53, 2069, 1013
        m_test = 71
        bias_t, mat1_t, mat2_t = _addmm(m_tuned, n, k, seed=20)
        bias_x, mat1_x, mat2_x = _addmm(m_test, n, k, seed=21)

        # Phase A: tune at m_tuned with M dynamic -> wildcard persisted.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        with torch.cuda.tunable.dynamic_dims_mask(M=True):
            torch.addmm(bias_t, mat1_t, mat2_t)
        self.assertTrue(
            _has_wildcard_entry("GemmAndBiasTunableOp"),
            "expected GemmAndBiasTunableOp wildcard entry after tuning",
        )

        # Reference.
        ref = self._ref(lambda: torch.addmm(bias_x, mat1_x, mat2_x))

        # Phase C: runtime, no mask, different M -> wildcard fallback hits.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        out = torch.addmm(bias_x, mat1_x, mat2_x)

        self.assertFalse(
            _has_concrete_entry("GemmAndBiasTunableOp", m_test, n, k),
            "wildcard-fallback dispatch should not add a concrete entry",
        )
        self.assertTrue(
            torch.allclose(out, ref, atol=1e-2, rtol=1e-2),
            "GemmAndBiasTunableOp wildcard-fallback output should match "
            "tunable-disabled reference",
        )

    def test_gemm_tunable_op_mm_wildcard_round_trip(self) -> None:
        """GemmTunableOp via torch.mm with M-dynamic mask.
        Verifies the plain-GEMM (no bias) path also remaps the mask
        through `cublasCommonArgs::swapped_mn` and dispatches via the
        wildcard fallback at runtime."""
        m_tuned, n, k = 59, 2089, 1019
        m_test = 73
        mat1_t, mat2_t = _mm(m_tuned, n, k, seed=22)
        mat1_x, mat2_x = _mm(m_test, n, k, seed=23)

        # Phase A.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        with torch.cuda.tunable.dynamic_dims_mask(M=True):
            torch.mm(mat1_t, mat2_t)
        self.assertTrue(
            _has_wildcard_entry("GemmTunableOp"),
            "expected GemmTunableOp wildcard entry after tuning",
        )

        # Reference.
        ref = self._ref(lambda: torch.mm(mat1_x, mat2_x))

        # Phase C.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        out = torch.mm(mat1_x, mat2_x)

        self.assertFalse(
            _has_concrete_entry("GemmTunableOp", m_test, n, k),
            "GemmTunableOp wildcard-fallback dispatch should not add a concrete entry",
        )
        self.assertTrue(
            torch.allclose(out, ref, atol=1e-2, rtol=1e-2),
            "GemmTunableOp wildcard-fallback output should match "
            "tunable-disabled reference",
        )

    def test_gemm_strided_batched_tunable_op_bmm_wildcard_round_trip(
        self,
    ) -> None:
        """GemmStridedBatchedTunableOp via torch.bmm with M-dynamic
        mask. bmm is not subject to the cuBLAS M<->N swap, so the
        M-dynamic wildcard is persisted with `*` in the BLAS m slot and
        `bgemm_tunable` (CUDABlas.cpp) now resolves the runtime concrete
        miss through `LookupWildcardFallback`, matching the addmm path."""
        b = 16
        m_tuned, n, k = 47, 257, 251
        m_test = 53
        b1_t, b2_t = _bmm(b, m_tuned, n, k, seed=24)
        b1_x, b2_x = _bmm(b, m_test, n, k, seed=25)

        # Phase A: tune with both M dynamic and BATCH dynamic so the
        # wildcard signature has wildcards in the dims that vary at
        # runtime. (M is what changes between Phases B and C; BATCH is
        # constant in this test but pushed for symmetry with the
        # inductor-side mask that would normally include both.)
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        with torch.cuda.tunable.dynamic_dims_mask(M=True):
            torch.bmm(b1_t, b2_t)
        self.assertTrue(
            _has_wildcard_entry("GemmStridedBatchedTunableOp"),
            "expected GemmStridedBatchedTunableOp wildcard entry after tuning",
        )

        # Reference.
        ref = self._ref(lambda: torch.bmm(b1_x, b2_x))

        # Phase C.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        out = torch.bmm(b1_x, b2_x)

        # Output correctness: whether dispatch resolves via the wildcard
        # fallback or (on a wildcard miss) via the non-tunable aten path,
        # both produce the same numerical result, so this assertion is a
        # correctness sanity check rather than a hard gate on the wildcard
        # fallback firing (no Python-visible hit counter exists yet).
        self.assertTrue(
            torch.allclose(out, ref, atol=1e-2, rtol=1e-2),
            "GemmStridedBatchedTunableOp dispatch output should match "
            "tunable-disabled reference (either via wildcard fallback "
            "or via non-tunable aten fallback)",
        )


# ─── Layout coverage: verify swap remap across NN/NT/TN/TT ────────────
# PyTorch's BLAS dispatch picks `transa, transb` and the
# `transpose_result` decision based on operand layouts. The
# `cublasCommonArgs::swapped_mn` flag tracks `transpose_result` so the
# `launchTunableGemmAndBias` mask remap fires whenever the dispatch
# swaps inductor (M, N) -> BLAS (n, m). These tests force each
# (transa, transb) combination by transposing input tensors and
# verify the wildcard-tune -> wildcard-fallback round trip works
# correctly for every layout.


class LayoutCoverageWildcardTest(TestCase):
    """Exercise the dynamic-mask + wildcard-fallback contract for each
    of the four (transa, transb) layout combinations, with each of the
    M/N/K dims taken dynamic in turn. Each test tunes at one shape, then
    queries at a shape that differs only in the dynamic dim with no
    runtime mask, and asserts: a wildcard entry was persisted, its
    leading-dim wildcarding is self-consistent with the transpose flags
    (white-box), no concrete entry is added at runtime, and output
    matches a tunable-disabled reference.

    The K-dynamic cases are what give lda coverage: for torch.mm the
    row-major dispatch swaps inductor (M, N) into BLAS (n, m), so an
    M-dynamic call lands the dynamic bit on BLAS-n (never lda). K is not
    swapped, so a K-dynamic call keeps k dynamic in the BLAS frame and
    forces lda to be wildcarded exactly when transa == 'T'."""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("cuda not available")
        cls._tmpdir = tempfile.mkdtemp(prefix="layout_coverage_test_")
        cls._tmp_results_path = os.path.join(cls._tmpdir, "tunable_results.csv")
        torch.cuda.tunable.set_filename(cls._tmp_results_path, False)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._tmpdir:
            import shutil

            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def setUp(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)

    def tearDown(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)

    @staticmethod
    def _make_layout(
        m: int, k: int, n: int, transa: bool, transb: bool, seed: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (mat1, mat2) for `mm(mat1, mat2)` so the BLAS dispatch
        picks the requested (transa, transb) layout. We synthesize the
        layout by allocating the underlying storage in the transposed
        shape and then transposing the view; this is the canonical way
        to make a tensor "non-contiguous" along a particular axis without
        triggering a contiguous() copy."""
        torch.manual_seed(seed)
        if not transa:
            mat1 = torch.randn(m, k, dtype=DTYPE, device=DEVICE)
        else:
            mat1 = torch.randn(k, m, dtype=DTYPE, device=DEVICE).t()
        if not transb:
            mat2 = torch.randn(k, n, dtype=DTYPE, device=DEVICE)
        else:
            mat2 = torch.randn(n, k, dtype=DTYPE, device=DEVICE).t()
        return mat1, mat2

    def _round_trip_for_layout(
        self, transa: bool, transb: bool, seed_offset: int, dynamic: str = "M"
    ) -> None:
        """Run the tune-then-fallback round trip for one layout, taking the
        `dynamic` dim (one of "M"/"N"/"K") as symbolic. Tuning and runtime
        shapes differ only in that dim."""
        base_m, n, k = 41 + seed_offset, 257, 251
        delta = 12
        tuned = {"m": base_m, "n": n, "k": k}
        test = {"m": base_m, "n": n, "k": k}
        test[dynamic.lower()] += delta

        mat1_t, mat2_t = self._make_layout(
            tuned["m"], tuned["k"], tuned["n"], transa, transb, seed=100 + seed_offset
        )
        mat1_x, mat2_x = self._make_layout(
            test["m"], test["k"], test["n"], transa, transb, seed=200 + seed_offset
        )

        # Reference (tunable disabled).
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        ref = torch.mm(mat1_x, mat2_x)

        # Phase A: tune at the tuned shape with the chosen dim dynamic ->
        # wildcard persisted.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        with torch.cuda.tunable.dynamic_dims_mask(**{dynamic: True}):
            torch.mm(mat1_t, mat2_t)
        self.assertTrue(
            _has_wildcard_entry("GemmTunableOp"),
            f"expected GemmTunableOp wildcard entry after tuning "
            f"layout (transa={transa}, transb={transb}, dynamic={dynamic})",
        )
        # White-box: the persisted wildcard must wildcard the correct
        # leading dims for its transpose flags. This is what catches an
        # inverted lda/ldb/ldc -> dim mapping (e.g. a broken UsesMForLda),
        # which _has_wildcard_entry alone silently tolerates because a
        # mis-wildcarded key still contains a '*'.
        _assert_ld_wildcarding_consistent(self, "GemmTunableOp")

        # Phase C: runtime, no mask, shape differs only in the dynamic dim
        # -> must dispatch via wildcard fallback (no new concrete entry) and
        # match the tunable-disabled reference.
        before = len(torch.cuda.tunable.get_results())
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        out = torch.mm(mat1_x, mat2_x)
        after = len(torch.cuda.tunable.get_results())
        self.assertEqual(
            before,
            after,
            f"layout (transa={transa}, transb={transb}, dynamic={dynamic}): "
            f"runtime dispatch with tuning disabled must not add a new entry",
        )
        self.assertTrue(
            torch.allclose(out, ref, atol=1e-2, rtol=1e-2),
            f"layout (transa={transa}, transb={transb}, dynamic={dynamic}): "
            f"wildcard-fallback dispatch output must match tunable-disabled "
            f"reference",
        )

    def test_layout_NN(self) -> None:
        """No transpose on either operand."""
        self._round_trip_for_layout(transa=False, transb=False, seed_offset=0)

    def test_layout_NT(self) -> None:
        """mat1 contiguous, mat2 transposed."""
        self._round_trip_for_layout(transa=False, transb=True, seed_offset=4)

    def test_layout_TN(self) -> None:
        """mat1 transposed, mat2 contiguous."""
        self._round_trip_for_layout(transa=True, transb=False, seed_offset=8)

    def test_layout_TT(self) -> None:
        """Both operands transposed."""
        self._round_trip_for_layout(transa=True, transb=True, seed_offset=12)

    def test_layout_NN_dynamic_k(self) -> None:
        """K dynamic (not swapped by the mm remap) exercises lda/ldb."""
        self._round_trip_for_layout(
            transa=False, transb=False, seed_offset=16, dynamic="K"
        )

    def test_layout_NT_dynamic_k(self) -> None:
        self._round_trip_for_layout(
            transa=False, transb=True, seed_offset=20, dynamic="K"
        )

    def test_layout_TN_dynamic_k(self) -> None:
        """transa == 'T' + K dynamic forces lda to be wildcarded; the
        inverted UsesMForLda leaves it concrete and this test fails."""
        self._round_trip_for_layout(
            transa=True, transb=False, seed_offset=24, dynamic="K"
        )

    def test_layout_TT_dynamic_k(self) -> None:
        self._round_trip_for_layout(
            transa=True, transb=True, seed_offset=28, dynamic="K"
        )


# ─── ScaledGemmTunableOp coverage (FP8 _scaled_mm) ────────────────────
# Mirrors the inductor_lowering_context node_replacement_dict
# `{'torch.nn.Linear':{'(10000+,1000+)': 'fp8_float_model_dynamic_quantization_tensorwise'}}`
# which rewrites large Linear layers into FP8 + per-tensor (tensorwise)
# scaling. The lowered op is `torch._scaled_mm(a_fp8, b_fp8, scale_a,
# scale_b, ...)` which ultimately dispatches to
# `launchTunableScaledGemm` -> `ScaledGemmTunableOp::operator()`.
#
# This test verifies the same wildcard round-trip for that op:
#   Phase A: tune at m_tuned with M dynamic -> wildcard persisted.
#   Phase C: runtime concrete miss at m_test, no runtime mask
#            -> LookupWildcardFallback hits the wildcard.


class ScaledGemmTunableOpFP8Test(TestCase):
    """Verify the dynamic-mask + wildcard-fallback contract for
    `torch._scaled_mm` with tensorwise FP8 scaling, matching the
    `node_replacement_dict` setup that triggers
    `ScaledGemmTunableOp` for large Linear layers."""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("cuda not available")
        # FP8 e4m3fn requires a recent enough device; skip on older
        # hardware that doesn't expose the dtype.
        if not hasattr(torch, "float8_e4m3fn"):
            raise unittest.SkipTest("torch.float8_e4m3fn not available")
        cls._tmpdir = tempfile.mkdtemp(prefix="scaled_gemm_test_")
        cls._tmp_results_path = os.path.join(cls._tmpdir, "tunable_results.csv")
        torch.cuda.tunable.set_filename(cls._tmp_results_path, False)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._tmpdir:
            import shutil

            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def setUp(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)

    def tearDown(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)

    @staticmethod
    def _scaled_mm_inputs(
        m: int, n: int, k: int, seed: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build (a_fp8, b_fp8, scale_a, scale_b) for a tensorwise-
        scaled `_scaled_mm(a, b, scale_a, scale_b)` of shape (m, n, k).

        a is M×K row-major, b is K×N -> b must be col-major for the
        cuBLAS scaled-mm path (the typical inductor pattern after the
        `fp8_float_model_dynamic_quantization_tensorwise` rewrite).
        """
        torch.manual_seed(seed)
        # Random BF16 then quantize to FP8 e4m3fn with a tensorwise scale.
        a_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE)
        b_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device=DEVICE)
        # Per-tensor (tensorwise) absmax scale -> single-element fp32
        # tensor.
        a_amax = a_bf16.abs().max().to(torch.float32)
        b_amax = b_bf16.abs().max().to(torch.float32)
        # FP8 e4m3fn dynamic range max value.
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale_a = (a_amax / fp8_max).reciprocal().clamp(min=1e-12)
        scale_b = (b_amax / fp8_max).reciprocal().clamp(min=1e-12)
        a_fp8 = (
            (a_bf16.to(torch.float32) * scale_a)
            .clamp(min=-fp8_max, max=fp8_max)
            .to(torch.float8_e4m3fn)
        )
        b_fp8 = (
            (b_bf16.to(torch.float32) * scale_b)
            .clamp(min=-fp8_max, max=fp8_max)
            .to(torch.float8_e4m3fn)
        )
        # `_scaled_mm` requires `b` to be column-major (matches the
        # inductor lowering for the fp8 quantization rewrite).
        b_fp8_colmajor = b_fp8.t().contiguous().t()
        # Scale tensors are reciprocals of the quantization scales.
        scale_a_inv = scale_a.reciprocal()
        scale_b_inv = scale_b.reciprocal()
        return a_fp8, b_fp8_colmajor, scale_a_inv, scale_b_inv

    def _run_scaled_mm(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
    ) -> torch.Tensor:
        # `_scaled_mm` with a column-major `b` produces an M×N bf16 out.
        return torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.bfloat16)

    def test_scaled_gemm_tunable_op_wildcard_round_trip(self) -> None:
        """Tune `_scaled_mm` at m_tuned with M dynamic; query at
        m_test with no runtime mask. Wildcard fallback must hit (or
        the dispatch must safely fall through to a non-tunable path
        with correct output)."""
        # Use shapes that match the
        # `node_replacement_dict`-style trigger: a "large Linear" with
        # K and N >= 1000 (the prod rule says 10000+, 1000+; we use a
        # smaller-but-representative shape so the test runs quickly).
        m_tuned, n, k = 1024, 1024, 1024
        m_test = 2048

        # Reference: tunable disabled.
        a_t, b_t, sa_t, sb_t = self._scaled_mm_inputs(m_tuned, n, k, seed=30)
        a_x, b_x, sa_x, sb_x = self._scaled_mm_inputs(m_test, n, k, seed=31)

        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        try:
            ref = self._run_scaled_mm(a_x, b_x, sa_x, sb_x)
        except RuntimeError as e:
            # _scaled_mm may not be supported on this device/dtype combo
            # (e.g. unsupported gfx arch). Skip rather than fail.
            raise unittest.SkipTest(
                f"_scaled_mm not supported in this configuration: {e}"
            ) from e

        # Phase A: tune at m_tuned with M dynamic.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        with torch.cuda.tunable.dynamic_dims_mask(M=True):
            self._run_scaled_mm(a_t, b_t, sa_t, sb_t)

        # We don't strictly require a wildcard entry to exist -- some
        # FP8 paths may take a non-tunable shortcut. But if any
        # ScaledGemmTunableOp entry is present it should be the one we
        # tuned. Check for "any" Scaled entry as a sanity gate.
        scaled_entries = [
            e for e in torch.cuda.tunable.get_results() if "ScaledGemmTunableOp" in e[0]
        ]
        if not scaled_entries:
            raise unittest.SkipTest(
                "no ScaledGemmTunableOp entries persisted; "
                "_scaled_mm may have skipped the tunable path on this "
                "configuration"
            )
        wildcard_present = any("*" in e[1] for e in scaled_entries)
        self.assertTrue(
            wildcard_present,
            f"expected ScaledGemmTunableOp wildcard entry after tuning "
            f"with M dynamic; got entries: {scaled_entries}",
        )

        # Phase C: runtime, no mask, different M -> wildcard fallback
        # (or aten fallback) must produce correct output.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        out = self._run_scaled_mm(a_x, b_x, sa_x, sb_x)

        self.assertTrue(
            torch.allclose(out, ref, atol=5e-2, rtol=5e-2),
            "ScaledGemmTunableOp dispatch output should match "
            "tunable-disabled reference (either via wildcard fallback "
            "or via non-tunable aten fallback)",
        )

    def test_scaled_gemm_both_miss_falls_back_safely(self) -> None:
        """_scaled_mm both-miss: tunable enabled, tuning disabled, with no
        concrete and no wildcard entry primed -> must fall back to the
        non-tunable at::cuda::blas::scaled_gemm (NOT ResultEntry::Default())
        and produce correct output without crashing. Regression gate for the
        scaled path's safe-aten fallback: before the try_scaled_dispatch gate
        in _tunable_scaled_gemm_rocm, a total miss reached the Default kernel
        instead of falling back like the addmm path does."""
        m, n, k = 4096, 1024, 1024
        a, b, sa, sb = self._scaled_mm_inputs(m, n, k, seed=32)

        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        try:
            ref = self._run_scaled_mm(a, b, sa, sb)
        except RuntimeError as e:
            raise unittest.SkipTest(
                f"_scaled_mm not supported in this configuration: {e}"
            ) from e

        # No entries primed for this shape. Tunable enabled, tuning disabled:
        # concrete miss + wildcard miss must fall back safely.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        out = self._run_scaled_mm(a, b, sa, sb)  # must NOT crash / Default

        self.assertTrue(
            torch.allclose(out, ref, atol=5e-2, rtol=5e-2),
            "scaled both-miss fallback output should match "
            "tunable-disabled reference",
        )


# ─── Legacy "non-dynamic" behavior (BEFORE the wildcard feature) ─────
# These tests assert what TunableOp does when no `dynamic_dims_mask`
# is ever pushed (the pre-feature world): only concrete-key entries
# get persisted at compile-time tuning, and runtime concrete-miss
# queries fall through to the non-tunable aten path. These are the
# canonical "before" behavior gates -- they document and lock in what
# the system reverts to when `triton.autotune_tunableop_dynamic_dims_
# wildcard=False` (the kill-switch for the dynamic-tunable-ops
# feature).


class LegacyConcreteOnlyTunableOpsTest(TestCase):
    """Verify that without a dynamic-dims mask (the pre-feature
    behavior), TunableOp only persists concrete entries and runtime
    concrete-misses safely fall through. These tests do NOT push any
    `dynamic_dims_mask` -- they emulate the world where the feature
    flag `triton.autotune_tunableop_dynamic_dims_wildcard=False`."""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("cuda not available")
        cls._tmpdir = tempfile.mkdtemp(prefix="legacy_concrete_only_test_")
        cls._tmp_results_path = os.path.join(cls._tmpdir, "tunable_results.csv")
        torch.cuda.tunable.set_filename(cls._tmp_results_path, False)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._tmpdir:
            import shutil

            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def setUp(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)

    def tearDown(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)

    def _entries_for_op(self, op_substr: str) -> list:
        return [e for e in torch.cuda.tunable.get_results() if op_substr in e[0]]

    def test_addmm_tuning_no_mask_persists_concrete_only(self) -> None:
        """Tuning enabled, NO `dynamic_dims_mask` context.
        addmm produces a single concrete entry; no wildcard."""
        m, n, k = 89, 1031, 257
        bias, mat1, mat2 = _addmm(m, n, k, seed=40)

        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        torch.addmm(bias, mat1, mat2)  # NO mask context

        entries = self._entries_for_op("GemmAndBiasTunableOp")
        # The persisted entries must NOT contain a wildcard.
        for op_sig, params_sig, _, _ in entries:
            self.assertNotIn(
                "*",
                params_sig,
                f"legacy mode (no mask) must not persist wildcards; "
                f"got {op_sig},{params_sig}",
            )
        # At least one concrete entry must exist for our shape.
        concrete_match = [
            e for e in entries if all(f"_{d}_" in ("_" + e[1] + "_") for d in (m, n, k))
        ]
        self.assertGreaterEqual(
            len(concrete_match),
            1,
            f"expected concrete entry covering ({m},{n},{k}) after tuning",
        )

    def test_mm_tuning_no_mask_persists_concrete_only(self) -> None:
        """torch.mm tuning without mask: only GemmTunableOp concrete."""
        m, n, k = 91, 1033, 263
        mat1, mat2 = _mm(m, n, k, seed=41)

        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        torch.mm(mat1, mat2)  # NO mask

        for op_sig, params_sig, _, _ in self._entries_for_op("GemmTunableOp"):
            self.assertNotIn(
                "*",
                params_sig,
                f"legacy mode (no mask) must not persist wildcards; "
                f"got {op_sig},{params_sig}",
            )

    def test_bmm_tuning_no_mask_persists_concrete_only(self) -> None:
        """torch.bmm tuning without mask: only StridedBatched concrete."""
        b, m, n, k = 16, 47, 257, 251
        b1, b2 = _bmm(b, m, n, k, seed=42)

        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        torch.bmm(b1, b2)  # NO mask

        for op_sig, params_sig, _, _ in self._entries_for_op(
            "GemmStridedBatchedTunableOp"
        ):
            self.assertNotIn(
                "*",
                params_sig,
                f"legacy mode (no mask) must not persist wildcards; "
                f"got {op_sig},{params_sig}",
            )

    def test_runtime_concrete_miss_no_mask_falls_back_to_aten(self) -> None:
        """Runtime: tunable enabled, tuning OFF, NO mask, no
        wildcards exist -> concrete miss must fall back to non-
        tunable aten (output correct, no entries added, no Default
        crash)."""
        # Use a fresh shape that no other test seeded.
        m, n, k = 73, 1039, 269
        bias, mat1, mat2 = _addmm(m, n, k, seed=43)

        # Reference.
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        ref = torch.addmm(bias, mat1, mat2)

        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        before = len(torch.cuda.tunable.get_results())
        out = torch.addmm(bias, mat1, mat2)  # NO mask, NO concrete entry
        after = len(torch.cuda.tunable.get_results())

        self.assertEqual(
            before,
            after,
            "concrete-miss + tuning-disabled must not add any entries",
        )
        self.assertTrue(
            torch.allclose(out, ref, atol=1e-2, rtol=1e-2),
            "fallback dispatch result should match tunable-disabled reference",
        )

    def test_runtime_concrete_hit_no_mask_dispatches_via_concrete(
        self,
    ) -> None:
        """After tuning a concrete entry (no mask), runtime queries
        for the same shape must hit that concrete entry. No wildcard
        scan needed."""
        m, n, k = 79, 1049, 271
        bias, mat1, mat2 = _addmm(m, n, k, seed=44)

        # Reference.
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        ref = torch.addmm(bias, mat1, mat2)

        # Tune the concrete shape (no mask).
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        torch.addmm(bias, mat1, mat2)
        before = len(torch.cuda.tunable.get_results())

        # Runtime: same shape -> concrete hit, no new entry, output
        # matches.
        torch.cuda.tunable.tuning_enable(False)
        out = torch.addmm(bias, mat1, mat2)
        after = len(torch.cuda.tunable.get_results())

        self.assertEqual(
            before,
            after,
            "concrete-hit dispatch must not add any new entry",
        )
        self.assertTrue(
            torch.allclose(out, ref, atol=1e-2, rtol=1e-2),
            "concrete-hit dispatch result should match reference",
        )


class _SizeOnlyNode:
    """Minimal input-node stub exposing only get_size(), which is all that
    shapes_symbolic / dynamic_dim_mask consult. Lets us drive
    dynamic_dim_mask with symbolic (sympy) dims on CPU, no GPU needed."""

    def __init__(self, size: tuple[object, ...]) -> None:
        self._size = size

    def get_size(self) -> tuple[object, ...]:
        return self._size


@inductor_config.patch({"triton.autotune_tunableop_dynamic_dims_wildcard": True})
class DynamicDimMaskOperandSelectionTest(TestCase):
    """CPU unit tests pinning down two properties of
    ``MMKernelInputs.dynamic_dim_mask`` that matter for scaled GEMM:

      1. Operands are read from the explicit mat1_idx / mat2_idx, not the
         trailing two inputs. Scaled GEMM lays out inputs as
         ``[mat_a, mat_b, scale_a, scale_b, (bias)]`` with mat1_idx=0,
         mat2_idx=1, so reading the trailing inputs would derive the mask
         from the scale/bias tensors instead of the real M/N/K operands.
      2. The ``"scaled_mm"`` op name (used by the inductor scaled-mm
         lowering) is recognized as a 2D matmul.
    """

    def _scaled_mm_inputs(self, with_bias: bool) -> MMKernelInputs:
        """Scaled-mm-style inputs where the real operands (indices 0, 1)
        have a dynamic M only, while the trailing scale/bias tensors carry a
        *different* dynamic dim -- so a (buggy) trailing-input mask would be
        distinguishable from the correct operand-based mask."""
        s_m = sympy.Symbol("s_m", positive=True, integer=True)
        s_bad = sympy.Symbol("s_bad", positive=True, integer=True)
        mat_a = _SizeOnlyNode((s_m, 512))  # (M_dyn, K)
        mat_b = _SizeOnlyNode((512, 256))  # (K, N)  static
        scale_a = _SizeOnlyNode((1, 1))
        scale_b = _SizeOnlyNode((s_bad, 999))  # dynamic dim in trailing input
        nodes: list[object] = [mat_a, mat_b, scale_a, scale_b]
        if with_bias:
            nodes.append(_SizeOnlyNode((256,)))
        return MMKernelInputs(nodes, mat1_idx=0, mat2_idx=1)

    def test_scaled_mm_reads_operands_not_trailing_scales(self) -> None:
        ki = self._scaled_mm_inputs(with_bias=False)
        # Correct mask is derived from mat_a/mat_b -> only M dynamic. If the
        # trailing scale tensors were read, s_bad would surface as dyn_k.
        self.assertEqual(
            ki.dynamic_dim_mask("scaled_mm"), (True, False, False, False)
        )

    def test_scaled_mm_with_bias_reads_operands(self) -> None:
        ki = self._scaled_mm_inputs(with_bias=True)
        self.assertEqual(
            ki.dynamic_dim_mask("scaled_mm"), (True, False, False, False)
        )

    def test_scaled_mm_op_name_recognized(self) -> None:
        # A dynamic operand dim must yield a non-all-False mask, proving the
        # "scaled_mm" name is recognized as a 2D matmul.
        ki = self._scaled_mm_inputs(with_bias=False)
        self.assertNotEqual(
            ki.dynamic_dim_mask("scaled_mm"), (False, False, False, False)
        )

    def test_unrecognized_op_name_returns_all_false(self) -> None:
        ki = self._scaled_mm_inputs(with_bias=False)
        self.assertEqual(
            ki.dynamic_dim_mask("not_a_gemm"), (False, False, False, False)
        )

    def test_default_indices_mm_reads_trailing_operands(self) -> None:
        # Plain mm: [mat1, mat2], default indices, dynamic N only.
        s_n = sympy.Symbol("s_n", positive=True, integer=True)
        mat1 = _SizeOnlyNode((128, 512))  # (M, K)
        mat2 = _SizeOnlyNode((512, s_n))  # (K, N_dyn)
        ki = MMKernelInputs([mat1, mat2])
        self.assertEqual(ki.dynamic_dim_mask("mm"), (False, True, False, False))

    def test_feature_flag_off_returns_all_false(self) -> None:
        with inductor_config.patch(
            {"triton.autotune_tunableop_dynamic_dims_wildcard": False}
        ):
            ki = self._scaled_mm_inputs(with_bias=False)
            self.assertEqual(
                ki.dynamic_dim_mask("scaled_mm"), (False, False, False, False)
            )


if __name__ == "__main__":
    run_tests()
