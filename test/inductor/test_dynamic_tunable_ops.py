# Owner(s): ["module: inductor"]
"""Tests for dynamic-shape TunableOp wildcard dispatch.

Tuning enabled (compile-time autotune):
  concrete miss -> tune, persist concrete, plus a wildcard key if a dim is
  dynamic; concrete hit -> persist the wildcard key if it is still missing.

Tuning disabled (runtime): the AOTI cpp_wrapper emits no
`TunableDynamicDimsGuard`, so the mask is empty and a concrete miss is
resolved by `LookupWildcardFallback`'s token scan. With no match,
`operator()` falls through to `ResultEntry::Default()`, which is the same
non-tunable aten kernel the caller would have used.

Observability caveat, applying to every runtime wildcard test here: nothing
reports whether a wildcard actually served a dispatch. A wildcard hit and a
fall-through to the non-tunable kernel produce identical numerics and add no
concrete entry, so the runtime assertions (output matches the
tunable-disabled reference, no new concrete entry) are necessary but not
sufficient. The real gate is the persistence side -- the wildcard entry
exists with the expected concrete dims, and its leading-dim wildcarding is
self-consistent with the transpose flags
(`_assert_ld_wildcarding_consistent`).

Covered: addmm, mm, bmm, baddbmm and _scaled_mm, each for all three runtime
outcomes (concrete hit, wildcard hit, both-miss), plus NN/NT/TN/TT mm layouts
to exercise the swapped_mn M<->N mask remap. Not covered: addbmm and other
batched-add variants, and scaled GEMM beyond the FP8 tensorwise shape.

Platform note: this suite targets ROCm, but OSS CI shards it onto NVIDIA too
(the fbcode BUCK exclusion does not apply there). On NVIDIA `setUpClass` turns
on the TunableOp numerical check so the tuner cannot select a candidate that
disagrees with the untuned kernel by more than the tolerance the assertions
use.
"""

# pyre-strict

import os
import shutil
import tempfile
import unittest
from collections.abc import Callable
from typing import cast, TypeAlias

import sympy

import torch
import torch.cuda.tunable
from torch._inductor import config as inductor_config
from torch._inductor.kernel_inputs import MMKernelInputs
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ROCM, TestCase


DEVICE: str = "cuda"
DTYPE: torch.dtype = torch.bfloat16
GEMM_ATOL: float = 1e-2
GEMM_RTOL: float = 1e-2
_TunableResultEntry: TypeAlias = tuple[str, str, str, float]


def _get_tunable_results() -> list[_TunableResultEntry]:
    return cast(list[_TunableResultEntry], torch.cuda.tunable.get_results())


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


def _entries_for(
    op_substr: str, *required_dim_tokens: str
) -> list[_TunableResultEntry]:
    """Entries whose op-signature contains `op_substr` and whose params
    signature contains every `_<token>_`. Substring matching means callers
    need not know the cuBLAS vs PyTorch dim ordering."""
    out: list[_TunableResultEntry] = []
    for entry in _get_tunable_results():
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
    for entry in _get_tunable_results():
        op_sig, params_sig, _, _ = entry
        if op_substr in op_sig and "*" in params_sig:
            return True
    return False


def _has_wildcard_with_dims(op_substr: str, *dims: int) -> bool:
    """True if a wildcard entry's m/n/k tokens include every one of `dims`.
    Restricted to those three positions so a coincidental lda/ldb/ldc or
    batch token cannot produce a false positive."""
    for entry in _get_tunable_results():
        op_sig, params_sig, _, _ = entry
        if op_substr not in op_sig or "*" not in params_sig:
            continue
        _, _, m_tok, n_tok, k_tok, _, _, _ = _parse_gemm_params_sig(params_sig)
        mnk = (m_tok, n_tok, k_tok)
        if all(str(d) in mnk for d in dims):
            return True
    return False


# Positions of (lda, ldb, ldc) tokens right after the "_ld_" marker.
_LD_MARKER = "_ld_"


def _parse_gemm_params_sig(
    params_sig: str,
) -> tuple[str, str, str, str, str, str, str, str]:
    """Split any Gemm*Params signature into
    ``(transa, transb, m, n, k, lda, ldb, ldc)`` string tokens, each a decimal
    integer or ``"*"``. Works across all variants because m/n/k are always the
    first three tokens after ``{ta}{tb}`` and lda/ldb/ldc the first three after
    ``_ld_``, whatever else the signature appends."""
    transa, transb = params_sig[0], params_sig[1]
    ld_idx = params_sig.find(_LD_MARKER)
    if ld_idx == -1:
        raise AssertionError(f"no '{_LD_MARKER}' marker in signature: {params_sig}")
    prefix_toks = params_sig[:ld_idx].split("_")
    m_tok, n_tok, k_tok = prefix_toks[1], prefix_toks[2], prefix_toks[3]
    ld_toks = params_sig[ld_idx + len(_LD_MARKER) :].split("_")
    lda_tok, ldb_tok, ldc_tok = ld_toks[0], ld_toks[1], ld_toks[2]
    return transa, transb, m_tok, n_tok, k_tok, lda_tok, ldb_tok, ldc_tok


def _assert_ld_wildcarding_consistent(test: TestCase, op_substr: str) -> None:
    """Assert every persisted wildcard entry wildcards its leading dims in
    step with its own transpose flags -- the invariant
    ``ShouldWildcardLda/Ldb/Ldc`` must satisfy: lda tracks M when transa is N
    else K, ldb tracks K when transb is N else N, ldc always tracks M.

    Reading both the flags and the ``*`` positions out of the same signature
    keeps the check independent of the M<->N remap. An inverted
    ``UsesMForLda`` wildcards lda on the wrong dim whenever exactly one of
    {m, k} is dynamic, which this catches."""
    checked = 0
    for op_sig, params_sig, _, _ in _get_tunable_results():
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


# C++ reads this env var (lazily, on first open) for the untuned-GEMM output
# path and inserts the device ordinal before the extension:
# `foo.csv` -> `foo<device>.csv`. See TuningContext::GetUntunedFile.
_UNTUNED_FILENAME_ENV = "PYTORCH_TUNABLEOP_UNTUNED_FILENAME"


def _read_untuned_lines(stem_path: str) -> list[str]:
    """Read back the untuned-GEMM file written for `stem_path`, accounting for
    the device ordinal the C++ layer splices in before the extension."""
    device = torch.cuda.current_device()
    root, ext = os.path.splitext(stem_path)
    actual = f"{root}{device}{ext}"
    if not os.path.exists(actual):
        return []
    with open(actual) as f:
        return [line.strip() for line in f if line.strip()]


def _untuned_has(lines: list[str], op_substr: str, m: int, n: int, k: int) -> bool:
    """True if any untuned line records a concrete (non-wildcard) shape for
    `op_substr` whose params signature contains all of (m, n, k).

    Each line is `op_signature,params_signature` (the BLAS_PARAMS suffix is only
    present when PYTORCH_TUNABLEOP_BLAS_LOG=1, which these tests do not set).
    Dim matching is order-independent so it does not depend on the cuBLAS
    row-major M<->N swap in the persisted concrete signature."""
    for line in lines:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        op_sig, params_sig = parts[0], parts[1]
        if op_substr not in op_sig or "*" in params_sig:
            continue
        padded = "_" + params_sig + "_"
        if all(f"_{d}_" in padded for d in (m, n, k)):
            return True
    return False


class DynamicTunableOpsTest(TestCase):
    """Verification of the full tuning enable/disable x dynamic-dim matrix.

    Each test clears the process-global TuningResultsManager in setUp so
    concrete and wildcard entries cannot leak between tests.
    """

    _tmpdir: str = ""
    _tmp_results_path: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("cuda not available")
        # Redirect TunableOp persistence to a fresh per-process tempfile so
        # this run never appends to (or lazily loads) the shared default
        # "tunableop_results.csv".
        cls._tmpdir = tempfile.mkdtemp(prefix="dynamic_tunable_ops_test_")
        cls._tmp_results_path = os.path.join(cls._tmpdir, "tunable_results.csv")
        torch.cuda.tunable.set_filename(cls._tmp_results_path, False)
        # TunableOp's numerical check is off by default, so on NVIDIA the
        # winning candidate can drift past GEMM_ATOL/GEMM_RTOL from
        # `gemm_internal` -- the reference every "output matches" assertion
        # uses. Screening at that same tolerance drops such candidates during
        # tuning; ROCm does not need it and the check is not free.
        if not TEST_WITH_ROCM:
            torch.cuda.tunable.set_numerical_check_tolerances(
                True, GEMM_ATOL, GEMM_RTOL
            )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._tmpdir:
            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def setUp(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        torch.cuda.tunable._clear_all()
        torch.cuda.tunable.wildcard_fallback_enable(True)

    def tearDown(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        torch.cuda.tunable.wildcard_fallback_enable(False)

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

        op_entries = _entries_for("GemmAndBiasTunableOp")
        self.assertTrue(
            _has_concrete_entry("GemmAndBiasTunableOp", m, n, k),
            f"expected concrete entry for ({m},{n},{k}), got {op_entries}",
        )
        wildcard_entries = [entry for entry in op_entries if "*" in entry[1]]
        self.assertEqual(
            wildcard_entries,
            [],
            f"unexpected wildcard entries for non-dynamic call: {op_entries}",
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
        before = len(_get_tunable_results())

        # Run again with same shape, no dynamic mask.
        torch.addmm(bias, mat1, mat2)
        after = len(_get_tunable_results())
        self.assertEqual(
            before,
            after,
            "concrete-hit + no-dynamic case should not add entries",
        )

    def test_tuning_enabled_dynamic_concrete_hit_persists_wildcard_if_missing(
        self,
    ) -> None:
        """Tuning enabled + dynamic dim + concrete hit -> persist the
        wildcard now.

        addmm's row-major dispatch swaps inductor (M, N) into BLAS (n, m), so
        `launchTunableGemmAndBias` remaps the test's M=True mask to BLAS N and
        the persisted wildcard carries `*` in the BLAS-n slot, with BLAS-m
        equal to inductor-N and BLAS-k equal to inductor-K."""
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
        # actual BLAS-frame signature swaps M<->N. `_has_wildcard_with_dims`
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
        before = len(_get_tunable_results())
        out = torch.addmm(bias, mat1, mat2)
        after = len(_get_tunable_results())

        self.assertEqual(before, after, "tuning disabled should not add entries")
        self.assertEqual(
            out,
            ref,
            atol=GEMM_ATOL,
            rtol=GEMM_RTOL,
            msg="tunable-enabled output should match tunable-disabled reference",
        )

    def test_tuning_disabled_concrete_miss_wildcard_hit_dispatches_via_wildcard(
        self,
    ) -> None:
        """Tuning disabled + concrete miss + wildcard match -> dispatch via
        the wildcard. Phase B deliberately runs without a `dynamic_dims_mask`
        to mirror the AOTI runtime, which emits no guard."""
        m_tuned, n, k = 47, 2053, 1019
        m_test = 53  # different M, same wildcard pattern
        bias_t, mat1_t, mat2_t = _addmm(m_tuned, n, k, seed=6)
        bias_x, mat1_x, mat2_x = _addmm(m_test, n, k, seed=7)

        # Phase A (compile-time analog): tune at m_tuned with M dynamic
        # so the wildcard `tn_*_n_k_...` is seeded into the manager.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        with torch.cuda.tunable.dynamic_dims_mask(M=True):
            torch.addmm(bias_t, mat1_t, mat2_t)

        # Shape-specific: the BLAS-frame wildcard keeps n and k concrete (M is
        # the wildcarded dim), so match on this test's own (n, k) rather than
        # any wildcard for the op -- otherwise a wildcard seeded by another
        # test would satisfy this assertion.
        self.assertTrue(
            _has_wildcard_with_dims("GemmAndBiasTunableOp", n, k),
            "expected wildcard entry with concrete (n, k) after phase A tuning",
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
        self.assertEqual(
            out,
            ref,
            atol=GEMM_ATOL,
            rtol=GEMM_RTOL,
            msg="wildcard-fallback dispatched output should match "
            "tunable-disabled reference",
        )

    def test_tuning_disabled_both_miss_falls_back_to_non_tunable_aten(self) -> None:
        """Tuning disabled + both miss -> falls back to the non-tunable aten
        kernel with no entries added. Uses a large-K shape that historically
        crashed on MI300X."""
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
        before_results = len(_get_tunable_results())
        out = torch.addmm(bias, mat1, mat2)  # must NOT crash
        after_results = len(_get_tunable_results())

        self.assertEqual(
            before_results,
            after_results,
            "both-miss + tuning disabled must not add any entries",
        )
        self.assertEqual(
            out,
            ref,
            atol=GEMM_ATOL,
            rtol=GEMM_RTOL,
            msg="fallback dispatch result should match tunable-disabled reference",
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
        """torch.mm: tune at m_tuned with M dynamic, then query a different M
        at runtime with no mask. Gated on the observable proxies only; see the
        module-level observability caveat."""
        m_tuned, n, k = 61, 2063, 1021
        m_test = 79
        mat1_t, mat2_t = _mm(m_tuned, n, k, seed=50)
        mat1_x, mat2_x = _mm(m_test, n, k, seed=51)

        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        with torch.cuda.tunable.dynamic_dims_mask(M=True):
            torch.mm(mat1_t, mat2_t)
        self.assertTrue(
            _has_wildcard_with_dims("GemmTunableOp", n, k),
            "expected GemmTunableOp wildcard entry with concrete (n, k) after "
            "phase A tuning",
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
        self.assertEqual(
            out,
            ref,
            atol=GEMM_ATOL,
            rtol=GEMM_RTOL,
            msg="mm wildcard-fallback output should match tunable-disabled reference",
        )

    def test_tuning_disabled_bmm_concrete_miss_wildcard_hit(self) -> None:
        """torch.bmm: tune at m_tuned with M dynamic, then query a different M
        at runtime. bmm is not subject to the M<->N swap, so the mask lands
        directly on the BLAS m slot."""
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
            _has_wildcard_with_dims("GemmStridedBatchedTunableOp", n, k),
            "expected GemmStridedBatchedTunableOp wildcard entry with concrete "
            "(n, k) after phase A",
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
        self.assertEqual(
            out,
            ref,
            atol=GEMM_ATOL,
            rtol=GEMM_RTOL,
            msg="bmm wildcard-fallback output should match tunable-disabled reference",
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
        before = len(_get_tunable_results())

        torch.cuda.tunable.tuning_enable(False)
        out = torch.mm(mat1, mat2)
        after = len(_get_tunable_results())

        self.assertEqual(before, after, "mm concrete-hit must not add an entry")
        self.assertEqual(
            out,
            ref,
            atol=GEMM_ATOL,
            rtol=GEMM_RTOL,
            msg="mm concrete-hit output should match reference",
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
        before = len(_get_tunable_results())
        out = torch.mm(mat1, mat2)  # must NOT crash
        after = len(_get_tunable_results())

        self.assertEqual(before, after, "mm both-miss must not add an entry")
        self.assertEqual(
            out,
            ref,
            atol=GEMM_ATOL,
            rtol=GEMM_RTOL,
            msg="mm both-miss fallback output should match reference",
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
        before = len(_get_tunable_results())

        torch.cuda.tunable.tuning_enable(False)
        out = torch.bmm(b1, b2)
        after = len(_get_tunable_results())

        self.assertEqual(before, after, "bmm concrete-hit must not add an entry")
        self.assertEqual(
            out,
            ref,
            atol=GEMM_ATOL,
            rtol=GEMM_RTOL,
            msg="bmm concrete-hit output should match reference",
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
        before = len(_get_tunable_results())
        out = torch.bmm(b1, b2)  # must NOT crash
        after = len(_get_tunable_results())

        self.assertEqual(before, after, "bmm both-miss must not add an entry")
        self.assertEqual(
            out,
            ref,
            atol=GEMM_ATOL,
            rtol=GEMM_RTOL,
            msg="bmm both-miss fallback output should match reference",
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
            _has_wildcard_with_dims("GemmStridedBatchedTunableOp", n, k),
            "expected GemmStridedBatchedTunableOp wildcard entry with concrete "
            "(n, k) after baddbmm phase A",
        )

        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        ref = torch.baddbmm(bias_x, b1_x, b2_x)

        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        out = torch.baddbmm(bias_x, b1_x, b2_x)

        self.assertEqual(
            out,
            ref,
            atol=GEMM_ATOL,
            rtol=GEMM_RTOL,
            msg="baddbmm wildcard-fallback output should match reference",
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

        self.assertEqual(
            out,
            ref,
            atol=GEMM_ATOL,
            rtol=GEMM_RTOL,
            msg="baddbmm both-miss fallback output should match reference",
        )


# --- Layout coverage: verify swap remap across NN/NT/TN/TT ---------------
# PyTorch's BLAS dispatch picks `transa, transb` and the
# `transpose_result` decision based on operand layouts. The
# `cublasCommonArgs::swapped_mn` flag tracks `transpose_result` so the
# `launchTunableGemmAndBias` mask remap fires whenever the dispatch
# swaps inductor (M, N) -> BLAS (n, m). These tests force each
# (transa, transb) combination by transposing input tensors and
# verify the wildcard-tune -> wildcard-fallback round trip works
# correctly for every layout.


class LayoutCoverageWildcardTest(TestCase):
    """The dynamic-mask + wildcard-fallback contract across all four
    (transa, transb) layouts, with M/N/K dynamic in turn.

    The K-dynamic cases are what cover lda: mm's row-major dispatch swaps
    inductor (M, N) into BLAS (n, m), so an M-dynamic call lands on BLAS-n and
    never touches lda, while K is not swapped and forces lda to be wildcarded
    exactly when transa == 'T'."""

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
            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def setUp(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        torch.cuda.tunable._clear_all()
        torch.cuda.tunable.wildcard_fallback_enable(True)

    def tearDown(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        torch.cuda.tunable.wildcard_fallback_enable(False)

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
        # Op-wide `_has_wildcard_entry` (not the shape-specific
        # `_has_wildcard_with_dims` used elsewhere) is deliberate here: every
        # layout variant in this class shares the same (n, k), so a
        # shape-specific check could not tell them apart. The per-entry
        # `_assert_ld_wildcarding_consistent` below is the real gate.
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
        # -> expected to dispatch via the wildcard fallback. Gated on the
        # observable proxies (no new concrete entry, output matches the
        # tunable-disabled reference); the persistence-side white-box check
        # above is the real gate on the wildcarding logic.
        before = len(_get_tunable_results())
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        out = torch.mm(mat1_x, mat2_x)
        after = len(_get_tunable_results())
        self.assertEqual(
            before,
            after,
            f"layout (transa={transa}, transb={transb}, dynamic={dynamic}): "
            f"runtime dispatch with tuning disabled must not add a new entry",
        )
        self.assertEqual(
            out,
            ref,
            atol=1e-2,
            rtol=1e-2,
            msg=f"layout (transa={transa}, transb={transb}, dynamic={dynamic}): "
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

    def test_layout_NN_dynamic_n(self) -> None:
        """N dynamic. For torch.mm the row-major dispatch swaps inductor
        (M, N) into BLAS (n, m), so an N-dynamic call lands the dynamic bit
        on BLAS-m and forces ldc (and lda for transa == 'N') to wildcard."""
        self._round_trip_for_layout(
            transa=False, transb=False, seed_offset=32, dynamic="N"
        )

    def test_layout_NT_dynamic_n(self) -> None:
        self._round_trip_for_layout(
            transa=False, transb=True, seed_offset=36, dynamic="N"
        )

    def test_layout_TN_dynamic_n(self) -> None:
        self._round_trip_for_layout(
            transa=True, transb=False, seed_offset=40, dynamic="N"
        )

    def test_layout_TT_dynamic_n(self) -> None:
        self._round_trip_for_layout(
            transa=True, transb=True, seed_offset=44, dynamic="N"
        )


# --- Layout coverage for the batched path (bmm / baddbmm) ---
# Why mm layout coverage is not duplicated for every op:
#   - addmm: torch.mm and torch.addmm both route through
#     addmm_out_cuda_impl (mm calls it with beta=0, alpha=1), so the mm
#     layout tests above already exercise addmm's transpose-flag derivation
#     (cublasCommonArgs), the swapped_mn mask remap (launchTunableGemmAndBias
#     / launchGemmCublas) and the ShouldWildcardLda/Ldb/Ldc logic verbatim.
#     No separate addmm layout tests are needed.
#   - _scaled_mm: reuses the SAME cublasCommonArgs transpose/swapped_mn
#     derivation and the SAME ShouldWildcardLd* logic (ScaledGemmParams).
#     Only the mask-remap stamping is a distinct function
#     (_tunable_scaled_gemm_rocm, ROCm-only) that reads the same swapped_mn
#     flag, and it is already exercised by the M-dynamic round trip in
#     ScaledGemmTunableOpFP8Test. No separate scaled layout matrix is added.
#   - bmm / baddbmm: DIFFERENT code path. baddbmm_out_cuda_impl does NOT
#     build a cublasCommonArgs; it derives transa/transb from its own
#     prepare_batch_matrix_for_cublas plus a local transpose_result, and
#     performs the M<->N mask remap itself via a TunableDynamicDimsGuard
#     (not the shared swapped_mn field). Only the leading-dim wildcarding is
#     shared (GemmStridedBatchedParams -> ShouldWildcardLd*). The mm layout
#     tests never touch this batched transpose/remap code, so it gets its own
#     coverage below. bmm and baddbmm share baddbmm_out_cuda_impl, so this bmm
#     coverage also covers baddbmm's layout path (baddbmm's bias handling is
#     covered by the round-trip tests in DynamicTunableOpsTest).


class BmmLayoutCoverageWildcardTest(TestCase):
    """Batched analog of LayoutCoverageWildcardTest. bmm derives its
    transpose flags and M<->N remap in baddbmm_out_cuda_impl, a separate path
    from the mm/addmm launchers."""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("cuda not available")
        cls._tmpdir = tempfile.mkdtemp(prefix="bmm_layout_coverage_test_")
        cls._tmp_results_path = os.path.join(cls._tmpdir, "tunable_results.csv")
        torch.cuda.tunable.set_filename(cls._tmp_results_path, False)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._tmpdir:
            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def setUp(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        torch.cuda.tunable._clear_all()
        torch.cuda.tunable.wildcard_fallback_enable(True)

    def tearDown(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        torch.cuda.tunable.wildcard_fallback_enable(False)

    @staticmethod
    def _make_layout(
        b: int, m: int, k: int, n: int, transa: bool, transb: bool, seed: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (batch1, batch2) for `bmm(batch1, batch2)` so the batched
        BLAS dispatch picks the requested (transa, transb) layout. As in the
        mm helper, a transposed operand is allocated in the swapped shape and
        then transposed on its last two dims so it is non-contiguous without a
        contiguous() copy."""
        torch.manual_seed(seed)
        if not transa:
            batch1 = torch.randn(b, m, k, dtype=DTYPE, device=DEVICE)
        else:
            batch1 = torch.randn(b, k, m, dtype=DTYPE, device=DEVICE).transpose(-2, -1)
        if not transb:
            batch2 = torch.randn(b, k, n, dtype=DTYPE, device=DEVICE)
        else:
            batch2 = torch.randn(b, n, k, dtype=DTYPE, device=DEVICE).transpose(-2, -1)
        return batch1, batch2

    def _round_trip_for_layout(
        self, transa: bool, transb: bool, seed_offset: int, dynamic: str = "M"
    ) -> None:
        """Tune-then-fallback round trip for one batched layout, taking the
        `dynamic` dim (one of "M"/"N"/"K") symbolic. Tuning and runtime shapes
        differ only in that dim. Shapes are kept disjoint from the mm layout
        tests via a distinct op (GemmStridedBatchedTunableOp) and seed range;
        see the module-level isolation note (the in-memory results table is
        process-global)."""
        b = 8
        base_m, n, k = 41 + seed_offset, 257, 251
        delta = 12
        tuned = {"m": base_m, "n": n, "k": k}
        test = {"m": base_m, "n": n, "k": k}
        test[dynamic.lower()] += delta

        b1_t, b2_t = self._make_layout(
            b,
            tuned["m"],
            tuned["k"],
            tuned["n"],
            transa,
            transb,
            seed=300 + seed_offset,
        )
        b1_x, b2_x = self._make_layout(
            b, test["m"], test["k"], test["n"], transa, transb, seed=400 + seed_offset
        )

        # Reference (tunable disabled).
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        ref = torch.bmm(b1_x, b2_x)

        # Phase A: tune at the tuned shape with the chosen dim dynamic ->
        # wildcard persisted.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        with torch.cuda.tunable.dynamic_dims_mask(**{dynamic: True}):
            torch.bmm(b1_t, b2_t)
        # Op-wide `_has_wildcard_entry` (not the shape-specific
        # `_has_wildcard_with_dims`) is deliberate here: every layout variant in
        # this class shares the same (n, k), so a shape-specific check could not
        # tell them apart. `_assert_ld_wildcarding_consistent` below is the real
        # per-entry gate.
        self.assertTrue(
            _has_wildcard_entry("GemmStridedBatchedTunableOp"),
            f"expected GemmStridedBatchedTunableOp wildcard entry after tuning "
            f"layout (transa={transa}, transb={transb}, dynamic={dynamic})",
        )
        # White-box: the batched path shares ShouldWildcardLda/Ldb/Ldc via
        # GemmStridedBatchedParams, so a broken ld->dim mapping surfaces here
        # just as it does for mm.
        _assert_ld_wildcarding_consistent(self, "GemmStridedBatchedTunableOp")

        # Phase C: runtime, no mask, shape differs only in the dynamic dim.
        # Gated on the observable proxies (no new concrete entry, output
        # matches reference); the white-box check above is the real gate on
        # the wildcarding logic.
        before = len(_get_tunable_results())
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        out = torch.bmm(b1_x, b2_x)
        after = len(_get_tunable_results())
        self.assertEqual(
            before,
            after,
            f"layout (transa={transa}, transb={transb}, dynamic={dynamic}): "
            f"runtime dispatch with tuning disabled must not add a new entry",
        )
        self.assertEqual(
            out,
            ref,
            atol=1e-2,
            rtol=1e-2,
            msg=f"layout (transa={transa}, transb={transb}, dynamic={dynamic}): "
            f"wildcard-fallback dispatch output must match tunable-disabled "
            f"reference",
        )

    def test_bmm_layout_NN(self) -> None:
        """No transpose on either operand."""
        self._round_trip_for_layout(transa=False, transb=False, seed_offset=0)

    def test_bmm_layout_NT(self) -> None:
        """batch1 contiguous, batch2 transposed."""
        self._round_trip_for_layout(transa=False, transb=True, seed_offset=4)

    def test_bmm_layout_TN(self) -> None:
        """batch1 transposed, batch2 contiguous."""
        self._round_trip_for_layout(transa=True, transb=False, seed_offset=8)

    def test_bmm_layout_TT(self) -> None:
        """Both operands transposed."""
        self._round_trip_for_layout(transa=True, transb=True, seed_offset=12)

    def test_bmm_layout_NN_dynamic_k(self) -> None:
        """K dynamic exercises lda/ldb on the batched path."""
        self._round_trip_for_layout(
            transa=False, transb=False, seed_offset=16, dynamic="K"
        )

    def test_bmm_layout_NT_dynamic_k(self) -> None:
        self._round_trip_for_layout(
            transa=False, transb=True, seed_offset=20, dynamic="K"
        )

    def test_bmm_layout_TN_dynamic_k(self) -> None:
        """transa == 'T' + K dynamic forces lda to be wildcarded on the
        batched path too."""
        self._round_trip_for_layout(
            transa=True, transb=False, seed_offset=24, dynamic="K"
        )

    def test_bmm_layout_TT_dynamic_k(self) -> None:
        self._round_trip_for_layout(
            transa=True, transb=True, seed_offset=28, dynamic="K"
        )

    def test_bmm_layout_NN_dynamic_n(self) -> None:
        """N dynamic exercises the batched M<->N remap independent of the
        shared swapped_mn field."""
        self._round_trip_for_layout(
            transa=False, transb=False, seed_offset=32, dynamic="N"
        )

    def test_bmm_layout_NT_dynamic_n(self) -> None:
        self._round_trip_for_layout(
            transa=False, transb=True, seed_offset=36, dynamic="N"
        )

    def test_bmm_layout_TN_dynamic_n(self) -> None:
        self._round_trip_for_layout(
            transa=True, transb=False, seed_offset=40, dynamic="N"
        )

    def test_bmm_layout_TT_dynamic_n(self) -> None:
        self._round_trip_for_layout(
            transa=True, transb=True, seed_offset=44, dynamic="N"
        )


# --- ScaledGemmTunableOp coverage (FP8 _scaled_mm) -----------------------
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


@unittest.skipIf(
    torch.version.hip is None,
    "_scaled_gemm only reaches TunableOp under USE_ROCM; off ROCm "
    "tunable_op_enabled is hardcoded false (ScaledBlas.cpp) so no "
    "ScaledGemmTunableOp entry is ever produced",
)
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
            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def setUp(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        torch.cuda.tunable._clear_all()
        torch.cuda.tunable.wildcard_fallback_enable(True)

    def tearDown(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        torch.cuda.tunable.wildcard_fallback_enable(False)

    @staticmethod
    def _scaled_mm_inputs(
        m: int, n: int, k: int, seed: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build (a_fp8, b_fp8, scale_a, scale_b) for a tensorwise-
        scaled `_scaled_mm(a, b, scale_a, scale_b)` of shape (m, n, k).

        a is MxK row-major, b is KxN -> b must be col-major for the
        cuBLAS scaled-mm path (the typical inductor pattern after the
        `fp8_float_model_dynamic_quantization_tensorwise` rewrite).
        """
        torch.manual_seed(seed)
        # Random BF16 then quantize to FP8 e4m3fn with a tensorwise scale.
        a_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE)
        b_bf16 = torch.randn(k, n, dtype=torch.bfloat16, device=DEVICE)
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
        # `_scaled_mm` with a column-major `b` produces an MxN bf16 out.
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
        m_tuned, n, k = 1024, 1024, 1152
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
            e for e in _get_tunable_results() if "ScaledGemmTunableOp" in e[0]
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

        self.assertEqual(
            out,
            ref,
            atol=5e-2,
            rtol=5e-2,
            msg="ScaledGemmTunableOp dispatch output should match "
            "tunable-disabled reference (either via wildcard fallback "
            "or via non-tunable aten fallback)",
        )

    def test_scaled_gemm_both_miss_falls_back_safely(self) -> None:
        """_scaled_mm both-miss: tunable enabled, tuning disabled, with no
        concrete and no wildcard entry primed -> must produce correct output
        without crashing. operator() resolves the miss to
        ResultEntry::Default(), which is at::cuda::blas::scaled_gemm, the same
        kernel _scaled_gemm falls back to when the dispatch reports failure."""
        m, n, k = 4096, 1024, 1152
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

        self.assertEqual(
            out,
            ref,
            atol=5e-2,
            rtol=5e-2,
            msg="scaled both-miss fallback output should match "
            "tunable-disabled reference",
        )


# --- Legacy "non-dynamic" behavior (BEFORE the wildcard feature) --------
# These tests assert what TunableOp does when no `dynamic_dims_mask`
# is ever pushed (the pre-feature world): only concrete-key entries
# get persisted at compile-time tuning, and runtime concrete-miss
# queries fall through to the non-tunable aten path. These are the
# canonical "before" behavior gates -- they document and lock in what
# the system reverts to when `cuda.autotune_tunableop_dynamic_dims_
# wildcard=False` (the kill-switch for the dynamic-tunable-ops
# feature).


class LegacyConcreteOnlyTunableOpsTest(TestCase):
    """Pre-feature behavior: with no `dynamic_dims_mask` pushed, only concrete
    entries persist and a runtime concrete miss falls through safely."""

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
            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def setUp(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        torch.cuda.tunable._clear_all()
        torch.cuda.tunable.wildcard_fallback_enable(False)

    def tearDown(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        torch.cuda.tunable.wildcard_fallback_enable(False)

    def _entries_for_op(self, op_substr: str) -> list[_TunableResultEntry]:
        return [e for e in _get_tunable_results() if op_substr in e[0]]

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
        before = len(_get_tunable_results())
        out = torch.addmm(bias, mat1, mat2)  # NO mask, NO concrete entry
        after = len(_get_tunable_results())

        self.assertEqual(
            before,
            after,
            "concrete-miss + tuning-disabled must not add any entries",
        )
        self.assertEqual(
            out,
            ref,
            atol=1e-2,
            rtol=1e-2,
            msg="fallback dispatch result should match tunable-disabled reference",
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
        before = len(_get_tunable_results())

        # Runtime: same shape -> concrete hit, no new entry, output
        # matches.
        torch.cuda.tunable.tuning_enable(False)
        out = torch.addmm(bias, mat1, mat2)
        after = len(_get_tunable_results())

        self.assertEqual(
            before,
            after,
            "concrete-hit dispatch must not add any new entry",
        )
        self.assertEqual(
            out,
            ref,
            atol=1e-2,
            rtol=1e-2,
            msg="concrete-hit dispatch result should match reference",
        )


# --- PYTORCH_TUNABLEOP_RECORD_UNTUNED collection on a concrete miss ---
# Regression coverage for the runtime record-untuned path across every GEMM
# category. TunableOp::operator() records the concrete shape on every miss,
# before consulting the wildcard entries, so the offline-tuning workflow sees
# the shape whether it ends up served by a wildcard or by the non-tunable aten
# fallback. Recording only on a total miss would silently drop exactly the
# shapes a wildcard is approximating.


class RecordUntunedConcreteMissTest(TestCase):
    """PYTORCH_TUNABLEOP_RECORD_UNTUNED collection on a runtime concrete miss,
    for every GEMM category, whether or not a wildcard then serves the call. A
    concrete hit records nothing.

    Untuned output is redirected per test via
    PYTORCH_TUNABLEOP_UNTUNED_FILENAME; record_untuned_enable(False) flushes
    and closes that file and clears the C++ dedup set."""

    _tmpdir: str = ""
    _tmp_results_path: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("cuda not available")
        cls._tmpdir = tempfile.mkdtemp(prefix="record_untuned_test_")
        cls._tmp_results_path = os.path.join(cls._tmpdir, "tunable_results.csv")
        torch.cuda.tunable.set_filename(cls._tmp_results_path, False)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._tmpdir:
            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def setUp(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        torch.cuda.tunable.record_untuned_enable(False)
        torch.cuda.tunable._clear_all()
        torch.cuda.tunable.wildcard_fallback_enable(True)

    def tearDown(self) -> None:
        torch.cuda.tunable.record_untuned_enable(False)
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        torch.cuda.tunable.wildcard_fallback_enable(False)

    def _record_untuned_run(self, stem: str, run_op: Callable[[], object]) -> list[str]:
        """Run `run_op` at runtime (TunableOp enabled, tuning disabled) with
        record-untuned redirected to `stem`, then close the file (flush) and
        return its recorded lines."""
        # Close first so the next open re-reads our env-provided filename and
        # the untuned dedup set starts empty.
        torch.cuda.tunable.record_untuned_enable(False)
        prev = os.environ.get(_UNTUNED_FILENAME_ENV)
        os.environ[_UNTUNED_FILENAME_ENV] = stem
        try:
            torch.cuda.tunable.enable(True)
            torch.cuda.tunable.tuning_enable(False)
            torch.cuda.tunable.record_untuned_enable(True)
            run_op()
        finally:
            torch.cuda.tunable.record_untuned_enable(False)
            if prev is None:
                os.environ.pop(_UNTUNED_FILENAME_ENV, None)
            else:
                os.environ[_UNTUNED_FILENAME_ENV] = prev
        return _read_untuned_lines(stem)

    def _assert_records_concrete_miss(
        self,
        op_substr: str,
        m: int,
        n: int,
        k: int,
        run_op: Callable[[], object],
    ) -> None:
        stem = os.path.join(self._tmpdir, f"untuned_{op_substr}_{m}_{n}_{k}.csv")
        lines = self._record_untuned_run(stem, run_op)
        self.assertTrue(
            _untuned_has(lines, op_substr, m, n, k),
            f"concrete miss ({m},{n},{k}) for {op_substr} must be appended to "
            f"the untuned file when record-untuned is enabled; got {lines}",
        )

    # -- Total miss records for every GEMM category ----------------------

    def test_addmm_concrete_miss_records_untuned(self) -> None:
        m, n, k = 107, 3001, 401
        bias, mat1, mat2 = _addmm(m, n, k, seed=70)
        self._assert_records_concrete_miss(
            "GemmAndBiasTunableOp", m, n, k, lambda: torch.addmm(bias, mat1, mat2)
        )

    def test_mm_concrete_miss_records_untuned(self) -> None:
        m, n, k = 109, 3011, 409
        mat1, mat2 = _mm(m, n, k, seed=71)
        self._assert_records_concrete_miss(
            "GemmTunableOp", m, n, k, lambda: torch.mm(mat1, mat2)
        )

    def test_bmm_concrete_miss_records_untuned(self) -> None:
        b, m, n, k = 8, 113, 3019, 419
        b1, b2 = _bmm(b, m, n, k, seed=72)
        self._assert_records_concrete_miss(
            "GemmStridedBatchedTunableOp", m, n, k, lambda: torch.bmm(b1, b2)
        )

    def test_baddbmm_concrete_miss_records_untuned(self) -> None:
        b, m, n, k = 8, 127, 3023, 421
        bias, b1, b2 = _baddbmm(b, m, n, k, seed=73)
        self._assert_records_concrete_miss(
            "GemmStridedBatchedTunableOp",
            m,
            n,
            k,
            lambda: torch.baddbmm(bias, b1, b2),
        )

    @unittest.skipIf(
        torch.version.hip is None,
        "_scaled_gemm only reaches TunableOp under USE_ROCM; off ROCm "
        "tunable_op_enabled is hardcoded false (ScaledBlas.cpp) so nothing "
        "is ever recorded",
    )
    def test_scaled_mm_concrete_miss_records_untuned(self) -> None:
        if not hasattr(torch, "float8_e4m3fn"):
            raise unittest.SkipTest("torch.float8_e4m3fn not available")
        # Use distinct aligned N and K to verify the helper's K-by-N layout.
        m, n, k = 2560, 1408, 1536
        a, b, sa, sb = ScaledGemmTunableOpFP8Test._scaled_mm_inputs(m, n, k, seed=74)

        # Confirm _scaled_mm is usable here before asserting on the tunable
        # path (mirrors the skips in ScaledGemmTunableOpFP8Test).
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        try:
            torch._scaled_mm(a, b, sa, sb, out_dtype=torch.bfloat16)
        except RuntimeError as e:
            raise unittest.SkipTest(
                f"_scaled_mm not supported in this configuration: {e}"
            ) from e

        self._assert_records_concrete_miss(
            "ScaledGemmTunableOp",
            m,
            n,
            k,
            lambda: torch._scaled_mm(a, b, sa, sb, out_dtype=torch.bfloat16),
        )

    # -- Already-covered shapes must NOT record --------------------------

    def test_concrete_hit_does_not_record_untuned(self) -> None:
        """A runtime concrete hit is already tuned, so nothing is recorded."""
        m, n, k = 131, 3041, 431
        bias, mat1, mat2 = _addmm(m, n, k, seed=75)

        # Phase A: tune the concrete shape (tuning on, outside the record
        # window; tuning writes the results file, never the untuned file).
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        torch.addmm(bias, mat1, mat2)
        self.assertTrue(
            _has_concrete_entry("GemmAndBiasTunableOp", m, n, k),
            f"expected concrete entry for ({m},{n},{k}) after phase A tuning",
        )

        stem = os.path.join(self._tmpdir, f"untuned_concrete_hit_{m}.csv")
        lines = self._record_untuned_run(stem, lambda: torch.addmm(bias, mat1, mat2))
        self.assertFalse(
            _untuned_has(lines, "GemmAndBiasTunableOp", m, n, k),
            f"concrete hit must not record an untuned entry; got {lines}",
        )

    def test_wildcard_fallback_hit_still_records_untuned(self) -> None:
        """A wildcard only approximates the shape -- it was tuned for a
        different one -- so offline tuning must still see the concrete miss."""
        m_tuned, n, k = 137, 3049, 433
        m_test = 139
        bias_t, mat1_t, mat2_t = _addmm(m_tuned, n, k, seed=76)
        bias_x, mat1_x, mat2_x = _addmm(m_test, n, k, seed=77)

        # Phase A: tune with M dynamic so the wildcard is seeded.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        with torch.cuda.tunable.dynamic_dims_mask(M=True):
            torch.addmm(bias_t, mat1_t, mat2_t)
        self.assertTrue(
            _has_wildcard_with_dims("GemmAndBiasTunableOp", n, k),
            "expected wildcard entry with concrete (n, k) after phase A",
        )
        self.assertFalse(
            _has_concrete_entry("GemmAndBiasTunableOp", m_test, n, k),
            "should have no concrete entry for the new M before the runtime call",
        )

        # Runtime: the concrete miss is recorded, then served by the wildcard.
        stem = os.path.join(self._tmpdir, f"untuned_wildcard_hit_{m_test}.csv")
        lines = self._record_untuned_run(
            stem, lambda: torch.addmm(bias_x, mat1_x, mat2_x)
        )
        self.assertTrue(
            _untuned_has(lines, "GemmAndBiasTunableOp", m_test, n, k),
            "a wildcard-served concrete miss must still be recorded as "
            f"untuned; got {lines}",
        )


class _SizeOnlyNode:
    """Minimal input-node stub exposing only get_size(), which is all that
    shapes_symbolic / dynamic_dim_mask consult. Lets us drive
    dynamic_dim_mask with symbolic (sympy) dims on CPU, no GPU needed."""

    def __init__(self, size: tuple[object, ...]) -> None:
        self._size = size

    def get_size(self) -> tuple[object, ...]:
        return self._size


@inductor_config.patch({"cuda.autotune_tunableop_dynamic_dims_wildcard": True})
class DynamicDimMaskOperandSelectionTest(TestCase):
    """CPU unit tests for ``MMKernelInputs.dynamic_dim_mask``: operands come
    from ``mat1_idx``/``mat2_idx``, not the trailing two inputs (scaled GEMM
    is ``[mat_a, mat_b, scale_a, scale_b, (bias)]``, so trailing reads would
    mask the scales), and ``"scaled_mm"`` is recognized as a 2D matmul."""

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
        self.assertEqual(ki.dynamic_dim_mask("scaled_mm"), (True, False, False, False))

    def test_scaled_mm_with_bias_reads_operands(self) -> None:
        ki = self._scaled_mm_inputs(with_bias=True)
        self.assertEqual(ki.dynamic_dim_mask("scaled_mm"), (True, False, False, False))

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
            {"cuda.autotune_tunableop_dynamic_dims_wildcard": False}
        ):
            ki = self._scaled_mm_inputs(with_bias=False)
            self.assertEqual(
                ki.dynamic_dim_mask("scaled_mm"), (False, False, False, False)
            )


if __name__ == "__main__":
    run_tests()
