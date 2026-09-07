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
(the fbcode BUCK exclusion does not apply there). On NVIDIA the shared
`_TunableOpGpuTestBase.setUpClass` turns on the TunableOp numerical check so
the tuner cannot select a candidate that disagrees with the untuned kernel by
more than the tolerance the assertions use, and `tearDownClass` resets it so
the process-global setting cannot leak into a class that never enabled it.
"""

# pyre-strict

import os
import shutil
import tempfile
import unittest
import warnings
from collections.abc import Callable
from typing import cast, TypeAlias

import sympy

import torch
import torch.cuda.tunable
from torch._inductor import config as inductor_config
from torch._inductor.kernel_inputs import MMKernelInputs
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_ROCM,
    TestCase,
)


DEVICE: str = "cuda"
DTYPE: torch.dtype = torch.bfloat16
GEMM_ATOL: float = 1e-2
GEMM_RTOL: float = 1e-2
# FP8 accumulates in a narrower format than bf16, so the scaled-GEMM round
# trips need a looser bound than GEMM_ATOL/GEMM_RTOL.
FP8_GEMM_ATOL: float = 5e-2
FP8_GEMM_RTOL: float = 5e-2
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


def _run_without_tunable_fallback_warning(
    test: TestCase, op: Callable[[], torch.Tensor]
) -> torch.Tensor:
    """Run `op` and assert no "falling back to the non-tunable kernel" warning,
    i.e. the wildcard-selected solution was accepted for the new shape.

    Only the accepted side of the backend compatibility check is covered. The
    rejection branch cannot be forced from Python: which backend a wildcard
    entry carries is not controllable, and varying anything other than the
    dynamic dim misses the wildcard entirely and takes the both-miss path.
    Covering it needs a C++ test calling RocblasGemmOp::Call with a solution
    index known to be invalid for the shape, so Tensile's canSolve rejects
    it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = op()
    fallback_warnings = [
        warning
        for warning in caught
        if "falling back to the non-tunable kernel" in str(warning.message)
    ]
    test.assertEqual(fallback_warnings, [])
    return out


class _TunableOpGpuTestBase(TestCase):
    """Shared setup for every GPU-backed class in this file.

    Deriving this per class is what let the NVIDIA numerical-check setup go
    missing from three of them: the setting is process-global and nothing
    reset it, so classes that never enabled it still ran under the value a
    lexically earlier class had left behind. That protection vanished under
    `-k` filtering or CI's rerun-a-single-test-in-a-new-process path, which
    is exactly when a flake gets re-run. `tearDownClass` now resets it so the
    leak cannot paper over a missing `setUpClass` again.

    Subclasses vary only via `_tmpdir_prefix` and
    `_wildcard_fallback_in_setup`.
    """

    _tmpdir_prefix: str = "tunable_ops_test_"
    # LegacyConcreteOnlyTunableOpsTest asserts pre-feature behavior and wants
    # the fallback off; every other class exercises it.
    _wildcard_fallback_in_setup: bool = True

    _tmpdir: str = ""
    _tmp_results_path: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("cuda not available")
        # Redirect TunableOp persistence to a fresh per-process tempfile so
        # this run never appends to (or lazily loads) the shared default
        # "tunableop_results.csv".
        cls._tmpdir = tempfile.mkdtemp(prefix=cls._tmpdir_prefix)
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
        cls._release_class_state()

    @classmethod
    def _release_class_state(cls) -> None:
        """Undo `setUpClass`. Split out because a subclass `setUpClass` that
        raises SkipTest *after* calling `super().setUpClass()` must run this
        itself -- unittest does not call `tearDownClass` when `setUpClass`
        raises, so the numerical-check setting would leak forward exactly the
        way this base class exists to prevent."""
        if not TEST_WITH_ROCM:
            torch.cuda.tunable.set_numerical_check_tolerances(False)
        if cls._tmpdir:
            shutil.rmtree(cls._tmpdir, ignore_errors=True)
            cls._tmpdir = ""

    def setUp(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        torch.cuda.tunable._clear_all()
        torch.cuda.tunable.wildcard_fallback_enable(self._wildcard_fallback_in_setup)

    def tearDown(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        torch.cuda.tunable.wildcard_fallback_enable(False)


class DynamicTunableOpsTest(_TunableOpGpuTestBase):
    """Verification of the full tuning enable/disable x dynamic-dim matrix.

    Each test clears the process-global TuningResultsManager in setUp so
    concrete and wildcard entries cannot leak between tests.
    """

    _tmpdir_prefix: str = "dynamic_tunable_ops_test_"

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
        out = _run_without_tunable_fallback_warning(
            self, lambda: torch.addmm(bias_x, mat1_x, mat2_x)
        )

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
        out = _run_without_tunable_fallback_warning(
            self, lambda: torch.mm(mat1_x, mat2_x)
        )

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
        out = _run_without_tunable_fallback_warning(self, lambda: torch.bmm(b1_x, b2_x))

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
        out = _run_without_tunable_fallback_warning(
            self, lambda: torch.baddbmm(bias_x, b1_x, b2_x)
        )

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


class _LayoutCoverageWildcardTestBase(_TunableOpGpuTestBase):
    """Shared tune-then-fallback round trip behind the mm and bmm layout
    matrices. Holds no `test_*` methods of its own, so the loader finds
    nothing to run here.

    Concrete classes supply the op hooks (`_op_substr`, `_make_operands`,
    `_run_op`) and their own parametrized entry point."""

    # Set by the concrete class.
    _op_substr: str = ""
    _TUNE_SEED_BASE: int = 0
    _TEST_SEED_BASE: int = 0

    # Order fixes the seed-offset numbering below; do not reorder.
    _DYNAMIC_DIMS: tuple[str, ...] = ("M", "K", "N")

    @classmethod
    def _seed_offset(cls, transa: bool, transb: bool, dynamic: str) -> int:
        """Reproduce the offsets these cases used to hard-code (0, 4, ... 44).
        `base_m` is derived from it, so it must stay injective over
        (transa, transb, dynamic) to keep every variant on its own shape."""
        return 4 * (
            cls._DYNAMIC_DIMS.index(dynamic) * 4 + int(transa) * 2 + int(transb)
        )

    def _make_operands(
        self, m: int, n: int, k: int, transa: bool, transb: bool, seed: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _run_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _round_trip_for_layout(self, transa: bool, transb: bool, dynamic: str) -> None:
        """Run the tune-then-fallback round trip for one layout, taking the
        `dynamic` dim (one of "M"/"N"/"K") as symbolic. Tuning and runtime
        shapes differ only in that dim."""
        seed_offset = self._seed_offset(transa, transb, dynamic)
        base_m, n, k = 41 + seed_offset, 257, 251
        delta = 12
        tuned = {"m": base_m, "n": n, "k": k}
        test = dict(tuned)
        test[dynamic.lower()] += delta

        a_t, b_t = self._make_operands(
            tuned["m"],
            tuned["n"],
            tuned["k"],
            transa,
            transb,
            seed=self._TUNE_SEED_BASE + seed_offset,
        )
        a_x, b_x = self._make_operands(
            test["m"],
            test["n"],
            test["k"],
            transa,
            transb,
            seed=self._TEST_SEED_BASE + seed_offset,
        )

        # Reference (tunable disabled).
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        ref = self._run_op(a_x, b_x)

        # Phase A: tune at the tuned shape with the chosen dim dynamic ->
        # wildcard persisted.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        with torch.cuda.tunable.dynamic_dims_mask(**{dynamic: True}):
            self._run_op(a_t, b_t)
        # Assert on the dims that are NOT dynamic for this variant. The
        # dynamic one is precisely what gets persisted as `*`, so asserting on
        # it could never match. Pinning the two static dims is strictly
        # stronger than a bare op-wide presence check: if the implementation
        # wildcarded the wrong dim, the dim expected to stay concrete goes
        # missing and this fires. `setUp` clears the results manager, so no
        # sibling variant's entries are present to satisfy it by accident.
        static_dims = [v for key, v in tuned.items() if key != dynamic.lower()]
        self.assertTrue(
            _has_wildcard_with_dims(self._op_substr, *static_dims),
            f"expected {self._op_substr} wildcard entry with concrete dims "
            f"{static_dims} after tuning layout (transa={transa}, "
            f"transb={transb}, dynamic={dynamic})",
        )
        # White-box: the persisted wildcard must wildcard the correct
        # leading dims for its transpose flags. This is what catches an
        # inverted lda/ldb/ldc -> dim mapping (e.g. a broken UsesMForLda),
        # which a mere presence check silently tolerates because a
        # mis-wildcarded key still contains a '*'.
        _assert_ld_wildcarding_consistent(self, self._op_substr)

        # Phase C: runtime, no mask, shape differs only in the dynamic dim
        # -> expected to dispatch via the wildcard fallback. Gated on the
        # observable proxies (no new concrete entry, output matches the
        # tunable-disabled reference); the persistence-side white-box check
        # above is the real gate on the wildcarding logic.
        before = len(_get_tunable_results())
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        out = self._run_op(a_x, b_x)
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
            atol=GEMM_ATOL,
            rtol=GEMM_RTOL,
            msg=f"layout (transa={transa}, transb={transb}, dynamic={dynamic}): "
            f"wildcard-fallback dispatch output must match tunable-disabled "
            f"reference",
        )


@instantiate_parametrized_tests
class LayoutCoverageWildcardTest(_LayoutCoverageWildcardTestBase):
    """The dynamic-mask + wildcard-fallback contract across all four
    (transa, transb) layouts, with M/N/K dynamic in turn.

    The K-dynamic cases are what cover lda: mm's row-major dispatch swaps
    inductor (M, N) into BLAS (n, m), so an M-dynamic call lands on BLAS-n and
    never touches lda, while K is not swapped and forces lda to be wildcarded
    exactly when transa == 'T'.

    N-dynamic is the mirror case: the same swap lands the dynamic bit on
    BLAS-m and forces ldc (and lda for transa == 'N') to wildcard."""

    _tmpdir_prefix: str = "layout_coverage_test_"
    _op_substr: str = "GemmTunableOp"
    _TUNE_SEED_BASE: int = 100
    _TEST_SEED_BASE: int = 200

    def _make_operands(
        self, m: int, n: int, k: int, transa: bool, transb: bool, seed: int
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

    def _run_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.mm(a, b)

    # Defined here rather than on the base on purpose:
    # instantiate_parametrized_tests walks dir(cls) -- which sees inherited
    # attributes -- and then delattr()s the generic method, which raises
    # AttributeError for anything that lives on a base class instead of in
    # this class's own __dict__.
    @parametrize("transa", [False, True])
    @parametrize("transb", [False, True])
    @parametrize("dynamic", ["M", "K", "N"])
    def test_layout(self, transa: bool, transb: bool, dynamic: str) -> None:
        self._round_trip_for_layout(transa, transb, dynamic)


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


@instantiate_parametrized_tests
class BmmLayoutCoverageWildcardTest(_LayoutCoverageWildcardTestBase):
    """Batched analog of LayoutCoverageWildcardTest. bmm derives its
    transpose flags and M<->N remap in baddbmm_out_cuda_impl, a separate path
    from the mm/addmm launchers.

    Shapes are kept disjoint from the mm layout tests via a distinct op
    (GemmStridedBatchedTunableOp) and seed range; see the module-level
    isolation note (the in-memory results table is process-global)."""

    _tmpdir_prefix: str = "bmm_layout_coverage_test_"
    _op_substr: str = "GemmStridedBatchedTunableOp"
    _TUNE_SEED_BASE: int = 300
    _TEST_SEED_BASE: int = 400
    _BATCH: int = 8

    def _make_operands(
        self, m: int, n: int, k: int, transa: bool, transb: bool, seed: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (batch1, batch2) for `bmm(batch1, batch2)` so the batched
        BLAS dispatch picks the requested (transa, transb) layout. As in the
        mm helper, a transposed operand is allocated in the swapped shape and
        then transposed on its last two dims so it is non-contiguous without a
        contiguous() copy."""
        torch.manual_seed(seed)
        b = self._BATCH
        if not transa:
            batch1 = torch.randn(b, m, k, dtype=DTYPE, device=DEVICE)
        else:
            batch1 = torch.randn(b, k, m, dtype=DTYPE, device=DEVICE).transpose(-2, -1)
        if not transb:
            batch2 = torch.randn(b, k, n, dtype=DTYPE, device=DEVICE)
        else:
            batch2 = torch.randn(b, n, k, dtype=DTYPE, device=DEVICE).transpose(-2, -1)
        return batch1, batch2

    def _run_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.bmm(a, b)

    # Must live in this class's own __dict__ -- see the note on the mm
    # counterpart.
    @parametrize("transa", [False, True])
    @parametrize("transb", [False, True])
    @parametrize("dynamic", ["M", "K", "N"])
    def test_bmm_layout(self, transa: bool, transb: bool, dynamic: str) -> None:
        self._round_trip_for_layout(transa, transb, dynamic)


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
class ScaledGemmTunableOpFP8Test(_TunableOpGpuTestBase):
    """Verify the dynamic-mask + wildcard-fallback contract for
    `torch._scaled_mm` with tensorwise FP8 scaling, matching the
    `node_replacement_dict` setup that triggers
    `ScaledGemmTunableOp` for large Linear layers.

    Support is established once in `setUpClass`, so every assertion below is
    a hard assertion: past that gate a failure is a regression, not an
    unsupported configuration."""

    _tmpdir_prefix: str = "scaled_gemm_test_"

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        reason = cls._scaled_mm_unsupported_reason()
        if reason is not None:
            cls._release_class_state()
            raise unittest.SkipTest(reason)

    @classmethod
    def _scaled_mm_unsupported_reason(cls) -> str | None:
        """Why FP8 `_scaled_mm` is unusable here, or None if it works.

        Establishing support once, up front, is what lets the tests hard-assert:
        catching RuntimeError around the call under test made a genuine
        regression indistinguishable from an unsupported arch and silently
        turned it into a skip. Capability rejection
        (HIPBLAS_STATUS_NOT_SUPPORTED) is arch/dtype-level rather than
        shape-level, so one small aligned GEMM is representative of the larger
        shapes the tests actually use."""
        if not hasattr(torch, "float8_e4m3fn"):
            return "torch.float8_e4m3fn not available"
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        a, b, scale_a, scale_b = cls._scaled_mm_inputs(256, 256, 256, seed=29)
        try:
            torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.bfloat16)
        except RuntimeError as e:
            return f"_scaled_mm not supported in this configuration: {e}"
        return None

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
        ref = self._run_scaled_mm(a_x, b_x, sa_x, sb_x)

        # Phase A: tune at m_tuned with M dynamic.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        with torch.cuda.tunable.dynamic_dims_mask(M=True):
            self._run_scaled_mm(a_t, b_t, sa_t, sb_t)

        scaled_entries = [
            e for e in _get_tunable_results() if "ScaledGemmTunableOp" in e[0]
        ]
        # "_scaled_mm stopped reaching the tunable path" is the primary
        # regression this test exists to catch, so it is an assertion rather
        # than the skip it used to be.
        self.assertGreaterEqual(
            len(scaled_entries),
            1,
            "tuning _scaled_mm with M dynamic must persist at least one "
            "ScaledGemmTunableOp entry; none present means the dispatch no "
            "longer reaches ScaledGemmTunableOp::operator()",
        )
        wildcard_present = any("*" in e[1] for e in scaled_entries)
        self.assertTrue(
            wildcard_present,
            f"expected ScaledGemmTunableOp wildcard entry after tuning "
            f"with M dynamic; got entries: {scaled_entries}",
        )

        # Phase C: runtime, no mask, different M -> wildcard fallback
        # and pass backend compatibility validation.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        out = _run_without_tunable_fallback_warning(
            self, lambda: self._run_scaled_mm(a_x, b_x, sa_x, sb_x)
        )

        self.assertEqual(
            out,
            ref,
            atol=FP8_GEMM_ATOL,
            rtol=FP8_GEMM_RTOL,
            msg="ScaledGemmTunableOp dispatch output should match "
            "tunable-disabled reference via wildcard fallback",
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
        ref = self._run_scaled_mm(a, b, sa, sb)

        # No entries primed for this shape. Tunable enabled, tuning disabled:
        # concrete miss + wildcard miss must fall back safely.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(False)
        out = self._run_scaled_mm(a, b, sa, sb)  # must NOT crash / Default

        self.assertEqual(
            out,
            ref,
            atol=FP8_GEMM_ATOL,
            rtol=FP8_GEMM_RTOL,
            msg="scaled both-miss fallback output should match "
            "tunable-disabled reference",
        )


# --- Legacy "non-dynamic" behavior (BEFORE the wildcard feature) --------
# These tests assert what TunableOp does when no `dynamic_dims_mask`
# is ever pushed: only concrete-key entries get persisted at
# compile-time tuning, and runtime concrete-miss queries fall through
# to the non-tunable aten path.
#
# This is the AOTI-runtime path -- a caller that never pushes a mask --
# NOT the inductor kill-switch. Those are different mechanisms:
# `cuda.autotune_tunableop_dynamic_dims_wildcard=False` makes
# `MMKernelInputs.dynamic_dim_mask` return all-False during lowering
# (kernel_inputs.py), which these tests never reach because they call
# torch.mm / torch.addmm directly. The kill-switch itself is covered by
# `DynamicDimMaskOperandSelectionTest.test_feature_flag_off_returns_all_false`.


class LegacyConcreteOnlyTunableOpsTest(_TunableOpGpuTestBase):
    """Pre-feature behavior: with no `dynamic_dims_mask` pushed, only concrete
    entries persist and a runtime concrete miss falls through safely."""

    _tmpdir_prefix: str = "legacy_concrete_only_test_"
    _wildcard_fallback_in_setup: bool = False

    def test_addmm_tuning_no_mask_persists_concrete_only(self) -> None:
        """Tuning enabled, NO `dynamic_dims_mask` context.
        addmm produces a single concrete entry; no wildcard."""
        m, n, k = 89, 1031, 257
        bias, mat1, mat2 = _addmm(m, n, k, seed=40)

        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        torch.addmm(bias, mat1, mat2)  # NO mask context

        self._assert_no_wildcards("GemmAndBiasTunableOp")
        self.assertTrue(
            _has_concrete_entry("GemmAndBiasTunableOp", m, n, k),
            f"expected concrete entry covering ({m},{n},{k}) after tuning",
        )

    def test_mm_tuning_no_mask_persists_concrete_only(self) -> None:
        """torch.mm tuning without mask: only GemmTunableOp concrete."""
        m, n, k = 91, 1033, 263
        mat1, mat2 = _mm(m, n, k, seed=41)

        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        torch.mm(mat1, mat2)  # NO mask

        self._assert_no_wildcards("GemmTunableOp")
        self.assertTrue(
            _has_concrete_entry("GemmTunableOp", m, n, k),
            f"expected concrete entry covering ({m},{n},{k}) after tuning",
        )

    def test_bmm_tuning_no_mask_persists_concrete_only(self) -> None:
        """torch.bmm tuning without mask: only StridedBatched concrete."""
        b, m, n, k = 16, 47, 257, 251
        b1, b2 = _bmm(b, m, n, k, seed=42)

        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        torch.bmm(b1, b2)  # NO mask

        self._assert_no_wildcards("GemmStridedBatchedTunableOp")
        self.assertTrue(
            _has_concrete_entry("GemmStridedBatchedTunableOp", m, n, k),
            f"expected concrete entry covering ({m},{n},{k}) after tuning",
        )

    def _assert_no_wildcards(self, op_substr: str) -> None:
        """No persisted entry for `op_substr` may carry a wildcard token.

        Paired with a positive `_has_concrete_entry` assertion at every call
        site: on its own this passes vacuously when tuning persists nothing,
        which is the regression it is meant to catch."""
        for op_sig, params_sig, _, _ in _entries_for(op_substr):
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
            atol=GEMM_ATOL,
            rtol=GEMM_RTOL,
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
            atol=GEMM_ATOL,
            rtol=GEMM_RTOL,
            msg="concrete-hit dispatch result should match reference",
        )


# --- PYTORCH_TUNABLEOP_RECORD_UNTUNED collection on a concrete miss ---
# Regression coverage for the runtime record-untuned path across every GEMM
# category. TunableOp::operator() records the concrete shape on every miss,
# before consulting the wildcard entries, so the offline-tuning workflow sees
# the shape whether it ends up served by a wildcard or by the non-tunable aten
# fallback. Recording only on a total miss would silently drop exactly the
# shapes a wildcard is approximating.


class RecordUntunedConcreteMissTest(_TunableOpGpuTestBase):
    """PYTORCH_TUNABLEOP_RECORD_UNTUNED collection on a runtime concrete miss,
    for every GEMM category, whether or not a wildcard then serves the call. A
    concrete hit records nothing.

    Untuned output is redirected per test via
    PYTORCH_TUNABLEOP_UNTUNED_FILENAME; record_untuned_enable(False) flushes
    and closes that file and clears the C++ dedup set."""

    _tmpdir_prefix: str = "record_untuned_test_"

    def setUp(self) -> None:
        torch.cuda.tunable.record_untuned_enable(False)
        super().setUp()

    def tearDown(self) -> None:
        torch.cuda.tunable.record_untuned_enable(False)
        super().tearDown()

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
        # Same up-front support gate ScaledGemmTunableOpFP8Test uses, so an
        # unsupported arch skips on an explicit precondition rather than on a
        # RuntimeError swallowed from the call under test.
        reason = ScaledGemmTunableOpFP8Test._scaled_mm_unsupported_reason()
        if reason is not None:
            raise unittest.SkipTest(reason)
        # Use distinct aligned N and K to verify the helper's K-by-N layout.
        m, n, k = 2560, 1408, 1536
        a, b, sa, sb = ScaledGemmTunableOpFP8Test._scaled_mm_inputs(m, n, k, seed=74)

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

    @staticmethod
    def _sym(name: str) -> sympy.Symbol:
        return sympy.Symbol(name, positive=True, integer=True)

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

    # -- Both halves of dyn_k = _dyn(k1) or _dyn(k2) ---------------------
    # k1 is mat1's trailing dim, k2 is mat2's leading one; a mask derived from
    # only one of them still passes every fixture that has K static on both.

    def test_dynamic_k_on_mat1_only(self) -> None:
        s_k = self._sym("s_k")
        ki = MMKernelInputs([_SizeOnlyNode((128, s_k)), _SizeOnlyNode((512, 256))])
        self.assertEqual(ki.dynamic_dim_mask("mm"), (False, False, True, False))

    def test_dynamic_k_on_mat2_only(self) -> None:
        s_k = self._sym("s_k")
        ki = MMKernelInputs([_SizeOnlyNode((128, 512)), _SizeOnlyNode((s_k, 256))])
        self.assertEqual(ki.dynamic_dim_mask("mm"), (False, False, True, False))

    def test_addmm_default_indices_read_the_matmul_operands(self) -> None:
        """addmm's inputs are [bias, mat1, mat2], so the default
        mat1_idx=-2 / mat2_idx=-1 must land on mat1/mat2 and skip the bias."""
        s_m = self._sym("s_m")
        nodes: list[object] = [
            _SizeOnlyNode((256,)),  # bias
            _SizeOnlyNode((s_m, 512)),  # (M_dyn, K)
            _SizeOnlyNode((512, 256)),  # (K, N)
        ]
        ki = MMKernelInputs(nodes)
        self.assertEqual(ki.dynamic_dim_mask("addmm"), (True, False, False, False))

    # -- Batched branch -------------------------------------------------
    # The bmm/baddbmm branch was never entered, so the fourth mask element
    # (dyn_batch) had no coverage at all despite BmmLayoutCoverageWildcardTest
    # depending on batched wildcarding.

    @staticmethod
    def _bmm_inputs(
        b1: object, m: object, k1: object, b2: object, k2: object, n: object
    ) -> MMKernelInputs:
        return MMKernelInputs([_SizeOnlyNode((b1, m, k1)), _SizeOnlyNode((b2, k2, n))])

    def test_bmm_dynamic_batch_on_mat1(self) -> None:
        ki = self._bmm_inputs(self._sym("s_b"), 128, 512, 8, 512, 256)
        self.assertEqual(ki.dynamic_dim_mask("bmm"), (False, False, False, True))

    def test_bmm_dynamic_batch_on_mat2(self) -> None:
        ki = self._bmm_inputs(8, 128, 512, self._sym("s_b"), 512, 256)
        self.assertEqual(ki.dynamic_dim_mask("bmm"), (False, False, False, True))

    def test_bmm_dynamic_m_and_n(self) -> None:
        ki = self._bmm_inputs(8, self._sym("s_m"), 512, 8, 512, self._sym("s_n"))
        self.assertEqual(ki.dynamic_dim_mask("bmm"), (True, True, False, False))

    def test_baddbmm_default_indices_read_the_matmul_operands(self) -> None:
        """baddbmm's inputs are [bias, batch1, batch2] and the bias is itself
        3D, so reading the wrong operands is not caught by rank alone."""
        nodes: list[object] = [
            _SizeOnlyNode((8, 128, 256)),  # bias
            _SizeOnlyNode((8, 128, 512)),  # (B, M, K)
            _SizeOnlyNode((8, self._sym("s_k"), 256)),  # (B, K_dyn, N)
        ]
        ki = MMKernelInputs(nodes)
        self.assertEqual(ki.dynamic_dim_mask("baddbmm"), (False, False, True, False))

    def test_feature_flag_off_returns_all_false(self) -> None:
        with inductor_config.patch(
            {"cuda.autotune_tunableop_dynamic_dims_wildcard": False}
        ):
            ki = self._scaled_mm_inputs(with_bias=False)
            self.assertEqual(
                ki.dynamic_dim_mask("scaled_mm"), (False, False, False, False)
            )


class WildcardFallbackGateTest(TestCase):
    """Coverage for the `wildcard_fallback_enable` setter, which is all that
    is observable -- see the module-level observability caveat.

    Opt-in behavior lives in `LegacyConcreteOnlyTunableOpsTest`. The
    off-by-default initial value cannot be asserted in-process: earlier suites
    have already written the process-global flag, and the getter
    short-circuits on a cached PYTORCH_TUNABLEOP_WILDCARD_FALLBACK=1."""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("cuda not available")

    def tearDown(self) -> None:
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
        torch.cuda.tunable.wildcard_fallback_enable(False)

    @unittest.skipIf(
        os.environ.get("PYTORCH_TUNABLEOP_WILDCARD_FALLBACK") == "1",
        "env var forces IsWildcardFallbackEnabled() true, so the off half "
        "of the round trip cannot be observed",
    )
    def test_gate_round_trips(self) -> None:
        """`wildcard_fallback_enable` is the flag `TunableOp::operator()`
        reads before consulting `LookupWildcardFallback`, so the setter and
        getter must agree in both directions."""
        torch.cuda.tunable.wildcard_fallback_enable(True)
        self.assertTrue(torch.cuda.tunable.wildcard_fallback_is_enabled())

        torch.cuda.tunable.wildcard_fallback_enable(False)
        self.assertFalse(torch.cuda.tunable.wildcard_fallback_is_enabled())


class TestDynamicDimsMaskAPI(TestCase):
    """Tests for the push/pop dynamic_dims_mask API and context manager."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_context_manager_push_pop(self) -> None:
        with torch.cuda.tunable.dynamic_dims_mask(M=True, K=True):
            mask = torch.cuda.tunable._pack_dynamic_dims_mask(
                M=True, N=False, K=True, BATCH=False
            )
            self.assertNotEqual(mask, 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_nested_context_managers(self) -> None:
        with torch.cuda.tunable.dynamic_dims_mask(M=True):
            with torch.cuda.tunable.dynamic_dims_mask(N=True):
                pass

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_double_pop_raises(self) -> None:
        handle = torch._C._cuda_tunableop_push_dynamic_dims_mask(0x1)
        torch._C._cuda_tunableop_pop_dynamic_dims_mask(handle)
        with self.assertRaises(RuntimeError):
            torch._C._cuda_tunableop_pop_dynamic_dims_mask(handle)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_invalid_mask_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            torch._C._cuda_tunableop_push_dynamic_dims_mask(0x10)


class ClearAllTest(_TunableOpGpuTestBase):
    """Coverage for the testing-only `torch.cuda.tunable._clear_all()`.

    Each test starts with an empty in-memory TuningResultsManager.
    """

    _tmpdir_prefix: str = "clear_all_test_"

    def test_clear_all_drops_concrete_and_wildcard_entries(self) -> None:
        m, n, k = 61, 2069, 1031
        bias, mat1, mat2 = _addmm(m, n, k, seed=91)

        # Tune with M dynamic so both a concrete and a wildcard entry land.
        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        with torch.cuda.tunable.dynamic_dims_mask(M=True):
            torch.addmm(bias, mat1, mat2)

        self.assertTrue(
            _has_concrete_entry("GemmAndBiasTunableOp", m, n, k),
            "expected a concrete entry after tuning",
        )
        self.assertTrue(
            _has_wildcard_with_dims("GemmAndBiasTunableOp", n, k),
            "expected a wildcard entry after tuning with M dynamic",
        )

        torch.cuda.tunable._clear_all()

        self.assertTrue(
            torch.cuda.tunable.wildcard_fallback_is_enabled(),
            "_clear_all() must preserve wildcard fallback enablement",
        )
        self.assertEqual(
            torch.cuda.tunable.get_results(),
            (),
            "_clear_all() must leave the in-memory results empty",
        )
        self.assertFalse(
            _has_concrete_entry("GemmAndBiasTunableOp", m, n, k),
            "_clear_all() must drop the concrete entry",
        )
        self.assertFalse(
            _has_wildcard_with_dims("GemmAndBiasTunableOp", n, k),
            "_clear_all() must drop the wildcard entry",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm required")
    def test_shape_can_be_retuned_after_clear_all(self) -> None:
        """The point of the reset: the same shape is tunable again afterwards,
        so tests no longer have to pick globally disjoint (m, n, k)."""
        m, n, k = 67, 2081, 1033
        bias, mat1, mat2 = _addmm(m, n, k, seed=92)

        torch.cuda.tunable.enable(True)
        torch.cuda.tunable.tuning_enable(True)
        torch.addmm(bias, mat1, mat2)
        self.assertTrue(_has_concrete_entry("GemmAndBiasTunableOp", m, n, k))

        torch.cuda.tunable._clear_all()
        self.assertFalse(_has_concrete_entry("GemmAndBiasTunableOp", m, n, k))

        # Re-tuning the very same shape repopulates the entry.
        torch.addmm(bias, mat1, mat2)
        self.assertTrue(
            _has_concrete_entry("GemmAndBiasTunableOp", m, n, k),
            "the same shape must be tunable again after _clear_all()",
        )


if __name__ == "__main__":
    run_tests()
