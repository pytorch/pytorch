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
"""

# pyre-strict

import os
import shutil
import tempfile
import unittest
from typing import cast, TypeAlias

import torch
import torch.cuda.tunable
from torch.testing._internal.common_utils import run_tests, TestCase


DEVICE: str = "cuda"
DTYPE: torch.dtype = torch.bfloat16
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
    assert ld_idx != -1, f"no '{_LD_MARKER}' marker in signature: {params_sig}"  # noqa: S101
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
            atol=1e-2,
            rtol=1e-2,
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
            atol=1e-2,
            rtol=1e-2,
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
            atol=1e-2,
            rtol=1e-2,
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
            atol=1e-2,
            rtol=1e-2,
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
            atol=1e-2,
            rtol=1e-2,
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
            atol=1e-2,
            rtol=1e-2,
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
            atol=1e-2,
            rtol=1e-2,
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
            atol=1e-2,
            rtol=1e-2,
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
            atol=1e-2,
            rtol=1e-2,
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
            atol=1e-2,
            rtol=1e-2,
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
            atol=1e-2,
            rtol=1e-2,
            msg="baddbmm both-miss fallback output should match reference",
        )


if __name__ == "__main__":
    run_tests()
