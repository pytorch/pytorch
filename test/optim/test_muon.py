# Owner(s): ["module: optimizer"]

import torch
from torch.optim._muon import (
    _zeropower_via_newtonschulz,
    EPS,
    JORDAN_COEFFICIENTS,
    PE_COEFFICIENTS,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def _orthogonality_error(M):
    """||M @ M^T - I||_F measuring distance from orthogonality, computed in float32."""
    M = M.float()
    rows = min(M.size(0), M.size(1))
    eye = torch.eye(rows, device=M.device)
    if M.size(0) <= M.size(1):
        return (M @ M.T - eye).norm()
    else:
        return (M.T @ M - eye).norm()


def _normalize_input(M, normalization, eps=EPS):
    """Reproduce the normalization step of _zeropower_via_newtonschulz to get a fair baseline."""
    M = M.bfloat16()
    if M.size(0) > M.size(1):
        M = M.T
    if normalization == "fro":
        M = M / M.norm().clamp(min=eps)
    elif normalization == "schatten":
        gram = M @ M.T
        s = gram.norm().clamp(min=eps)
        M = M * s.rsqrt()
    elif normalization == "aol":
        gram = M @ M.T
        s_vec = torch.rsqrt(torch.clamp_min(gram.abs().sum(dim=-1), min=eps))
        M = M * s_vec.unsqueeze(-1)
    return M


class TestNewtonSchulz(TestCase):
    """Tests for _zeropower_via_newtonschulz covering coefficient presets,
    normalization types, matrix shapes, and error handling."""

    def _make_matrix(self, rows, cols, seed=42):
        torch.manual_seed(seed)
        return torch.randn(rows, cols)

    # ── Coefficient preset tests ──────────────────────────────────────

    def test_jordan_coefficients(self):
        M = self._make_matrix(64, 32)
        ns_coeffs = (JORDAN_COEFFICIENTS,)
        result = _zeropower_via_newtonschulz(M, ns_coeffs, ns_steps=5, eps=EPS)
        baseline = _normalize_input(M, "schatten")
        self.assertLess(
            _orthogonality_error(result).item(),
            _orthogonality_error(baseline).item(),
        )

    def test_polar_express_coefficients(self):
        M = self._make_matrix(64, 32)
        result = _zeropower_via_newtonschulz(M, PE_COEFFICIENTS, ns_steps=5, eps=EPS)
        baseline = _normalize_input(M, "schatten")
        self.assertLess(
            _orthogonality_error(result).item(),
            _orthogonality_error(baseline).item(),
        )

    def test_custom_single_coefficient(self):
        M = self._make_matrix(32, 64)
        custom = ((3.4445, -4.7750, 2.0315),)
        result = _zeropower_via_newtonschulz(M, custom, ns_steps=5, eps=EPS)
        baseline = _normalize_input(M, "schatten")
        self.assertLess(
            _orthogonality_error(result).item(),
            _orthogonality_error(baseline).item(),
        )

    def test_custom_per_step_coefficients(self):
        M = self._make_matrix(32, 32)
        per_step = (
            (4.0848, -6.8946, 2.9270),
            (3.9505, -6.3029, 2.6377),
            (3.7418, -5.5913, 2.3037),
            (2.8769, -3.1427, 1.2046),
            (2.8366, -3.0525, 1.2012),
        )
        result = _zeropower_via_newtonschulz(M, per_step, ns_steps=3, eps=EPS)
        baseline = _normalize_input(M, "schatten")
        self.assertLess(
            _orthogonality_error(result).item(),
            _orthogonality_error(baseline).item(),
        )

    def test_coefficients_repeat_when_fewer_than_steps(self):
        """When ns_coefficients has fewer entries than ns_steps, last entry repeats."""
        M = self._make_matrix(32, 64)
        single = (JORDAN_COEFFICIENTS,)
        result = _zeropower_via_newtonschulz(M, single, ns_steps=8, eps=EPS)
        baseline = _normalize_input(M, "schatten")
        self.assertLess(
            _orthogonality_error(result).item(),
            _orthogonality_error(baseline).item(),
        )

    # ── Normalization type tests ──────────────────────────────────────

    def test_normalization_fro(self):
        M = self._make_matrix(64, 32)
        result = _zeropower_via_newtonschulz(
            M, PE_COEFFICIENTS, ns_steps=5, eps=EPS, normalization="fro"
        )
        baseline = _normalize_input(M, "fro")
        self.assertLess(
            _orthogonality_error(result).item(),
            _orthogonality_error(baseline).item(),
        )

    def test_normalization_schatten(self):
        M = self._make_matrix(64, 32)
        result = _zeropower_via_newtonschulz(
            M, PE_COEFFICIENTS, ns_steps=5, eps=EPS, normalization="schatten"
        )
        baseline = _normalize_input(M, "schatten")
        self.assertLess(
            _orthogonality_error(result).item(),
            _orthogonality_error(baseline).item(),
        )

    def test_normalization_aol(self):
        M = self._make_matrix(64, 32)
        result = _zeropower_via_newtonschulz(
            M, PE_COEFFICIENTS, ns_steps=5, eps=EPS, normalization="aol"
        )
        baseline = _normalize_input(M, "aol")
        self.assertLess(
            _orthogonality_error(result).item(),
            _orthogonality_error(baseline).item(),
        )

    def test_all_normalizations_with_jordan(self):
        """All normalization types improve orthogonality with Jordan coefficients."""
        M = self._make_matrix(48, 32)
        ns_coeffs = (JORDAN_COEFFICIENTS,)
        for norm in ("fro", "schatten", "aol"):
            with self.subTest(normalization=norm):
                result = _zeropower_via_newtonschulz(
                    M, ns_coeffs, ns_steps=5, eps=EPS, normalization=norm
                )
                baseline = _normalize_input(M, norm)
                self.assertLess(
                    _orthogonality_error(result).item(),
                    _orthogonality_error(baseline).item(),
                )

    # ── Shape tests ───────────────────────────────────────────────────

    def test_wide_matrix(self):
        M = self._make_matrix(16, 64)
        result = _zeropower_via_newtonschulz(M, PE_COEFFICIENTS, ns_steps=5, eps=EPS)
        self.assertEqual(result.shape, M.shape)
        baseline = _normalize_input(M, "schatten")
        self.assertLess(
            _orthogonality_error(result).item(),
            _orthogonality_error(baseline).item(),
        )

    def test_tall_matrix(self):
        M = self._make_matrix(64, 16)
        result = _zeropower_via_newtonschulz(M, PE_COEFFICIENTS, ns_steps=5, eps=EPS)
        self.assertEqual(result.shape, M.shape)
        baseline = _normalize_input(M, "schatten")
        self.assertLess(
            _orthogonality_error(result).item(),
            _orthogonality_error(baseline).item(),
        )

    def test_square_matrix(self):
        M = self._make_matrix(32, 32)
        result = _zeropower_via_newtonschulz(M, PE_COEFFICIENTS, ns_steps=5, eps=EPS)
        self.assertEqual(result.shape, M.shape)
        baseline = _normalize_input(M, "schatten")
        self.assertLess(
            _orthogonality_error(result).item(),
            _orthogonality_error(baseline).item(),
        )

    # ── Iteration count tests ─────────────────────────────────────────

    def test_more_steps_more_orthogonal(self):
        """Increasing ns_steps should yield progressively lower orthogonality error."""
        M = self._make_matrix(32, 64)
        ns_coeffs = (JORDAN_COEFFICIENTS,)
        errors = []
        # note that we stop at 5 step because Jordan coefficients do not have convergence guarantees
        for steps in (1, 2, 3):
            result = _zeropower_via_newtonschulz(M, ns_coeffs, ns_steps=steps, eps=EPS)
            errors.append(_orthogonality_error(result).item())
        for i in range(len(errors) - 1):
            self.assertGreaterEqual(errors[i], errors[i + 1])

    def test_ns_steps_zero_only_normalizes(self):
        """With ns_steps=0, output should be the normalized input (no NS iterations)."""
        M = self._make_matrix(32, 64)
        result = _zeropower_via_newtonschulz(
            M, PE_COEFFICIENTS, ns_steps=0, eps=EPS, normalization="schatten"
        )
        expected = _normalize_input(M, "schatten")
        self.assertEqual(result.shape, M.shape)
        self.assertEqual(result.dtype, torch.bfloat16)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)

    def test_polar_express_more_steps_more_orthogonal(self):
        """PE coefficients with increasing steps yield better orthogonality."""
        M = self._make_matrix(48, 32)
        errors = []
        for steps in (1, 3, 5, 10):
            result = _zeropower_via_newtonschulz(
                M, PE_COEFFICIENTS, ns_steps=steps, eps=EPS
            )
            errors.append(_orthogonality_error(result).item())
        for i in range(len(errors) - 1):
            self.assertGreaterEqual(errors[i], errors[i + 1])

    # ── Error handling tests ──────────────────────────────────────────

    def test_invalid_ns_steps_raises(self):
        M = self._make_matrix(16, 16)
        with self.assertRaisesRegex(ValueError, "less than 100"):
            _zeropower_via_newtonschulz(
                M, (JORDAN_COEFFICIENTS,), ns_steps=100, eps=EPS
            )

    def test_non_2d_input_raises(self):
        M = torch.randn(4, 4, 4)
        with self.assertRaisesRegex(ValueError, "2D matrix"):
            _zeropower_via_newtonschulz(M, (JORDAN_COEFFICIENTS,), ns_steps=5, eps=EPS)

    def test_invalid_normalization_raises(self):
        M = self._make_matrix(16, 16)
        with self.assertRaisesRegex(ValueError, "Unsupported normalization"):
            _zeropower_via_newtonschulz(
                M, (JORDAN_COEFFICIENTS,), ns_steps=5, eps=EPS, normalization="bad"
            )

    def test_1d_input_raises(self):
        M = torch.randn(16)
        with self.assertRaisesRegex(ValueError, "2D matrix"):
            _zeropower_via_newtonschulz(M, (JORDAN_COEFFICIENTS,), ns_steps=5, eps=EPS)

    # ── Output property tests ─────────────────────────────────────────

    def test_output_dtype_is_bfloat16(self):
        M = self._make_matrix(32, 16)
        result = _zeropower_via_newtonschulz(M, PE_COEFFICIENTS, ns_steps=5, eps=EPS)
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_output_shape_preserved(self):
        for rows, cols in [(16, 64), (64, 16), (32, 32)]:
            with self.subTest(rows=rows, cols=cols):
                M = self._make_matrix(rows, cols)
                result = _zeropower_via_newtonschulz(
                    M, PE_COEFFICIENTS, ns_steps=5, eps=EPS
                )
                self.assertEqual(result.shape, (rows, cols))


if __name__ == "__main__":
    run_tests()
