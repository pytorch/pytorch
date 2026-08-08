#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <c10/util/complex.h>

using namespace at;

// Regression test for pytorch/pytorch#162358. The CPU unpacker used to trust
// LAPACK's classification of eigenvalues and read the "partner" column of the
// packed real-eigenvector output. If LAPACK reported a complex eigenvalue at
// the trailing index (j == n-1), the partner column j+1 == n is outside the
// n-column buffer, causing a heap-buffer-overflow. The fix replaces the silent
// OOB read with a TORCH_CHECK. This test constructs the exact structurally
// inconsistent input directly, bypassing LAPACK, so it exercises the fixed
// branch deterministically on every CI configuration regardless of BLAS
// backend.
TEST(LinalgEigTest, TrailingComplexEigenvalueRaises) {
  const int64_t n = 2;

  auto values = at::zeros({n}, at::kComplexDouble);
  values[0] = c10::complex<double>(1.0, 0.0);
  // Trailing index carries a nonzero imag with no partner column in VR.
  values[1] = c10::complex<double>(2.0, 0.5);

  auto vectors = at::eye(n, at::kDouble);
  auto result = at::empty({n, n}, at::kComplexDouble);

  EXPECT_THROW(
      at::native::linalg_eig_make_complex_eigenvectors(result, values, vectors),
      c10::Error);
}

// Well-formed input still works: real eigenvalues, matching real vectors.
TEST(LinalgEigTest, RealEigenvaluesUnpackCorrectly) {
  const int64_t n = 2;

  auto values = at::zeros({n}, at::kComplexDouble);
  values[0] = c10::complex<double>(1.0, 0.0);
  values[1] = c10::complex<double>(2.0, 0.0);

  auto vectors = at::eye(n, at::kDouble);
  auto result = at::empty({n, n}, at::kComplexDouble);

  at::native::linalg_eig_make_complex_eigenvectors(result, values, vectors);

  auto expected = at::eye(n, at::kComplexDouble);
  ASSERT_TRUE(result.equal(expected));
}
