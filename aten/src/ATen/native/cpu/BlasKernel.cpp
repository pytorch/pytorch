#include <ATen/Dispatch.h>
#include <ATen/native/CPUBlas.h>

namespace at {
namespace native {
namespace cpublas {
namespace {

template <typename scalar_t>
void scale_(int64_t m, int64_t n, scalar_t alpha, scalar_t *a, int64_t lda) {
  if (alpha == scalar_t(1)) {
    return;  // identity
  }

  if (alpha == scalar_t(0)) {
    for (int64_t j = 0; j < n; j++) {
      for (int64_t i = 0; i < m; i++) {
        a[j * lda + i] = scalar_t(0);
      }
    }
    return;
  }

  for (int64_t j = 0; j < n; j++) {
    for (int64_t i = 0; i < m; i++) {
      a[j * lda + i] *= alpha;
    }
  }
}


template <typename scalar_t>
void gemm_notrans_(
    int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    scalar_t beta,
    scalar_t *c, int64_t ldc) {
  // c *= beta
  scale_(m, n, beta, c, ldc);

  // c += alpha * (a @ b)
  for (int64_t l = 0; l < k; l++) {
    for (int64_t j = 0; j < n; j++) {
      scalar_t val = b[l + j * ldb] * alpha;
      int64_t i_m = m / 4;
      for (int64_t i_i = 0; i_i < i_m; i_i++) {
        c[j * ldc + i_i * 4 + 0] += a[i_i * 4 + 0 + l * lda] * val;
        c[j * ldc + i_i * 4 + 1] += a[i_i * 4 + 1 + l * lda] * val;
        c[j * ldc + i_i * 4 + 2] += a[i_i * 4 + 2 + l * lda] * val;
        c[j * ldc + i_i * 4 + 3] += a[i_i * 4 + 3 + l * lda] * val;
      }
      int64_t i = i_m * 4;
      for (; i < m; i++)
        c[j * ldc + i] += a[i + l * lda] * val;
    }
  }
}

template <typename scalar_t>
void gemm_transa_(
    int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    scalar_t beta,
    scalar_t *c, int64_t ldc) {
  // c = alpha * (a.T @ b) + beta * c
  const scalar_t *a_ = a;
  for (int64_t i = 0; i < m; i++)
  {
    const scalar_t *b_ = b;
    for (int64_t j = 0; j < n; j++)
    {
      scalar_t sum = 0;
      for(int64_t l = 0; l < k; l++)
        sum += a_[l]*b_[l];
      b_ += ldb;
      if (beta == scalar_t(0))
        c[j*ldc+i] = alpha*sum;
      else
        c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
    }
    a_ += lda;
  }
}

template <typename scalar_t>
void gemm_transb_(
    int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    scalar_t beta,
    scalar_t *c, int64_t ldc) {
  // c *= beta
  scale_(m, n, beta, c, ldc);

  // c += alpha * (a @ b.T)
  for (int64_t l = 0; l < k; l++) {
    for (int64_t j = 0; j < n; j++) {
      scalar_t val = b[j + l * ldb] * alpha;
      int64_t i_m = m / 4;
      for (int64_t i_i = 0; i_i < i_m; i_i++) {
        c[j * ldc + i_i * 4 + 0] += a[i_i * 4 + 0 + l * lda] * val;
        c[j * ldc + i_i * 4 + 1] += a[i_i * 4 + 1 + l * lda] * val;
        c[j * ldc + i_i * 4 + 2] += a[i_i * 4 + 2 + l * lda] * val;
        c[j * ldc + i_i * 4 + 3] += a[i_i * 4 + 3 + l * lda] * val;
      }
      int64_t i = i_m * 4;
      for (; i < m; i++)
        c[j * ldc + i] += a[i + l * lda] * val;
    }
  }
}

template <typename scalar_t>
void gemm_transab_(
    int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    scalar_t beta,
    scalar_t *c, int64_t ldc) {
  // c *= beta
  scale_(m, n, beta, c, ldc);

  // c += alpha * (a.T @ b.T)
  for (int64_t i = 0; i < m; i++) {
    for (int64_t j = 0; j < n; j++) {
      int64_t l_k = k / 4;
      for (int64_t l_l = 0; l_l < l_k; l_l++) {
        c[j * ldc + i] += a[i * lda + l_l * 4 + 0] //
          * b[(l_l * 4 + 0) * ldb + j] * alpha;
        c[j * ldc + i] += a[i * lda + l_l * 4 + 1] //
          * b[(l_l * 4 + 1) * ldb + j] * alpha;
        c[j * ldc + i] += a[i * lda + l_l * 4 + 2] //
          * b[(l_l * 4 + 2) * ldb + j] * alpha;
        c[j * ldc + i] += a[i * lda + l_l * 4 + 3] //
          * b[(l_l * 4 + 3) * ldb + j] * alpha;
      }
      int64_t l = l_k * 4;
      for (; l < k; l++)
        c[j * ldc + i] += a[i * lda + l] * b[l * ldb + j] * alpha;
    }
  }
}

template <typename scalar_t>
void gemm_core_(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    scalar_t beta,
    scalar_t *c, int64_t ldc) {
  if(transa == NoTranspose && transb == NoTranspose) {
    return gemm_notrans_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else if(transa == Transpose && transb != Transpose) {
    gemm_transa_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else if(transa == NoTranspose && transb == Transpose) {
    gemm_transb_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else {  // transa == Transpose && transb == Transpose
    gemm_transab_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}

void cpublas_gemm_impl(
    at::ScalarType type,
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const Scalar& alpha,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const Scalar& beta,
    void *c, int64_t ldc) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(at::kHalf, at::kBFloat16,
    type, "cpublas_gemm_impl",
      [&]{
        gemm_core_(
            transa, transb, m, n, k,
            alpha.to<scalar_t>(),
            static_cast<const scalar_t *>(a), lda,
            static_cast<const scalar_t *>(b), ldb,
            beta.to<scalar_t>(),
            static_cast<scalar_t *>(c), ldc);
      });
}

void cpublas_axpy_impl(at::ScalarType type, int64_t n, const Scalar& _a, const void *_x, int64_t incx, void *_y, int64_t incy){
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(type, "cpublas_axpy_impl",
    [&] {
      auto a = _a.to<scalar_t>();
      auto x = static_cast<const scalar_t *>(_x);
      auto y = static_cast<scalar_t *>(_y);
      int64_t i;
      for(i = 0; i < n; i++)
        y[i*incy] += a*x[i*incx];
    });
}

void cpublas_copy_impl(at::ScalarType type, int64_t n, const void *_x, int64_t incx, void *_y, int64_t incy){
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(type, "cpublas_copy_impl",
    [&] {
      auto x = static_cast<const scalar_t *>(_x);
      auto y = static_cast<scalar_t *>(_y);
      int64_t i;
      for(i = 0; i < n; i++)
        y[i*incy] = x[i*incx];
    });
}

}}  // namespace cpublas::(anonymous)


// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(cpublas::gemm_stub, &cpublas::cpublas_gemm_impl);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(cpublas::axpy_stub, &cpublas::cpublas_axpy_impl);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(cpublas::copy_stub, &cpublas::cpublas_copy_impl);

}}  // namespace at::native
