#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/cpu/zmath.h>
#include <c10/util/irange.h>
#include <c10/util/Unroll.h>

#if defined(__aarch64__) && !defined(C10_MOBILE)
#include <arm_neon.h>

namespace at::native::blas_impl {
void fp16_gemv_notrans(
    const int m,
    const int n,
    const float alpha,
    const float16_t* a,
    const int lda,
    const float16_t* x,
    const int incx,
    const float beta,
    float16_t* y,
    const int incy);

void fp16_gemv_trans(
    const int m,
    const int n,
    const float alpha,
    const float16_t* a,
    const int lda,
    const float16_t* x,
    const int incx,
    const float beta,
    float16_t* y,
    const int incy);
}
#endif

namespace at::native {
namespace cpublas {
namespace {

template <typename scalar_t, typename opmath_t>
void scale_(int64_t m, int64_t n, opmath_t alpha, scalar_t *a, int64_t lda) {
  if (alpha == opmath_t(1)) {
    return;  // identity
  }

  if (alpha == opmath_t(0)) {
    for (const auto j : c10::irange(n)) {
      for (const auto i : c10::irange(m)) {
        a[j * lda + i] = scalar_t(0);
      }
    }
    return;
  }

  for (const auto j : c10::irange(n)) {
    for (const auto i : c10::irange(m)) {
      a[j * lda + i] *= alpha;
    }
  }
}

template <typename Func>
auto sum(int64_t N, Func f) {
  constexpr int ilp_factor = 4;
  using acc_t = decltype(f(0));

  // Calculate independent partial sums then add together at the end
  std::array<acc_t, ilp_factor> partial_sums{};

  int64_t i = 0;
  for (; i + ilp_factor <= N; i += ilp_factor) {
    c10::ForcedUnroll<ilp_factor>{}([&](int k) {
      partial_sums[k] += f(i + k);
    });
  }
  for (; i < N; ++i) {
    partial_sums[0] += f(i);
  }
  for (int k = 1; k < ilp_factor; ++k) {
    partial_sums[0] += partial_sums[k];
  }
  return partial_sums[0];
}

template <typename scalar_t, typename opmath_t>
typename std::enable_if<std::is_same<scalar_t, opmath_t>::value, void>::type
gemm_notrans_(
    int64_t m,
    int64_t n,
    int64_t k,
    opmath_t alpha,
    const scalar_t* a,
    int64_t lda,
    const scalar_t* b,
    int64_t ldb,
    opmath_t beta,
    scalar_t* c,
    int64_t ldc) {
  // c *= beta
  scale_(m, n, beta, c, ldc);

  // c += alpha * (a @ b)
  for (const auto l : c10::irange(k)) {
    for (const auto j : c10::irange(n)) {
      opmath_t val = b[l + j * ldb] * alpha;
      int64_t i_m = m / 4;
      for (const auto i_i : c10::irange(i_m)) {
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

// std::is_same<scalar_t, at::BFloat16> || std::is_same<scalar_t, at::Half>
template <typename scalar_t, typename opmath_t>
typename std::enable_if<!std::is_same<scalar_t, opmath_t>::value, void>::type
gemm_notrans_(
    int64_t m,
    int64_t n,
    int64_t k,
    opmath_t alpha,
    const scalar_t* a,
    int64_t lda,
    const scalar_t* b,
    int64_t ldb,
    opmath_t beta,
    scalar_t* c,
    int64_t ldc) {
  // c += alpha * (a @ b)
  for (const auto i : c10::irange(m)) {
    for (const auto j : c10::irange(n)) {
      const auto dot = sum(k, [&](int64_t l) -> opmath_t {
        return static_cast<opmath_t>(a[l * lda + i]) *
            static_cast<opmath_t>(b[j * ldb + l]);
      });
      if (beta == opmath_t(0)) {
        c[j * ldc + i] = alpha * dot;
      } else {
        c[j * ldc + i] = beta * c[j * ldc + i] + alpha * dot;
      }
    }
  }
}

template <typename scalar_t, typename opmath_t>
void gemm_transa_(
    TransposeType transa,
    int64_t m, int64_t n, int64_t k,
    opmath_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    opmath_t beta,
    scalar_t *c, int64_t ldc) {
  // c = alpha * (a.T @ b) + beta * c
  const scalar_t *a_ = a;
  for (const auto i : c10::irange(m)) {
    const scalar_t *b_ = b;
    for (const auto j : c10::irange(n)) {
      const auto dot = sum(k, [&](int64_t l) -> opmath_t {
        return static_cast<opmath_t>(transa == TransposeType::ConjTranspose ? conj_impl(a_[l]) : a_[l]) * static_cast<opmath_t>(b_[l]);
      });
      b_ += ldb;
      if (beta == opmath_t(0)) {
        c[j*ldc+i] = alpha*dot;
      } else {
        c[j*ldc+i] = beta*c[j*ldc+i]+alpha*dot;
      }
    }
    a_ += lda;
  }
}

template <typename scalar_t, typename opmath_t>
typename std::enable_if<std::is_same<scalar_t, opmath_t>::value, void>::type
gemm_transb_(
    TransposeType transb,
    int64_t m,
    int64_t n,
    int64_t k,
    opmath_t alpha,
    const scalar_t* a,
    int64_t lda,
    const scalar_t* b,
    int64_t ldb,
    opmath_t beta,
    scalar_t* c,
    int64_t ldc) {
  // c *= beta
  scale_(m, n, beta, c, ldc);

  // c += alpha * (a @ b.T)
  for (const auto l : c10::irange(k)) {
    for (const auto j : c10::irange(n)) {
      opmath_t val = (transb == TransposeType::ConjTranspose ? conj_impl(b[j + l * ldb]) : b[j + l * ldb]) * alpha;
      int64_t i_m = m / 4;
      for (const auto i_i : c10::irange(i_m)) {
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

// std::is_same<scalar_t, at::BFloat16> || std::is_same<scalar_t, at::Half>
template <typename scalar_t, typename opmath_t>
typename std::enable_if<!std::is_same<scalar_t, opmath_t>::value, void>::type
gemm_transb_(
    TransposeType transb,
    int64_t m,
    int64_t n,
    int64_t k,
    opmath_t alpha,
    const scalar_t* a,
    int64_t lda,
    const scalar_t* b,
    int64_t ldb,
    opmath_t beta,
    scalar_t* c,
    int64_t ldc) {
  // c += alpha * (a @ b.T)
  for (const auto i : c10::irange(m)) {
    for (const auto j : c10::irange(n)) {
      const auto dot = sum(k, [&](int64_t l) -> opmath_t {
        return static_cast<opmath_t>(a[l * lda + i]) *
            static_cast<opmath_t>(transb == TransposeType::ConjTranspose ? conj_impl(b[l * ldb + j]) : b[l * ldb + j]);
      });
      if (beta == opmath_t(0)) {
        c[j * ldc + i] = alpha * dot;
      } else {
        c[j * ldc + i] = beta * c[j * ldc + i] + alpha * dot;
      }
    }
  }
}

template <typename scalar_t, typename opmath_t>
void gemm_transab_(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    opmath_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    opmath_t beta,
    scalar_t *c, int64_t ldc) {
  // c = beta * c + alpha * (a.T @ b.T)
  for (const auto i : c10::irange(m)) {
    for (const auto j : c10::irange(n)) {
      const auto dot = sum(k, [&](int64_t l) -> opmath_t {
        return static_cast<opmath_t>(transa == TransposeType::ConjTranspose ? conj_impl(a[i * lda + l]) : a[i * lda + l]) *
            static_cast<opmath_t>(transb == TransposeType::ConjTranspose ? conj_impl(b[l * ldb + j]) : b[l * ldb + j]);
      });

      if (beta == opmath_t(0)) {
        c[j * ldc + i] = alpha * dot;
      } else {
        c[j * ldc + i] = beta * c[j * ldc + i] + alpha * dot;
      }
    }
  }
}

#if defined(__aarch64__) && !defined(C10_MOBILE)
template <>
void gemm_notrans_(
    int64_t m,
    int64_t n,
    int64_t k,
    float alpha,
    const at::Half* a,
    int64_t lda,
    const at::Half* b,
    int64_t ldb,
    float beta,
    at::Half* c,
    int64_t ldc) {
  // c += alpha * (a @ b)
  if (n == 1 && beta == 0.0) {
    at::native::blas_impl::fp16_gemv_notrans(m, k, alpha, reinterpret_cast<const float16_t*>(a), lda, reinterpret_cast<const float16_t*>(b), 1, beta, reinterpret_cast<float16_t*>(c), 1);
    return;
  }
  for (const auto i : c10::irange(m)) {
    for (const auto j : c10::irange(n)) {
      const auto dot = sum(k, [&](int64_t l) -> float {
        return float(c10::detail::fp16_from_bits(a[l * lda + i].x)) *
            float(c10::detail::fp16_from_bits(b[j * ldb + l].x));
      });
      if (beta == 0) {
        c[j * ldc + i] = alpha * dot;
      } else {
        c[j * ldc + i] = beta * c[j * ldc + i] + alpha * dot;
      }
    }
  }
}


static float compute_dot(const float16_t *a, const float16_t *b, int64_t l) {
    if ((l&3) != 0) {
      return sum(l, [&](int64_t i) -> float {
        return float(a[i]) * float(b[i]);
      });
    }
    float32x4_t rcv = vdupq_n_f32(0);
    for (int64_t idx = 0; idx < l; idx += 4) {
      float32x4_t aVec = vcvt_f32_f16(vld1_f16(a + idx));
      float32x4_t bVec = vcvt_f32_f16(vld1_f16(b + idx));
      rcv = vaddq_f32(rcv, vmulq_f32(aVec, bVec));
    }
    auto sum = vpaddq_f32(rcv, rcv);
    return vgetq_lane_f32(vpaddq_f32(sum, sum), 0);
}

template <>
void gemm_transa_(
    TransposeType transa,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const at::Half *a, int64_t lda,
    const at::Half *b, int64_t ldb,
    float beta,
    at::Half *c, int64_t ldc) {
  // c = alpha * (a.T @ b) + beta * c
  if (n == 1 && beta == 0.0) {
    at::native::blas_impl::fp16_gemv_trans(k, m, alpha, reinterpret_cast<const float16_t*>(a), lda, reinterpret_cast<const float16_t*>(b), 1, beta, reinterpret_cast<float16_t*>(c), 1);
    return;
  }
  const auto *a_ = a;
  for (const auto i : c10::irange(m)) {
    const auto *b_ = b;
    for (const auto j : c10::irange(n)) {
      const auto dot = compute_dot(reinterpret_cast<const float16_t*>(a_), reinterpret_cast<const float16_t*>(b_), k);
      b_ += ldb;
      if (beta == 0) {
        c[j*ldc+i] = alpha*dot;
      } else {
        c[j*ldc+i] = beta*c[j*ldc+i]+alpha*dot;
      }
    }
    a_ += lda;
  }
}

#endif

template <typename scalar_t, typename opmath_t>
void gemm_core_(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    opmath_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    opmath_t beta,
    scalar_t *c, int64_t ldc) {
  if (transa == TransposeType::NoTranspose &&
      transb == TransposeType::NoTranspose) {
    return gemm_notrans_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else if (
      transa != TransposeType::NoTranspose &&
      transb == TransposeType::NoTranspose) {
    gemm_transa_(transa, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else if (
      transa == TransposeType::NoTranspose &&
      transb != TransposeType::NoTranspose) {
    gemm_transb_(transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else {
    gemm_transab_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}

#if !defined(C10_MOBILE)
#define _AT_DISPATCH_GEMM_TYPES(TYPE, NAME, ...)                                                \
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND6(                                                 \
            kHalf, kBFloat16, kFloat8_e5m2, kFloat8_e4m3fn, kFloat8_e5m2fnuz, kFloat8_e4m3fnuz, \
            TYPE, NAME, __VA_ARGS__)
#else
#define _AT_DISPATCH_GEMM_TYPES(TYPE, NAME, ...)         \
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(          \
            kHalf, kBFloat16,                            \
            TYPE, NAME, __VA_ARGS__)
#endif
void cpublas_gemm_impl(
    at::ScalarType type,
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const Scalar& alpha,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const Scalar& beta,
    void *c, int64_t ldc) {
  _AT_DISPATCH_GEMM_TYPES(type, "cpublas_gemm_impl", [&]{
        using opmath_t = at::opmath_type<scalar_t>;
        gemm_core_(
            transa, transb, m, n, k,
            alpha.to<opmath_t>(),
            static_cast<const scalar_t *>(a), lda,
            static_cast<const scalar_t *>(b), ldb,
            beta.to<opmath_t>(),
            static_cast<scalar_t *>(c), ldc);
      });
}

void cpublas_axpy_impl(at::ScalarType type, int64_t n, const Scalar& _a, const void *_x, int64_t incx, void *_y, int64_t incy){
  if (type == at::kBool) {
      auto a = _a.to<bool>();
      auto x = static_cast<const bool *>(_x);
      auto y = static_cast<bool *>(_y);
      int64_t i;
      for(i = 0; i < n; i++)
        y[i*incy] |= a & x[i*incx];
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(at::kHalf, at::kBFloat16, type, "cpublas_axpy_impl",
      [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        auto a = _a.to<opmath_t>();
        auto x = static_cast<const scalar_t *>(_x);
        auto y = static_cast<scalar_t *>(_y);
        int64_t i;
        for(i = 0; i < n; i++)
          y[i*incy] += a*x[i*incx];
      });
  }
}

void cpublas_copy_impl(at::ScalarType type, int64_t n, const void *_x, int64_t incx, void *_y, int64_t incy){
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(at::kComplexHalf, at::kHalf, at::kBFloat16, at::kBool, type, "cpublas_copy_impl",
    [&] {
      auto x = static_cast<const scalar_t *>(_x);
      auto y = static_cast<scalar_t *>(_y);
      int64_t i;
      for(i = 0; i < n; i++)
        y[i*incy] = x[i*incx];
    });
}

}}  // namespace cpublas::(anonymous)


REGISTER_DISPATCH(cpublas::gemm_stub, &cpublas::cpublas_gemm_impl);
REGISTER_DISPATCH(cpublas::axpy_stub, &cpublas::cpublas_axpy_impl);
REGISTER_DISPATCH(cpublas::copy_stub, &cpublas::cpublas_copy_impl);

}  // namespace at::native
