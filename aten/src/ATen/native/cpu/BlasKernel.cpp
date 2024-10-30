#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/cpu/zmath.h>
#include <c10/util/irange.h>
#include <c10/util/Unroll.h>

#if !defined(C10_MOBILE)
namespace at::native::blas_impl {
void fp16_gemv_trans(
    const int m,
    const int n,
    const float alpha,
    const Half* a,
    const int lda,
    const Half* x,
    const int incx,
    const float beta,
    Half* y,
    const int incy);

float fp16_dot_with_fp32_arith(
  const Half* x,
  const Half* a,
  int64_t len);
} // namespace at::native::blas_impl
#endif
#if defined(__aarch64__) && !defined(C10_MOBILE)
#include <arm_neon.h>

namespace at::native::blas_impl {
void fp16_gemv_notrans(
    const int m,
    const int n,
    const float alpha,
    const Half* a,
    const int lda,
    const Half* x,
    const int incx,
    const float beta,
    Half* y,
    const int incy);

float bf16_dot_with_fp32_arith(
  const at::BFloat16* x,
  const at::BFloat16* a,
  int64_t len);
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
std::enable_if_t<std::is_same_v<scalar_t, opmath_t>, void>
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
std::enable_if_t<!std::is_same_v<scalar_t, opmath_t>, void>
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
void gemm_transb_impl(
    TransposeType transb,
    int64_t m,
    int64_t n,
    int64_t k,
    opmath_t alpha,
    const scalar_t* a,
    int64_t lda,
    const scalar_t* b,
    int64_t ldb,
    /* we expect pre-applied beta */
    opmath_t* c,
    int64_t ldc) {
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

template <typename scalar_t, typename opmath_t>
std::enable_if_t<std::is_same_v<scalar_t, opmath_t>, void>
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

  gemm_transb_impl(transb, m, n, k, alpha, a, lda, b, ldb, c, ldc);
}

// std::is_same<scalar_t, at::BFloat16> || std::is_same<scalar_t, at::Half>
template <typename scalar_t, typename opmath_t>
std::enable_if_t<!std::is_same_v<scalar_t, opmath_t>, void>
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
  // We need to calculate full-precision dot products for correctness;
  // users notice error accumulation with reduced-width types (e.g.,
  // https://github.com/pytorch/pytorch/issues/95125 and
  // https://github.com/pytorch/pytorch/issues/83863, which were filed
  // when we used gemm_transb_impl naively, accumulating into
  // float16/bfloat16). The straightforward way to do this is to use
  // the vector dot column algorithm anyway, but this gives terrible
  // performance because of the non-contiguous matrix
  // access. Therefore, we instead elect to allocate temporary space
  // to hold the output at higher-precision so that we can accumulate
  // into it using the above cache-friendly "load one vector element,
  // FMA it with an entire matrix row into the entire result vector"
  // algorithm instead.
  const auto c_size = m * n;
  auto c_accum = std::make_unique<opmath_t[]>(c_size);
  if (beta == 1) {
    for (const auto j : c10::irange(n)) {
      for (const auto i : c10::irange(m)) {
        c_accum[j * m + i] = c[j * ldc + i];
      }
    }
  } else if (beta == 0) {
    for (const auto j : c10::irange(n)) {
      for (const auto i : c10::irange(m)) {
        c_accum[j * m + i] = 0;
      }
    }
  } else {
    for (const auto j : c10::irange(n)) {
      for (const auto i : c10::irange(m)) {
        c_accum[j * m + i] = beta * c[j * ldc + i];
      }
    }
  }
  gemm_transb_impl(transb, m, n, k, alpha, a, lda, b, ldb, c_accum.get(), m);
  for (const auto j : c10::irange(n)) {
    for (const auto i : c10::irange(m)) {
      c[j * ldc + i] = c_accum[j * m + i];
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
  if (n == 1 && beta == 0.0 && alpha == 1.0) {
    at::native::blas_impl::fp16_gemv_notrans(m, k, 1.0, a, lda, b, 1, 0.0, c, 1);
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

static float compute_dot(const at::BFloat16* a, const at::BFloat16* b, int64_t len) {
  return at::native::blas_impl::bf16_dot_with_fp32_arith(a, b, len);
}

template <>
void gemm_transa_(
    TransposeType transa,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const at::BFloat16 *a, int64_t lda,
    const at::BFloat16 *b, int64_t ldb,
    float beta,
    at::BFloat16 *c, int64_t ldc) {
  // c = alpha * (a.T @ b) + beta * c
  parallel_for(0, m, 1, [&](int64_t begin, int64_t end) {
    const auto *a_ = a + begin * lda;
    for (const auto i : c10::irange(begin, end)) {
      const auto *b_ = b;
      for (const auto j : c10::irange(n)) {
        const auto dot = compute_dot(a_, b_, k);
        b_ += ldb;
        if (beta == 0) {
          c[j*ldc+i] = alpha*dot;
        } else {
          c[j*ldc+i] = beta*c[j*ldc+i]+alpha*dot;
        }
      }
      a_ += lda;
    }
  });
}

#endif // defined(__aarch64__) && !defined(C10_MOBILE)

#if !defined(C10_MOBILE)
static float compute_dot(const at::Half* a, const at::Half* b, int64_t len) {
  return at::native::blas_impl::fp16_dot_with_fp32_arith(
      a, b, len);
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
  if (n == 1 && beta == 0.0 && alpha == 1.0) {
    at::native::blas_impl::fp16_gemv_trans(k, m, 1.0, a, lda, b, 1, 0.0, c, 1);
    return;
  }
  parallel_for(0, m, 1, [&](int64_t begin, int64_t end) {
    const auto *a_ = a + begin * lda;
    for (const auto i : c10::irange(begin, end)) {
      const auto *b_ = b;
      for (const auto j : c10::irange(n)) {
        const auto dot = compute_dot(a_, b_, k);
        b_ += ldb;
        if (beta == 0) {
          c[j*ldc+i] = alpha*dot;
        } else {
          c[j*ldc+i] = beta*c[j*ldc+i]+alpha*dot;
        }
      }
      a_ += lda;
    }
  });
}
#endif // !defined(C10_MOBILE)

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
