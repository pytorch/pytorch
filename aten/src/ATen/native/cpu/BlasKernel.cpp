#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/cpu/zmath.h>
#include <ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.h>
#include <c10/util/irange.h>
#include <c10/util/Unroll.h>
#include <algorithm>

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

float bf16_dot_with_fp32_arith(
  const at::BFloat16* x,
  const at::BFloat16* a,
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
} // namespace at::native::blas_impl
#endif
#if (defined(__aarch64__) || defined(_M_ARM64)) && !defined(C10_MOBILE)
#define AT_BLASKERNEL_ARM_NEON 1
#if !defined(__aarch64__)
#include <arm_neon.h>
#endif
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
#if defined(__aarch64__) || defined(_M_ARM64)
  constexpr int ilp_factor = 8;
#else
  constexpr int ilp_factor = 4;
#endif
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

template <typename scalar_t, typename opmath_t, typename out_t>
__ubsan_ignore_signed_int_overflow__
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
    out_t* c,
    int64_t ldc) {
  // c *= beta
  scale_(m, n, beta, c, ldc);

  // c += alpha * (a @ b)
  const uint64_t unsigned_m = m;
  const uint64_t i_m = unsigned_m / 4;
  for (const uint64_t l : c10::irange(k)) {
    for (const uint64_t j : c10::irange(n)) {
      opmath_t val = b[l + j * ldb] * alpha;
      for (const auto i_i : c10::irange(i_m)) {
        c[j * ldc + i_i * 4 + 0] += a[i_i * 4 + 0 + l * lda] * val;
        c[j * ldc + i_i * 4 + 1] += a[i_i * 4 + 1 + l * lda] * val;
        c[j * ldc + i_i * 4 + 2] += a[i_i * 4 + 2 + l * lda] * val;
        c[j * ldc + i_i * 4 + 3] += a[i_i * 4 + 3 + l * lda] * val;
      }
      uint64_t i = i_m * 4;
      for (; i < unsigned_m; i++)
        c[j * ldc + i] += a[i + l * lda] * val;
    }
  }
}

#if defined(AT_BLASKERNEL_ARM_NEON)
constexpr int64_t kNeonGemmMinWorkPerChunk = 32768;

C10_ALWAYS_INLINE float32x4_t load_bf16_as_f32(const at::BFloat16* p) {
  const uint16x4_t bits = vld1_u16(reinterpret_cast<const uint16_t*>(p));
#if defined(_MSC_VER) && !defined(__clang__)
  return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(bits), 16));
#else
  return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
#endif
}

template <typename out_t>
C10_ALWAYS_INLINE void store_f32x4(float32x4_t acc, float beta, out_t* dst) {
  float tmp[4];
  vst1q_f32(tmp, acc);
  for (const auto t : c10::irange(4)) {
    dst[t] = static_cast<out_t>(
        beta == 0.0f ? tmp[t] : beta * static_cast<float>(dst[t]) + tmp[t]);
  }
}

template <int NRC, typename out_t>
void gemm_notrans_bf16_neon_panel(
    int64_t m,
    int64_t k,
    float alpha,
    const at::BFloat16* a,
    int64_t lda,
    const at::BFloat16* b,
    int64_t ldb,
    float beta,
    out_t* c,
    int64_t ldc) {
  int64_t i = 0;
  for (; i + 16 <= m; i += 16) {
    float32x4_t acc[NRC][4];
    c10::ForcedUnroll<NRC>{}([&](auto jj) {
      acc[jj][0] = vdupq_n_f32(0.0f);
      acc[jj][1] = vdupq_n_f32(0.0f);
      acc[jj][2] = vdupq_n_f32(0.0f);
      acc[jj][3] = vdupq_n_f32(0.0f);
    });
    for (const auto l : c10::irange(k)) {
      const at::BFloat16* a_col = a + l * lda + i;
      const float32x4_t a0 = load_bf16_as_f32(a_col);
      const float32x4_t a1 = load_bf16_as_f32(a_col + 4);
      const float32x4_t a2 = load_bf16_as_f32(a_col + 8);
      const float32x4_t a3 = load_bf16_as_f32(a_col + 12);
      c10::ForcedUnroll<NRC>{}([&](auto jj) {
        const float32x4_t bv = vdupq_n_f32(static_cast<float>(b[jj * ldb + l]) * alpha);
        acc[jj][0] = vfmaq_f32(acc[jj][0], a0, bv);
        acc[jj][1] = vfmaq_f32(acc[jj][1], a1, bv);
        acc[jj][2] = vfmaq_f32(acc[jj][2], a2, bv);
        acc[jj][3] = vfmaq_f32(acc[jj][3], a3, bv);
      });
    }
    c10::ForcedUnroll<NRC>{}([&](auto jj) {
      out_t* c_col = c + jj * ldc + i;
      store_f32x4(acc[jj][0], beta, c_col);
      store_f32x4(acc[jj][1], beta, c_col + 4);
      store_f32x4(acc[jj][2], beta, c_col + 8);
      store_f32x4(acc[jj][3], beta, c_col + 12);
    });
  }
  for (; i + 4 <= m; i += 4) {
    float32x4_t acc[NRC];
    c10::ForcedUnroll<NRC>{}([&](auto jj) { acc[jj] = vdupq_n_f32(0.0f); });
    for (const auto l : c10::irange(k)) {
      const float32x4_t av = load_bf16_as_f32(a + l * lda + i);
      c10::ForcedUnroll<NRC>{}([&](auto jj) {
        const float32x4_t bv = vdupq_n_f32(static_cast<float>(b[jj * ldb + l]) * alpha);
        acc[jj] = vfmaq_f32(acc[jj], av, bv);
      });
    }
    c10::ForcedUnroll<NRC>{}([&](auto jj) { store_f32x4(acc[jj], beta, c + jj * ldc + i); });
  }
  for (; i < m; ++i) {
    float acc[NRC] = {};
    for (const auto l : c10::irange(k)) {
      const float av = static_cast<float>(a[l * lda + i]);
      c10::ForcedUnroll<NRC>{}([&](auto jj) {
        acc[jj] += av * (static_cast<float>(b[jj * ldb + l]) * alpha);
      });
    }
    c10::ForcedUnroll<NRC>{}([&](auto jj) {
      out_t* dst = c + jj * ldc + i;
      *dst = static_cast<out_t>(
          beta == 0.0f ? acc[jj] : beta * static_cast<float>(*dst) + acc[jj]);
    });
  }
}

template <typename out_t>
void gemm_notrans_bf16_neon(
    int64_t m,
    int64_t n,
    int64_t k,
    float alpha,
    const at::BFloat16* a,
    int64_t lda,
    const at::BFloat16* b,
    int64_t ldb,
    float beta,
    out_t* c,
    int64_t ldc) {
  constexpr int NR = 4;
  const int64_t num_blocks = (n + NR - 1) / NR;
  const int64_t work_per_block = std::max<int64_t>(1, NR * m * k);
  const int64_t grain = std::max<int64_t>(1, kNeonGemmMinWorkPerChunk / work_per_block);
  parallel_for(0, num_blocks, grain, [&](int64_t begin, int64_t end) {
    for (const auto block : c10::irange(begin, end)) {
      const int64_t j = block * NR;
      if (j + NR <= n) {
        gemm_notrans_bf16_neon_panel<NR>(m, k, alpha, a, lda, b + j * ldb, ldb, beta, c + j * ldc, ldc);
      } else {
        for (const auto jt : c10::irange(j, n)) {
          gemm_notrans_bf16_neon_panel<1>(m, k, alpha, a, lda, b + jt * ldb, ldb, beta, c + jt * ldc, ldc);
        }
      }
    }
  });
}
#endif // defined(AT_BLASKERNEL_ARM_NEON)

// std::is_same<scalar_t, at::BFloat16> || std::is_same<scalar_t, at::Half>
template <typename scalar_t, typename opmath_t, typename out_t>
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
    out_t* c,
    int64_t ldc) {
#if defined(AT_BLASKERNEL_ARM_NEON)
  if constexpr (std::is_same_v<scalar_t, at::BFloat16> && std::is_same_v<opmath_t, float>) {
    gemm_notrans_bf16_neon(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    return;
  }
#endif
  const auto c_size = m * n;
  auto c_accum = std::make_unique<opmath_t[]>(c_size);
  if (beta == opmath_t(0)) {
    std::fill_n(c_accum.get(), c_size, opmath_t(0));
  } else {
    for (const auto j : c10::irange(n)) {
      for (const auto i : c10::irange(m)) {
        c_accum[j * m + i] = beta * static_cast<opmath_t>(c[j * ldc + i]);
      }
    }
  }

  for (const auto l : c10::irange(k)) {
    for (const auto j : c10::irange(n)) {
      const opmath_t val = static_cast<opmath_t>(b[j * ldb + l]) * alpha;
      opmath_t* c_col = c_accum.get() + j * m;
      const scalar_t* a_col = a + l * lda;
      for (const auto i : c10::irange(m)) {
        c_col[i] += static_cast<opmath_t>(a_col[i]) * val;
      }
    }
  }

  for (const auto j : c10::irange(n)) {
    for (const auto i : c10::irange(m)) {
      c[j * ldc + i] = c_accum[j * m + i];
    }
  }
}

template <typename scalar_t, typename opmath_t, typename out_t>
void gemm_transa_(
    TransposeType transa,
    int64_t m, int64_t n, int64_t k,
    opmath_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    opmath_t beta,
    out_t *c, int64_t ldc) {
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

// in this case, scalar_t == opmath_t == out_t so out_t template param is not needed
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
template <typename scalar_t, typename opmath_t, typename out_t>
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
    out_t* c,
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

template <typename scalar_t, typename opmath_t, typename out_t>
void gemm_transab_(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    opmath_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    opmath_t beta,
    out_t *c, int64_t ldc) {
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
#endif // defined(__aarch64__) && !defined(C10_MOBILE)

#if !defined(C10_MOBILE)
float compute_dot(const at::Half* a, const at::Half* b, int64_t len) {
  return at::native::CPU_CAPABILITY::fp16_dot_with_fp32_arith(
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
  if (n == 1 && alpha == 1.0) {
    at::native::blas_impl::fp16_gemv_trans(k, m, 1.0, a, lda, b, 1, beta, c, 1);
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

float compute_dot(const at::BFloat16* a, const at::BFloat16* b, int64_t len) {
  return at::native::CPU_CAPABILITY::bf16_dot_with_fp32_arith(a, b, len);
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
#endif // !defined(C10_MOBILE)

template <typename scalar_t, typename opmath_t, typename out_t>
void gemm_core_(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    opmath_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    opmath_t beta,
    out_t *c, int64_t ldc) {
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

void cpublas_gemm_no_downcast_impl(
  at::ScalarType type,
  TransposeType transa, TransposeType transb,
  int64_t m, int64_t n, int64_t k,
  const Scalar& alpha,
  const void *a, int64_t lda,
  const void *b, int64_t ldb,
  const Scalar& beta,
  void *c, int64_t ldc) {
_AT_DISPATCH_GEMM_TYPES(type, "cpublas_gemm_no_downcast_impl", [&]{
      using opmath_t = at::opmath_type<scalar_t>;
      gemm_core_(
          transa, transb, m, n, k,
          alpha.to<opmath_t>(),
          static_cast<const scalar_t *>(a), lda,
          static_cast<const scalar_t *>(b), ldb,
          beta.to<opmath_t>(),
          static_cast<opmath_t *>(c), ldc);
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


REGISTER_DISPATCH(cpublas::gemm_stub, &cpublas::cpublas_gemm_impl)
REGISTER_DISPATCH(cpublas::gemm_no_downcast_stub, &cpublas::cpublas_gemm_no_downcast_impl)
REGISTER_DISPATCH(cpublas::axpy_stub, &cpublas::cpublas_axpy_impl)
REGISTER_DISPATCH(cpublas::copy_stub, &cpublas::cpublas_copy_impl)

}  // namespace at::native
