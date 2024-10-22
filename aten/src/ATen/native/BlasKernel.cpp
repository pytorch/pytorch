#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/Config.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Unroll.h>
#include <c10/util/complex.h>
#include <c10/util/irange.h>
#include <algorithm>
#include <climits>
#include <limits>

#if defined(__aarch64__) && !defined(C10_MOBILE)
#include <arm_neon.h>
#include <cpuinfo.h>
#endif

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-function")
namespace {

/// Wrapper for const_cast<T*> with type-inference.
///
/// Use this to call into APIs that are not const-correct.
template <typename T>
T* remove_const(const T* x) {
  return const_cast<T*>(x);
}

} // namespace

#if AT_BUILD_WITH_BLAS()
extern "C" double ddot_(int *n, double *x, int *incx, double *y, int *incy);
extern "C" void dscal_(int *n, double *a, double *x, int *incx);
extern "C" void sscal_(int *n, float *a, float *x, int *incx);
extern "C" void dgemv_(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern "C" void sgemv_(char *trans, int *m, int *n, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy);

#if AT_BLAS_F2C()
# define ffloat double
#else
# define ffloat float
#endif

#if AT_BLAS_USE_CBLAS_DOT()
  extern "C" float cblas_sdot(const int n, const float *x, const int incx, const float *y, const int incy);
  extern "C" void cblas_cdotu_sub(const int n, const void *x, const int incx, const void *y, const int incy, void *dotu);
  extern "C" void cblas_zdotu_sub(const int n, const void *x, const int incx, const void *y, const int incy, void *dotu);
  extern "C" void cblas_cdotc_sub(const int n, const void *x, const int incx, const void *y, const int incy, void *dotc);
  extern "C" void cblas_zdotc_sub(const int n, const void *x, const int incx, const void *y, const int incy, void *dotc);

  static inline ffloat sdot_(const int *n, const float *x, const int *incx, const float *y, const int *incy)
  {
    return cblas_sdot(*n, x, *incx, y, *incy);
  }
  static inline void cdotu_(std::complex<float> *res, const int *n, const std::complex<float> *x, const int *incx,
  const std::complex<float> *y, const int *incy) {
    cblas_cdotu_sub(*n, x, *incx, y, *incy, res);
  }
  static inline void zdotu_(std::complex<double> *res, const int *n, const std::complex<double> *x, const int *incx,
  const std::complex<double> *y, const int *incy) {
    cblas_zdotu_sub(*n, x, *incx, y, *incy, res);
  }
  static inline void cdotc_(std::complex<float> *res, const int *n, const std::complex<float> *x, const int *incx,
  const std::complex<float> *y, const int *incy) {
    cblas_cdotc_sub(*n, x, *incx, y, *incy, res);
  }
  static inline void zdotc_(std::complex<double> *res, const int *n, const std::complex<double> *x, const int *incx,
  const std::complex<double> *y, const int *incy) {
    cblas_zdotc_sub(*n, x, *incx, y, *incy, res);
  }

#else
  extern "C" ffloat sdot_(int *n, float *x, int *incx, float *y, int *incy);
  extern "C" void cdotu_(std::complex<float> *res, int *n, std::complex<float> *x, int *incx, std::complex<float> *y, int *incy);
  extern "C" void zdotu_(std::complex<double> *res, int *n, std::complex<double> *x, int *incx, std::complex<double> *y, int *incy);
  extern "C" void cdotc_(std::complex<float> *res, int *n, std::complex<float> *x, int *incx, std::complex<float> *y, int *incy);
  extern "C" void zdotc_(std::complex<double> *res, int *n, std::complex<double> *x, int *incx, std::complex<double> *y, int *incy);
#endif // AT_BLAS_USE_CBLAS_DOT
#endif // AT_BUILD_WITH_BLAS

namespace at::native {

namespace blas_impl {
#if defined(__aarch64__) && !defined(C10_MOBILE)
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

float fp16_dot_with_fp32_arith(
    const float16_t* vec1,
    const float16_t* vec2,
    int64_t len);

void bf16_gemv_trans(
    const int m,
    const int n,
    const at::BFloat16 alpha,
    const at::BFloat16* a,
    const int lda,
    const at::BFloat16* x,
    const int incx,
    const at::BFloat16 beta,
    at::BFloat16* y,
    const int incy);

float bf16_dot_with_fp32_arith(
    const at::BFloat16* vec1,
    const at::BFloat16* vec2,
    int64_t len);
#endif

template <typename scalar_t>
bool scal_use_fast_path(
    [[maybe_unused]] int64_t n,
    [[maybe_unused]] int64_t incx) {
  return false;
}

template <typename scalar_t>
bool gemv_use_fast_path(
    [[maybe_unused]] char trans,
    [[maybe_unused]] int64_t m,
    [[maybe_unused]] int64_t n,
    [[maybe_unused]] scalar_t alpha,
    [[maybe_unused]] int64_t lda,
    [[maybe_unused]] int64_t incx,
    [[maybe_unused]] scalar_t beta,
    [[maybe_unused]] int64_t incy) {
  return false;
}

template <typename scalar_t>
void scal_fast_path(
    [[maybe_unused]] int* n,
    [[maybe_unused]] scalar_t* a,
    [[maybe_unused]] scalar_t* x,
    [[maybe_unused]] int* incx) {
  TORCH_INTERNAL_ASSERT(
      false, "scal_fast_path shouldn't be called for this configuration");
}

template <typename scalar_t>
void gemv_fast_path(
    [[maybe_unused]] const char* trans,
    [[maybe_unused]] const int* m,
    [[maybe_unused]] const int* n,
    [[maybe_unused]] const scalar_t* alpha,
    [[maybe_unused]] const scalar_t* a,
    [[maybe_unused]] const int* lda,
    [[maybe_unused]] const scalar_t* x,
    [[maybe_unused]] const int* incx,
    [[maybe_unused]] const scalar_t* beta,
    [[maybe_unused]] scalar_t* y,
    [[maybe_unused]] const int* incy) {
  TORCH_INTERNAL_ASSERT(
      false, "gemv_fast_path shouldn't be called for this configuration");
}

#define INSTANTIATE(scalar_t)                                                                                                                                                     \
template bool scal_use_fast_path<scalar_t>(int64_t n, int64_t incx);                                                                                                              \
template bool gemv_use_fast_path<scalar_t>(char trans, int64_t m, int64_t n, scalar_t alpha, int64_t lda, int64_t incx, scalar_t beta, int64_t incy); \
template void gemv_fast_path<scalar_t>(const char *trans, const int *m, const int *n, const scalar_t *alpha, const scalar_t *a, const int *lda, const scalar_t *x, const int *incx, const scalar_t *beta, scalar_t *y, const int *incy);      \
template void scal_fast_path<scalar_t>(int *n, scalar_t *a, scalar_t *x, int *incx);

#if AT_BUILD_WITH_BLAS()
template <>
bool scal_use_fast_path<double>(int64_t n, int64_t incx) {
  auto intmax = std::numeric_limits<int>::max();
  return n <= intmax && incx <= intmax;
}

template <>
bool scal_use_fast_path<float>(int64_t n, int64_t incx) {
  return scal_use_fast_path<double>(n, incx);
}

template <>
void scal_fast_path<double>(int *n, double *a, double *x, int *incx) {
  dscal_(n, a, x, incx);
}

template <>
void scal_fast_path<float>(int *n, float *a, float *x, int *incx) {
  sscal_(n, a, x, incx);
}

template <>
bool gemv_use_fast_path<float>(
    [[maybe_unused]] char trans,
    int64_t m,
    int64_t n,
    [[maybe_unused]] float alpha,
    int64_t lda,
    int64_t incx,
    [[maybe_unused]] float beta,
    int64_t incy) {
  auto intmax = std::numeric_limits<int>::max();
  return (m <= intmax) && (n <= intmax) && (lda <= intmax) &&
         (incx > 0) && (incx <= intmax) && (incy > 0) && (incy <= intmax);
}

template <>
bool gemv_use_fast_path<double>(
    [[maybe_unused]] char trans,
    int64_t m,
    int64_t n,
    [[maybe_unused]] double alpha,
    int64_t lda,
    int64_t incx,
    [[maybe_unused]] double beta,
    int64_t incy) {
  return gemv_use_fast_path<float>(
      trans, m, n, (float)alpha, lda, incx, (float)beta, incy);
}

template <>
void gemv_fast_path<double>(const char *trans, const int *m, const int *n, const double *alpha, const double *a, const int *lda, const double *x, const int *incx, const double *beta, double *y, const int *incy) {
  dgemv_(remove_const(trans), remove_const(m), remove_const(n), remove_const(alpha), remove_const(a), remove_const(lda), remove_const(x), remove_const(incx), remove_const(beta), y, remove_const(incy));
}

template <>
void gemv_fast_path<float>(const char *trans, const int *m, const int *n, const float *alpha, const float *a, const int *lda, const float *x, const int *incx, const float *beta, float *y, const int *incy) {
  sgemv_(remove_const(trans), remove_const(m), remove_const(n), remove_const(alpha), remove_const(a), remove_const(lda), remove_const(x), remove_const(incx), remove_const(beta), y, remove_const(incy));
}
#else
INSTANTIATE(float);
INSTANTIATE(double);
#endif // AT_BUILD_WITH_BLAS

INSTANTIATE(uint8_t);
INSTANTIATE(int8_t);
INSTANTIATE(int16_t);
INSTANTIATE(int);
INSTANTIATE(int64_t);
#if defined(__aarch64__) && !defined(C10_MOBILE)
template <>
bool scal_use_fast_path<at::Half>(
    [[maybe_unused]] int64_t n,
    [[maybe_unused]] int64_t incx) {
  return false;
}

template <>
bool gemv_use_fast_path<at::Half>(
    [[maybe_unused]] char trans,
    [[maybe_unused]] int64_t m,
    [[maybe_unused]] int64_t n,
    at::Half alpha,
    [[maybe_unused]] int64_t lda,
    [[maybe_unused]] int64_t incx,
    at::Half beta,
    [[maybe_unused]] int64_t incy) {
  return incx == 1 && c10::detail::fp16_from_bits(alpha.x) == 1.0f &&
      c10::detail::fp16_from_bits(beta.x) == 0.0f;
}

template <>
bool gemv_use_fast_path<at::BFloat16>(
    [[maybe_unused]] char trans,
    [[maybe_unused]] int64_t m,
    [[maybe_unused]] int64_t n,
    at::BFloat16 alpha,
    [[maybe_unused]] int64_t lda,
    [[maybe_unused]] int64_t incx,
    at::BFloat16 beta,
    [[maybe_unused]] int64_t incy) {
  return (trans == 'T' || trans == 't') && incx == 1 && alpha == 1.0 &&
      beta == 0.0;
}

#ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC

/*
 * NOTE [ GGML Copyright Notice ]
 * The below reduce overload and fp16_dot_with_fp16_arith function is
 * adapted from llama.cpp's ggml_vec_dot_f16 and surrounding utility
 * functions, so here is the required copyright notice:
 *
 * MIT License
 *
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
// We need the shift for reduce(), hence the extra constants.
static constexpr auto kF16ElementsPerIterationShift = 7;
static constexpr auto kF16ElementsPerIteration = 1 << kF16ElementsPerIterationShift;
static_assert(kF16ElementsPerIteration == 128);

static constexpr auto kF16ElementsPerRegisterShift = 3;
static constexpr auto kF16ElementsPerRegister = 1 << kF16ElementsPerRegisterShift;
static_assert(kF16ElementsPerRegister == 8);

static constexpr auto kF16RegistersPerIterationShift = kF16ElementsPerIterationShift - kF16ElementsPerRegisterShift;
static constexpr auto kF16RegistersPerIteration = 1 << kF16RegistersPerIterationShift;
static_assert(kF16RegistersPerIteration == kF16ElementsPerIteration / kF16ElementsPerRegister);

static inline float reduce(vec::VectorizedN<Half, kF16RegistersPerIteration>& x) {
  int offset = kF16RegistersPerIteration;
  c10::ForcedUnroll<kF16RegistersPerIterationShift>{}([&offset, &x](auto idx) {
    offset /= 2;
    for (int i = 0; i < offset; ++i) {
      x[i] = x[i] + x[offset + i];
    }
  });
  const auto [t0, t1] = vec::convert_half_float(x[0]);
  return vaddvq_f32(t0 + t1);
}

static float fp16_dot_with_fp16_arith(const float16_t* x, const float16_t* a, int len) {
  vec::VectorizedN<Half, kF16RegistersPerIteration> sum(0);

  const auto len_aligned = len & ~(kF16ElementsPerIteration - 1);
  for (int j = 0; j < len_aligned ; j += kF16ElementsPerIteration) {
    for (int k = 0; k < kF16RegistersPerIteration; ++k) {
      const auto temp_x = vec::Vectorized<Half>::loadu(x + j + k * vec::Vectorized<Half>::size());
      const auto temp_a = vec::Vectorized<Half>::loadu(a + j + k * vec::Vectorized<Half>::size());
      sum[k] = vec::fmadd(temp_x, temp_a, sum[k]);
    }
  }
  auto reduced_sum = reduce(sum);

  for (int j = len_aligned; j < len; ++j) {
    reduced_sum += x[j] * a[j];
  }
  return reduced_sum;
}

// Rather than unrolling to process multiple rows (transposed columns)
// of matrix A at once as done in fp16_gemv_trans_fp16_arith, unroll
// along an individual dot product.
static void fp16_gemv_trans_fp16_arith_by_dot_products(const int m, const int n, const float16_t* a, const int lda, const float16_t *x, float16_t* y, int incy) {
  parallel_for(0, n, 1, [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      y[i * incy] = fp16_dot_with_fp16_arith(x, a + lda * i, m);
    }
  });
}

#endif // __ARM_FEATURE_FP16_SCALAR_ARITHMETIC

// The below reduce overload and fp16_dot_with_fp32_arith are adapted
// from llama.cpp's ggml_vec_dot_f32 and surrounding utility
// functions. See NOTE [ GGML Copyright Notice ] above for the
// required notice.

// We need the shift for reduce(), hence the extra constants.
static constexpr auto kF32ElementsPerIterationShift = 5;
static constexpr auto kF32ElementsPerIteration = 1 << kF32ElementsPerIterationShift;
static_assert(kF32ElementsPerIteration == 32);

static constexpr auto kF32ElementsPerRegisterShift = 2;
static constexpr auto kF32ElementsPerRegister = 1 << kF32ElementsPerRegisterShift;
static_assert(kF32ElementsPerRegister == 4);

static constexpr auto kF32RegisterPairsPerIteration = 4;
static constexpr auto kF32RegistersPerIteration = kF32RegisterPairsPerIteration * 2;
static constexpr auto kF32RegistersPerIterationShift = 3;
static_assert(kF32RegistersPerIteration == kF32ElementsPerIteration / kF32ElementsPerRegister);
static_assert(kF32RegistersPerIteration == 1 << kF32RegistersPerIterationShift);

static inline float reduce(vec::VectorizedN<float, kF32RegistersPerIteration>& x) {
  int offset = kF32RegistersPerIteration;
  c10::ForcedUnroll<kF32RegistersPerIterationShift>{}([&offset, &x](auto idx) {
    offset /= 2;
    for (int i = 0; i < offset; ++i) {
      x[i] = vaddq_f32(x[i], x[offset + i]);
    }
  });
  return vaddvq_f32(x[0]);
}

static C10_ALWAYS_INLINE void dot_with_fp32_arith_main_inner_loop_no_bfdot(
  const float16_t* vec1,
  const float16_t* vec2,
  vec::VectorizedN<float, kF32RegistersPerIteration>& sum,
  int registerPairIndex) {
  // Load a pair of f32 registers at a time.
  const auto temp_vec1 = vec::Vectorized<Half>::loadu(&vec1[registerPairIndex * vec::Vectorized<Half>::size()]);
  const auto temp_vec2 = vec::Vectorized<Half>::loadu(&vec2[registerPairIndex * vec::Vectorized<Half>::size()]);

  const auto [result_low, result_high] = vec::fmadd(temp_vec1, temp_vec2, sum[2 * registerPairIndex], sum[2 * registerPairIndex + 1]);
  sum[2 * registerPairIndex] = result_low;
  sum[2 * registerPairIndex + 1] = result_high;
}


static inline float32x4_t f32_fma_f16(float32x4_t a, float16x4_t b, float16x4_t c) {
#ifdef __ARM_FEATURE_FP16_FML
  // NOTE: this instruction is an optional instruction in ARM v8.2 and
  // v8.3, but mandatory in v8.4 per
  // https://developer.arm.com/documentation/ddi0596/2021-03/SIMD-FP-Instructions/FMLAL--FMLAL2--vector---Floating-point-fused-Multiply-Add-Long-to-accumulator--vector--?lang=en
  // I'm not certain that I have the right feature test macro.
  return vfmlalq_low_f16(a, vcombine_f16(b, vdup_n_f16(0)), vcombine_f16(c, vdup_n_f16(0)));
#else
  return vec::fmadd(vec::Vectorized<float>(vcvt_f32_f16(b)), vec::Vectorized<float>(vcvt_f32_f16(c)), vec::Vectorized<float>(a));
#endif
}

static C10_ALWAYS_INLINE void dot_with_fp32_arith_vectorized_tail_inner_loop_no_bfdot(
    const float16_t* vec1,
    const float16_t* vec2,
    vec::Vectorized<float>* tail_sum,
    int idx) {
  const auto temp_vec1 = vld1_f16(&vec1[idx]);
  const auto temp_vec2 = vld1_f16(&vec2[idx]);
  *tail_sum = f32_fma_f16(*tail_sum, temp_vec1, temp_vec2);
}

static float32x4_t to_bfloat16(uint16x4_t u16) {
  int32x4_t shift = vdupq_n_s32(16);
  return vreinterpretq_f32_u32(vshlq_u32(vmovl_u16(u16), shift));
}

static inline float32x4_t f32_fma(float32x4_t a, float32x4_t b, float32x4_t c) {
#ifdef __ARM_FEATURE_FMA
  return vfmaq_f32(a, b, c);
#else
  return vaddq_f32(a, vmulq_f32(b, c));
#endif
}

static float32x4_t f32_fma_bf16(float32x4_t a, uint16x4_t b, uint16x4_t c) {
  return f32_fma(a, to_bfloat16(b), to_bfloat16(c));
}

#if defined(__clang__) && __clang_major__ > 15
// https://godbolt.org/z/z8P4Yncra
#define COMPILER_SUPPORTS_BF16_TARGET 1
#elif !defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 10
// https://gcc.gnu.org/gcc-10/changes.html
// https://godbolt.org/z/cdGG7vn8o
#define COMPILER_SUPPORTS_BF16_TARGET 1
#else
#define COMPILER_SUPPORTS_BF16_TARGET 0
#endif

#if COMPILER_SUPPORTS_BF16_TARGET
#define TARGET_ARM_BF16_ATTRIBUTE __attribute__((target("arch=armv8.2-a+bf16")))

TARGET_ARM_BF16_ATTRIBUTE static C10_ALWAYS_INLINE float32x4_t
f32_dot_bf16(float32x4_t a, bfloat16x8_t b, bfloat16x8_t c) {
  return vbfdotq_f32(a, b, c);
}

TARGET_ARM_BF16_ATTRIBUTE static C10_ALWAYS_INLINE void
dot_with_fp32_arith_main_inner_loop_bfdot(
    const BFloat16* vec1,
    const BFloat16* vec2,
    vec::VectorizedN<float, kF32RegistersPerIteration>& sum,
    int registerPairIndex) {
  const bfloat16x8_t temp_vec1 = vld1q_bf16(reinterpret_cast<const __bf16*>(
                                                &vec1[registerPairIndex * 2 * vec::Vectorized<float>::size()]));
  const bfloat16x8_t temp_vec2 = vld1q_bf16(reinterpret_cast<const __bf16*>(
                                                &vec2[registerPairIndex * 2 * vec::Vectorized<float>::size()]));
  sum[registerPairIndex] =
    f32_dot_bf16(sum[registerPairIndex], temp_vec1, temp_vec2);
}

// See NOTE [GCC code duplication] below for why we have _bfdot and
// _no_bfdot versions of
// dot_with_fp32_arith_vectorized_tail_inner_loop.
TARGET_ARM_BF16_ATTRIBUTE C10_ALWAYS_INLINE
static void dot_with_fp32_arith_vectorized_tail_inner_loop_bfdot(
    const at::BFloat16* vec1,
    const at::BFloat16* vec2,
    vec::Vectorized<float>* tail_sum,
    int idx) {
  const auto temp_vec1 = vld1_u16(reinterpret_cast<const uint16_t*>(&vec1[idx]));
  const auto temp_vec2 = vld1_u16(reinterpret_cast<const uint16_t*>(&vec2[idx]));
  *tail_sum = f32_fma_bf16(*tail_sum, temp_vec1, temp_vec2);
}

#else
#define TARGET_ARM_BF16_ATTRIBUTE
#endif // COMPILER_SUPPORTS_BF16_TARGET

static C10_ALWAYS_INLINE void dot_with_fp32_arith_main_inner_loop_no_bfdot(
    const BFloat16* vec1,
    const BFloat16* vec2,
    vec::VectorizedN<float, kF32RegistersPerIteration>& sum,
    int registerPairIndex) {
  const uint16x8_t temp_vec1 = vld1q_u16(reinterpret_cast<const uint16_t*>(
                                             &vec1[registerPairIndex * 2 * vec::Vectorized<float>::size()]));
  const uint16x8_t temp_vec2 = vld1q_u16(reinterpret_cast<const uint16_t*>(
                                             &vec2[registerPairIndex * 2 * vec::Vectorized<float>::size()]));

  sum[2 * registerPairIndex] = f32_fma_bf16(
      sum[2 * registerPairIndex],
      vget_low_u16(temp_vec1),
      vget_low_u16(temp_vec2));
  sum[2 * registerPairIndex + 1] = f32_fma_bf16(
      sum[2 * registerPairIndex + 1],
      vget_high_u16(temp_vec1),
      vget_high_u16(temp_vec2));
}

static C10_ALWAYS_INLINE void dot_with_fp32_arith_vectorized_tail_inner_loop_no_bfdot(
    const at::BFloat16* vec1,
    const at::BFloat16* vec2,
    vec::Vectorized<float>* tail_sum,
    int idx) {
  const auto temp_vec1 = vld1_u16(reinterpret_cast<const uint16_t*>(&vec1[idx]));
  const auto temp_vec2 = vld1_u16(reinterpret_cast<const uint16_t*>(&vec2[idx]));
  *tail_sum = f32_fma_bf16(*tail_sum, temp_vec1, temp_vec2);
}

namespace {
#if COMPILER_SUPPORTS_BF16_TARGET
template <int n>
struct ForcedUnrollTargetBFloat16 {
  template <typename Func>
  TARGET_ARM_BF16_ATTRIBUTE C10_ALWAYS_INLINE void operator()(const Func& f) const {
    ForcedUnrollTargetBFloat16<n - 1>{}(f);
    f(n - 1);
  }
};

template <>
struct ForcedUnrollTargetBFloat16<1> {
  template <typename Func>
  TARGET_ARM_BF16_ATTRIBUTE C10_ALWAYS_INLINE void operator()(const Func& f) const {
    f(0);
  }
};

C10_ALWAYS_INLINE TARGET_ARM_BF16_ATTRIBUTE auto
dot_with_fp32_arith_main_loop_bfdot(
    const BFloat16* vec1,
    const BFloat16* vec2,
    int64_t len) {
  vec::VectorizedN<float, kF32RegistersPerIteration> sum(0);
  const auto len_aligned = len & ~(kF32ElementsPerIteration - 1);
  for (int j = 0; j < len_aligned ; j += kF32ElementsPerIteration) {
    const auto* vec1_ = vec1 + j;
    const auto* vec2_ = vec2 + j;
    ForcedUnrollTargetBFloat16<kF32RegisterPairsPerIteration>{}([vec1_, vec2_, &sum](auto k)
                                                                C10_ALWAYS_INLINE_ATTRIBUTE TARGET_ARM_BF16_ATTRIBUTE {
      dot_with_fp32_arith_main_inner_loop_bfdot(vec1_, vec2_, sum, k);
    });
  }
  return reduce(sum);
}
#endif // COMPILER_SUPPORTS_BF16_TARGET

template <typename T>
C10_ALWAYS_INLINE auto
dot_with_fp32_arith_main_loop_no_bfdot(
    const T* vec1,
    const T* vec2,
    int64_t len) {
  vec::VectorizedN<float, kF32RegistersPerIteration> sum(0);

  const auto len_aligned = len & ~(kF32ElementsPerIteration - 1);
  for (int j = 0; j < len_aligned ; j += kF32ElementsPerIteration) {
    const auto* vec1_ = vec1 + j;
    const auto* vec2_ = vec2 + j;
    c10::ForcedUnroll<kF32RegisterPairsPerIteration>{}([vec1_, vec2_, &sum](auto k) C10_ALWAYS_INLINE_ATTRIBUTE {
      dot_with_fp32_arith_main_inner_loop_no_bfdot(vec1_, vec2_, sum, k);
    });
  }
  return reduce(sum);
}

// NOTE [GCC code duplication]: The first attempt at landing BFDOT support with
// TARGET_ARM_BF16_ATTRIBUTE failed because unlike clang, GCC will not
// allow inlining a non-bf16-specific function into a bf16-specific
// function. We can work around this by duplicating the code into the
// bfdot and non-bfdot callsites. The code is in this macro to avoid
// actual copy/paste.
#define DOT_WITH_FP32_ARITH_TAIL_AFTER_MAIN_LOOP_BODY(bfdot_suffix)     \
  /* First-tier tail fixup: make sure we handle workloads that can */   \
  /* benefit from vectorization, but don't fit into our fully unrolled */ \
  /* loop above. */                                                     \
  vec::Vectorized<float> tail_sum(0);                                   \
  const auto len_aligned = len & ~(kF32ElementsPerIteration - 1);       \
  const auto len_aligned_4 = len & ~3;                                  \
  for (int j = len_aligned; j < len_aligned_4; j += 4) {                \
    dot_with_fp32_arith_vectorized_tail_inner_loop##bfdot_suffix(vec1, vec2, &tail_sum, j); \
  }                                                                     \
  auto reduced_tail = vpaddq_f32(tail_sum, tail_sum);                   \
  reduced_sum += vgetq_lane_f32(vpaddq_f32(reduced_tail, reduced_tail), 0); \
                                                                        \
  /* Second-tier tail fixup: handle all workloads. */                   \
  for (int j = len_aligned_4; j < len; ++j) {                           \
    reduced_sum += vec1[j] * vec2[j];                                   \
  }                                                                     \
  return reduced_sum

#if COMPILER_SUPPORTS_BF16_TARGET
TARGET_ARM_BF16_ATTRIBUTE float
dot_with_fp32_arith_bfdot(const BFloat16* vec1, const BFloat16* vec2, int64_t len) {
  auto reduced_sum = dot_with_fp32_arith_main_loop_bfdot(vec1, vec2, len);
  DOT_WITH_FP32_ARITH_TAIL_AFTER_MAIN_LOOP_BODY(_bfdot);
}
#endif // COMPILER_SUPPORTS_BF16_TARGET

template <typename T>
C10_ALWAYS_INLINE float
dot_with_fp32_arith_no_bfdot(const T* vec1, const T* vec2, int64_t len) {
  auto reduced_sum = dot_with_fp32_arith_main_loop_no_bfdot(vec1, vec2, len);
  DOT_WITH_FP32_ARITH_TAIL_AFTER_MAIN_LOOP_BODY(_no_bfdot);
}
#undef DOT_WITH_FP32_ARITH_TAIL_AFTER_MAIN_LOOP_BODY
} // namespace

float fp16_dot_with_fp32_arith(const float16_t* vec1, const float16_t* vec2, int64_t len) {
  return dot_with_fp32_arith_no_bfdot(vec1, vec2, len);
}

float bf16_dot_with_fp32_arith(const at::BFloat16* vec1, const at::BFloat16* vec2, int64_t len) {
#if COMPILER_SUPPORTS_BF16_TARGET
  if (cpuinfo_has_arm_bf16()) {
    return dot_with_fp32_arith_bfdot(vec1, vec2, len);
  } else
#endif
  {
    return dot_with_fp32_arith_no_bfdot(vec1, vec2, len);
  }
}

// On my Apple M1 Macbook (which is ARM v8.5 and thus has the
// instructions f32_fma_{low,high}_f16 is targeting), this kernel has
// equivalent performance to the fp16-native kernel.
static void fp16_gemv_trans_fp32_arith_by_dot_products(const int m, const int n, const float16_t* a, const int lda, const float16_t *x, float16_t* y, int incy) {
  parallel_for(0, n, 1, [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      y[i * incy] = fp16_dot_with_fp32_arith(x, a + lda * i, m);
    }
  });
}

static void bf16_gemv_trans_fp32_arith_by_dot_products(const int m, const int n, const at::BFloat16* a, const int lda, const at::BFloat16 *x, at::BFloat16* y, int incy) {
  parallel_for(0, n, 1, [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      y[i * incy] = bf16_dot_with_fp32_arith(x, a + lda * i, m);
    }
  });
}

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
    const int incy) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(incx == 1 && alpha == 1.0 && beta == 0.0);
#ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
  if (at::globalContext().allowFP16ReductionCPU()) {
    return fp16_gemv_trans_fp16_arith_by_dot_products(m, n, a, lda, x, y, incy);
  }
#endif
  return fp16_gemv_trans_fp32_arith_by_dot_products(m, n, a, lda, x, y, incy);
}

void bf16_gemv_trans(
  const int m,
  const int n,
  const at::BFloat16 alpha,
  const at::BFloat16* a,
  const int lda,
  const at::BFloat16* x,
  const int incx,
  const at::BFloat16 beta,
  at::BFloat16* y,
  const int incy) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(incx == 1 && alpha == 1.0 && beta == 0.0);
  return bf16_gemv_trans_fp32_arith_by_dot_products(m, n, a, lda, x, y, incy);
}


#ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
static void fp16_gemv_notrans_fp16_arith(int m, int n, const float16_t* a, const int lda, const float16_t *x, float16_t *y) {
  for (auto j = 0; j < n; j++) {
    auto vecCol = vdup_n_f16(x[j]);
    const auto* column = a + lda * j;
    for (auto i = 0; i < m; i += 4) {
      auto yf16 = y + i;
      auto matRow = vld1_f16(column + i);
      auto resVec = j != 0 ? vld1_f16(yf16) : vdup_n_f16(0);
      resVec = vfma_lane_f16(resVec, matRow, vecCol, 0);
      vst1_f16(yf16, resVec);
    }
  }
}
#endif

static void fp16_gemv_notrans_fp32_arith(int m, int n, const float16_t* a, const int lda, const float16_t *x, float16_t *y) {
  std::vector<float> sum(m);
  for (auto j = 0; j < n; j++) {
    auto vecCol = vdup_n_f32(x[j]);
    const auto* column = a + lda * j;
    for (auto i = 0; i < m; i += 4) {
      auto sf32 = sum.data() + i;
      auto matRow = vcvt_f32_f16(vld1_f16(column + i));
      auto resVec = j != 0 ? vld1q_f32(sf32) : vdupq_n_f32(0);
      resVec = vfmaq_lane_f32(resVec, matRow, vecCol, 0);
      vst1q_f32(sf32, resVec);
    }
  }

  for (auto i = 0; i < m; i+= 4) {
    vst1_f16(y + i, vcvt_f16_f32(vld1q_f32(sum.data() + i)));
  }
}

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
    const int incy) {
  if (incx == 1 && alpha == 1.0 && beta == 0.0 && m % 4 == 0 && incy == 1) {
#ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
    return at::globalContext().allowFP16ReductionCPU() ? fp16_gemv_notrans_fp16_arith(m, n, a, lda, x, y)
                                                       : fp16_gemv_notrans_fp32_arith(m, n, a, lda, x, y);
#else
    return fp16_gemv_notrans_fp32_arith(m, n, a, lda, x, y);
#endif
  }
  std::vector<float> sum(m);
  for (const auto j : c10::irange(n)) {
    const auto* column_ = a + lda * j;
    auto z = alpha * x[j * incx];
    for (const auto i : c10::irange(m)) {
      sum[i] += z * column_[i];
    }
  }
  if (beta == 0.0) {
    for (const auto i : c10::irange(m)) {
      y[i * incy] = sum[i];
    }
  } else {
    for (const auto i : c10::irange(m)) {
      y[i * incy] += sum[i];
    }
  }
}

template <>
void gemv_fast_path<at::Half>(
    const char* trans,
    const int* m,
    const int* n,
    const at::Half* alpha,
    const at::Half* a,
    const int* lda,
    const at::Half* x,
    const int* incx,
    const at::Half* beta,
    at::Half* y,
    const int* incy) {
  using namespace c10::detail;
  if ((trans[0] == 'T') || (trans[0] == 't')) {
    fp16_gemv_trans(
        *m,
        *n,
        fp16_from_bits(alpha->x),
        reinterpret_cast<const float16_t*>(a),
        *lda,
        reinterpret_cast<const float16_t*>(x),
        *incx,
        fp16_from_bits(beta->x),
        reinterpret_cast<float16_t*>(y),
        *incy);
  } else {
    fp16_gemv_notrans(
        *m,
        *n,
        fp16_from_bits(alpha->x),
        reinterpret_cast<const float16_t*>(a),
        *lda,
        reinterpret_cast<const float16_t*>(x),
        *incx,
        fp16_from_bits(beta->x),
        reinterpret_cast<float16_t*>(y),
        *incy);
  }
}

template <>
void gemv_fast_path<at::BFloat16>(
    const char* trans,
    const int* m,
    const int* n,
    const at::BFloat16* alpha,
    const at::BFloat16* a,
    const int* lda,
    const at::BFloat16* x,
    const int* incx,
    const at::BFloat16* beta,
    at::BFloat16* y,
    const int* incy) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(trans[0] == 'T' || trans[0] == 't');
  bf16_gemv_trans(
    *m,
    *n,
    *alpha,
    a,
    *lda,
    x,
    *incx,
    *beta,
    y,
    *incy);
}
#else // defined(__aarch64__) && !defined(C10_MOBILE)
INSTANTIATE(c10::Half);
INSTANTIATE(c10::BFloat16);
#endif // defined(__aarch64__) && !defined(C10_MOBILE)
#undef INSTANTIATE

} // namespace blas_impl

template <typename scalar_t>
inline void scal(int64_t n, scalar_t a, scalar_t *x, int64_t incx)
{
  if (n == 1) incx = 1;
#if AT_BUILD_WITH_BLAS()
  if (blas_impl::scal_use_fast_path<scalar_t>(n, incx)) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    blas_impl::scal_fast_path<scalar_t>(&i_n, &a, x, &i_incx);
    return;
  }
#endif
  for (const auto i : c10::irange(n)) {
    if (a == scalar_t(0)) {
      x[i * incx] = 0;
    } else {
      x[i * incx] *= a;
    }
  }
}

template<typename scalar_t>
void gemv(char trans, int64_t m, int64_t n, scalar_t alpha, const scalar_t *a, int64_t lda, const scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy) {
  if(n == 1) lda = m;

#if AT_BUILD_WITH_BLAS()
  if (blas_impl::gemv_use_fast_path<scalar_t>(trans, m, n, alpha, lda, incx, beta, incy)) {
    TORCH_CHECK(lda >= std::max<int64_t>(1L, m), "lda should be at least max(1,", m, "), but have ", lda);
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    blas_impl::gemv_fast_path<scalar_t>(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
    return;
  }
#endif

  using opmath_t = at::opmath_type<scalar_t>;
  if ((trans == 'T') || (trans == 't')) {
    for (const auto i : c10::irange(n)) {
      opmath_t sum = 0;
      const scalar_t *row_ = a + lda * i;
      for (const auto j : c10::irange(m)) {
        sum += x[j * incx] * row_[j];
      }
      if (beta == scalar_t(0)) {
        y[i * incy] = alpha * sum;
      } else {
        y[i * incy] = beta * y[i * incy] + alpha * sum;
      }
    }
  } else {
    if (beta != scalar_t(1) && beta != scalar_t(0)) scal<scalar_t>(m, beta, y, incy);

    constexpr bool is_low_precision = !std::is_same_v<opmath_t, scalar_t>;
    std::vector<opmath_t> sum;
    if constexpr (is_low_precision) {
      sum.resize(m);
    }
    for (const auto j : c10::irange(n)) {
      const scalar_t *column_ = a + lda * j;
      opmath_t z = alpha * static_cast<opmath_t>(x[j * incx]);
      for (const auto i : c10::irange(m)) {
        //output values are ignored if beta is 0, and set to 0, nans and infs are not propagated
        if (j==0 && beta==scalar_t(0)) {
          if constexpr (!is_low_precision) {
            y[i * incy] = 0;
          }
        }
        if constexpr (is_low_precision) {
          sum[i] += z * column_[i];
        } else {
          y[i * incy] += z * column_[i];
        }
      }
    }
    if constexpr (is_low_precision) {
      if (beta == scalar_t(0)) {
        for (const auto i : c10::irange(m)) {
          y[i * incy] = sum[i];
        }
      } else {
        for (const auto i : c10::irange(m)) {
          y[i * incy] += sum[i];
        }
      }
    }
  }
  return;
}

#define INSTANTIATE(scalar_t, _) \
template void gemv<scalar_t>(char trans, int64_t m, int64_t n, scalar_t alpha, const scalar_t *a, int64_t lda, const scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy);
AT_FORALL_SCALAR_TYPES_AND2(BFloat16, Half, INSTANTIATE);
AT_FORALL_COMPLEX_TYPES(INSTANTIATE);
#undef INSTANTIATE

namespace blas_impl {
#if AT_BUILD_WITH_BLAS()
static float dot_fast_path(int n, float* x, int incx, float* y, int incy) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return sdot_(&n, x, &incx, y, &incy);
}

static double dot_fast_path(int n, double* x, int incx, double* y, int incy) {
  return ddot_(&n, x, &incx, y, &incy);
}

static c10::complex<float> vdot_fast_path(int n, c10::complex<float>* x, int incx, c10::complex<float>* y, int incy) {
  c10::complex<float> result;
  cdotc_(reinterpret_cast<std::complex<float>* >(&result), &n, reinterpret_cast<std::complex<float>*>(x), &incx, reinterpret_cast<std::complex<float>*>(y), &incy);
  return result;
}

static c10::complex<double> vdot_fast_path(int n, c10::complex<double>* x, int incx, c10::complex<double>* y, int incy) {
  c10::complex<double> result;
  zdotc_(reinterpret_cast<std::complex<double>* >(&result), &n, reinterpret_cast<std::complex<double>*>(x), &incx, reinterpret_cast<std::complex<double>*>(y), &incy);
  return result;
}

static c10::complex<double> dot_fast_path(int n, c10::complex<double>* x, int incx, c10::complex<double>* y, int incy) {
  c10::complex<double> result;
  zdotu_(reinterpret_cast<std::complex<double>* >(&result), &n, reinterpret_cast<std::complex<double>*>(x), &incx, reinterpret_cast<std::complex<double>*>(y), &incy);
  return result;
}

static c10::complex<float> dot_fast_path(int n, c10::complex<float>* x, int incx, c10::complex<float>* y, int incy) {
  c10::complex<float> result;
  cdotu_(reinterpret_cast<std::complex<float>* >(&result), &n, reinterpret_cast<std::complex<float>*>(x), &incx, reinterpret_cast<std::complex<float>*>(y), &incy);
  return result;
}
#endif

template <typename scalar_t, typename Functor>
scalar_t dot_naive(
    int64_t n,
    scalar_t* x,
    int64_t incx,
    scalar_t* y,
    int64_t incy,
    Functor op) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t i;
  using opmath_t = at::opmath_type<scalar_t>;
  opmath_t sum = 0;
  for (i = 0; i < n; i++) {
    sum += op(static_cast<opmath_t>(x[i * incx]), static_cast<opmath_t>(y[i * incy]));
  }
  return static_cast<scalar_t>(sum);
}

} // namespace blas_impl

template <typename scalar_t>
scalar_t dot_impl_floating(int64_t n, scalar_t* x, int64_t incx, scalar_t* y, int64_t incy)
{
  if (n == 1) {
    incx = 1;
    incy = 1;
  }
#if AT_BUILD_WITH_BLAS()
        if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
          return blas_impl::dot_fast_path(n, x, incx, y, incy);
        } else {
          return blas_impl::dot_naive(n, x, incx, y, incy, std::multiplies<scalar_t>{});
        }
#else
        { return blas_impl::dot_naive(n, x, incx, y, incy, std::multiplies<scalar_t>{}); }
#endif
}

template <typename scalar_t>
scalar_t dot_impl(int64_t n, scalar_t* x, int64_t incx, scalar_t* y, int64_t incy) {
  if (n == 1) {
    incx = 1;
    incy = 1;
  }
  return blas_impl::dot_naive(n, x, incx, y, incy, std::multiplies<scalar_t>{});
}

template <>
float dot_impl(int64_t n, float* x, int64_t incx, float* y, int64_t incy) {
  return dot_impl_floating(n, x, incx, y, incy);
}

template <>
double dot_impl(int64_t n, double* x, int64_t incx, double* y, int64_t incy) {
  return dot_impl_floating(n, x, incx, y, incy);
}

template <>
c10::complex<double> dot_impl(int64_t n, c10::complex<double>* x, int64_t incx, c10::complex<double>* y, int64_t incy) {
  return dot_impl_floating(n, x, incx, y, incy);
}

template <>
c10::complex<float> dot_impl(int64_t n, c10::complex<float>* x, int64_t incx, c10::complex<float>* y, int64_t incy) {
  return dot_impl_floating(n, x, incx, y, incy);
}

namespace {
template <typename scalar_t>
struct vdot_op {
  scalar_t operator()(scalar_t x, scalar_t y) {
    return std::conj(x) * y;
  }
};
} // anonymous namespace

template <typename scalar_t>
scalar_t vdot_impl(int64_t n, scalar_t* x, int64_t incx, scalar_t* y, int64_t incy) {
  if (n == 1) {
    incx = 1;
    incy = 1;
  }
#if AT_BUILD_WITH_BLAS()
        if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
          return blas_impl::vdot_fast_path(n, x, incx, y, incy);
        } else {
          return blas_impl::dot_naive(n, x, incx, y, incy, vdot_op<scalar_t>{});
        }
#else
        { return blas_impl::dot_naive(n, x, incx, y, incy, vdot_op<scalar_t>{}); }
#endif
}

// Skip reinstantiating the explicitly specialized types `float` and `double`.
#define INSTANTIATE_DOT_IMPL(scalar_t)  \
  template scalar_t dot_impl<scalar_t>( \
      int64_t n, scalar_t * x, int64_t incx, scalar_t * y, int64_t incy);
INSTANTIATE_DOT_IMPL(uint8_t);
INSTANTIATE_DOT_IMPL(int8_t);
INSTANTIATE_DOT_IMPL(int16_t);
INSTANTIATE_DOT_IMPL(int);
INSTANTIATE_DOT_IMPL(int64_t);
INSTANTIATE_DOT_IMPL(c10::Half);
INSTANTIATE_DOT_IMPL(c10::BFloat16);

#define INSTANTIATE_VDOT_IMPL(scalar_t)  \
  template scalar_t vdot_impl<scalar_t>( \
      int64_t n, scalar_t * x, int64_t incx, scalar_t * y, int64_t incy);
INSTANTIATE_VDOT_IMPL(c10::complex<float>);
INSTANTIATE_VDOT_IMPL(c10::complex<double>);

#undef INSTANTIATE_DOT_IMPL

} // namespace at::native
C10_DIAGNOSTIC_POP()
