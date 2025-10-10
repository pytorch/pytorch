#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <c10/util/Unroll.h>
#include <c10/util/irange.h>

#if defined(__aarch64__) && !defined(C10_MOBILE)
#include <arm_neon.h>
#include <cpuinfo.h>
#endif

namespace at::native {
inline namespace CPU_CAPABILITY {
#if !defined(C10_MOBILE)

constexpr auto kF32RegisterPairsPerIteration = 4;
constexpr auto kF32RegistersPerIteration = kF32RegisterPairsPerIteration * 2;
constexpr auto kF32ElementsPerRegister = vec::Vectorized<float>::size();
constexpr auto kF32ElementsPerIteration = kF32RegistersPerIteration * kF32ElementsPerRegister;

namespace {
template <typename T>
constexpr int IntegerLog2(T n, int p = 0) {
  return (n <= 1) ? p : IntegerLog2(n / 2, p + 1);
}
} // namespace

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
#if !defined(__aarch64__) || defined( __ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
constexpr auto kF16RegistersPerIteration = 16;
constexpr auto kF16ElementsPerRegister = vec::Vectorized<Half>::size();
constexpr auto kF16ElementsPerIteration = kF16RegistersPerIteration * kF16ElementsPerRegister;

float reduce(vec::VectorizedN<Half, kF16RegistersPerIteration>& x) {
  int offset = kF16RegistersPerIteration;
  c10::ForcedUnroll<IntegerLog2(kF16RegistersPerIteration)>{}([&offset, &x](auto idx) {
    offset /= 2;
    for (const auto i : c10::irange(offset)) {
      x[i] = x[i] + x[offset + i];
    }
  });
  const auto [t0, t1] = vec::convert_half_float(x[0]);
  return vec::vec_reduce_all<float>(
      std::plus<vec::Vectorized<float>>(),
      t0 + t1);
}

float fp16_dot_with_fp16_arith(const Half* x, const Half* a, int len) {
  vec::VectorizedN<Half, kF16RegistersPerIteration> sum(0);

  const auto len_aligned = len & ~(kF16ElementsPerIteration - 1);
  for (int j = 0; j < len_aligned ; j += kF16ElementsPerIteration) {
    for (const auto k : c10::irange(kF16RegistersPerIteration)) {
      const auto temp_x = vec::Vectorized<Half>::loadu(x + j + k * vec::Vectorized<Half>::size());
      const auto temp_a = vec::Vectorized<Half>::loadu(a + j + k * vec::Vectorized<Half>::size());
      sum[k] = vec::fmadd(temp_x, temp_a, sum[k]);
    }
  }
  auto reduced_sum = reduce(sum);

  for (const auto j : c10::irange(len_aligned, len)) {
    reduced_sum += x[j] * a[j];
  }
  return reduced_sum;
}

// Rather than unrolling to process multiple rows (transposed columns)
// of matrix A at once as done in fp16_gemv_trans_fp16_arith, unroll
// along an individual dot product.
static void fp16_gemv_trans_fp16_arith_by_dot_products(const int m, const int n, const Half* a, const int64_t lda, const Half *x, const float beta, Half* y, int incy) {
  if (beta == 0.0f) {
    parallel_for(0, n, 1, [&](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        y[i * incy] = fp16_dot_with_fp16_arith(x, a + lda * i, m);
      }
    });
  } else if (beta == 1.0f) {
    parallel_for(0, n, 1, [&](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        y[i * incy] += fp16_dot_with_fp16_arith(x, a + lda * i, m);
      }
    });
  } else {
    parallel_for(0, n, 1, [&](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        y[i * incy] = beta * y[i * incy] + fp16_dot_with_fp16_arith(x, a + lda * i, m);
      }
    });
  }
}

#endif // !defined(__aarch64__) || defined( __ARM_FEATURE_FP16_SCALAR_ARITHMETIC)

float reduce(vec::Vectorized<float> x) {
  return vec::vec_reduce_all<float>(
      std::plus<vec::Vectorized<float>>(),
      x);
}

// The below reduce overload and fp16_dot_with_fp32_arith are adapted
// from llama.cpp's ggml_vec_dot_f32 and surrounding utility
// functions. See NOTE [ GGML Copyright Notice ] above for the
// required notice.
float reduce(vec::VectorizedN<float, kF32RegistersPerIteration>& x) {
  int offset = kF32RegistersPerIteration;
  c10::ForcedUnroll<IntegerLog2(kF32RegistersPerIteration)>{}([&offset, &x](auto idx) {
    offset /= 2;
    for (const auto i : c10::irange(offset)) {
      x[i] = x[i] + x[offset + i];
    }
  });
  return reduce(x[0]);
}

// We would have to write a separate SVE-specific path to use SVE
// BFDOT. Deferring that for now to get the NEON/ASIMD BFDOT path
// working.
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
#if defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE) && defined(__clang__) && __clang_major__ > 15
// https://godbolt.org/z/z8P4Yncra
#define COMPILER_SUPPORTS_BF16_TARGET 1
#elif defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE) && !defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 10
// https://gcc.gnu.org/gcc-10/changes.html
// https://godbolt.org/z/cdGG7vn8o
#define COMPILER_SUPPORTS_BF16_TARGET 1
#else // defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE) && defined(__clang__) && __clang_major__ > 15
#define COMPILER_SUPPORTS_BF16_TARGET 0
#endif // defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE) && defined(__clang__) && __clang_major__ > 15
#else // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
#define COMPILER_SUPPORTS_BF16_TARGET 0
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

#if COMPILER_SUPPORTS_BF16_TARGET
#define TARGET_ARM_BF16_ATTRIBUTE __attribute__((target("arch=armv8.2-a+bf16")))

TARGET_ARM_BF16_ATTRIBUTE C10_ALWAYS_INLINE void
dot_with_fp32_arith_main_inner_loop_bfdot(
    const BFloat16* vec1,
    const BFloat16* vec2,
    vec::VectorizedN<float, kF32RegistersPerIteration>& sum,
    int registerPairIndex) {
  // NOTE[Intrinsics in bfdot variant]: We can't use
  // vec::Vectorized<BFloat16>::loadu here because linux-aarch64 GCC
  // inexplicably can't convert Vectorized<BFloat16> to
  // bfloat16x8_t. I suspect a bug or incomplete
  // __attribute__((target)) implementation. Intrinsics should be fine
  // because we're using vbfdotq_f32 below anyway.
  const auto temp_vec1 = vld1q_bf16(
      reinterpret_cast<const bfloat16_t*>(
          &vec1[registerPairIndex * vec::Vectorized<BFloat16>::size()]));
  const auto temp_vec2 = vld1q_bf16(
      reinterpret_cast<const bfloat16_t*>(
          &vec2[registerPairIndex * vec::Vectorized<BFloat16>::size()]));
  sum[registerPairIndex] =
    vbfdotq_f32(sum[registerPairIndex], temp_vec1, temp_vec2);
}

TARGET_ARM_BF16_ATTRIBUTE C10_ALWAYS_INLINE
void dot_with_fp32_arith_vectorized_tail_inner_loop_bfdot(
    const at::BFloat16* vec1,
    const at::BFloat16* vec2,
    vec::Vectorized<float>* tail_sum,
    int idx) {
  // See NOTE[Intrinsics in bfdot variant] above.
  const auto temp_vec1 = vld1q_bf16(reinterpret_cast<const bfloat16_t*>(&vec1[idx]));
  const auto temp_vec2 = vld1q_bf16(reinterpret_cast<const bfloat16_t*>(&vec2[idx]));
  *tail_sum = vbfdotq_f32(*tail_sum, temp_vec1, temp_vec2);
}

#else
#define TARGET_ARM_BF16_ATTRIBUTE
#endif // COMPILER_SUPPORTS_BF16_TARGET

namespace {
// Returns (acc_low + a_low_half * b_low_half, acc_high + a_high_half * b_high_half)
std::pair<vec::Vectorized<float>, vec::Vectorized<float>> fmadd(
    const vec::Vectorized<c10::Half>& a,
    const vec::Vectorized<c10::Half>& b,
    const vec::Vectorized<float>& acc_low,
    const vec::Vectorized<float>& acc_high) {
#if defined(__aarch64__) && ((defined(__ARM_FEATURE_FP16_FML) && !defined(__ARM_FEATURE_SVE)) || (defined(CPU_CAPABILITY_SVE128)))
  return std::make_pair(vfmlalq_low_f16(acc_low, a, b), vfmlalq_high_f16(acc_high, a, b));
#else
  const auto [a_float_low, a_float_high] = convert_half_float(a);
  const auto [b_float_low, b_float_high] = convert_half_float(b);
  return std::make_pair(fmadd(a_float_low, b_float_low, acc_low), fmadd(a_float_high, b_float_high, acc_high));
#endif
}

[[maybe_unused]] std::pair<vec::Vectorized<float>, vec::Vectorized<float>> fmadd(
    const vec::Vectorized<c10::BFloat16>& a,
    const vec::Vectorized<c10::BFloat16>& b,
    const vec::Vectorized<float>& acc_low,
    const vec::Vectorized<float>& acc_high) {
  const auto [a_float_low, a_float_high] = convert_bfloat16_float(a);
  const auto [b_float_low, b_float_high] = convert_bfloat16_float(b);
  return std::make_pair(fmadd(a_float_low, b_float_low, acc_low), fmadd(a_float_high, b_float_high, acc_high));
}

// Return a + b_low * c_low + b_high * c_high
vec::Vectorized<float> fmadd(vec::Vectorized<float> a, vec::Vectorized<Half> b, vec::Vectorized<Half> c) {
#if defined(__aarch64__) && ((defined(__ARM_FEATURE_FP16_FML) && !defined(__ARM_FEATURE_SVE)) || (defined(CPU_CAPABILITY_SVE128)))
  // NOTE: this instruction is an optional instruction in ARM v8.2 and
  // v8.3, but mandatory in v8.4 per
  // https://developer.arm.com/documentation/ddi0596/2021-03/SIMD-FP-Instructions/FMLAL--FMLAL2--vector---Floating-point-fused-Multiply-Add-Long-to-accumulator--vector--?lang=en
  // I'm not certain that I have the right feature test macro.
  vec::Vectorized<float> first = vfmlalq_low_f16(a, b, c);
  return vfmlalq_high_f16(first, b, c);
#else
  const auto [b_float_low, b_float_high] = convert_half_float(b);
  const auto [c_float_low, c_float_high] = convert_half_float(c);
  const auto first = vec::fmadd(b_float_low, c_float_low, a);
  return vec::fmadd(b_float_high, c_float_high, first);
#endif
}

[[maybe_unused]] vec::Vectorized<float> fmadd(
    const vec::Vectorized<float>& acc,
    const vec::Vectorized<c10::BFloat16>& a,
    const vec::Vectorized<c10::BFloat16>& b) {
  const auto [a_float_low, a_float_high] = convert_bfloat16_float(a);
  const auto [b_float_low, b_float_high] = convert_bfloat16_float(b);
  return fmadd(a_float_high, b_float_high, fmadd(a_float_low, b_float_low, acc));
}
} // namespace

template <typename T>
C10_ALWAYS_INLINE void dot_with_fp32_arith_main_inner_loop_no_bfdot(
  const T* vec1,
  const T* vec2,
  vec::VectorizedN<float, kF32RegistersPerIteration>& sum,
  int registerPairIndex) {
  static_assert(std::is_same_v<T, Half> || std::is_same_v<T, BFloat16>);
  const auto temp_vec1 = vec::Vectorized<T>::loadu(&vec1[registerPairIndex * vec::Vectorized<T>::size()]);
  const auto temp_vec2 = vec::Vectorized<T>::loadu(&vec2[registerPairIndex * vec::Vectorized<T>::size()]);

  const auto [result_low, result_high] = fmadd(temp_vec1, temp_vec2, sum[2 * registerPairIndex], sum[2 * registerPairIndex + 1]);
  sum[2 * registerPairIndex] = result_low;
  sum[2 * registerPairIndex + 1] = result_high;
}

template <typename T>
C10_ALWAYS_INLINE void dot_with_fp32_arith_vectorized_tail_inner_loop_no_bfdot(
    const T* vec1,
    const T* vec2,
    vec::Vectorized<float>* tail_sum,
    int idx) {
  const auto temp_vec1 = vec::Vectorized<T>::loadu(&vec1[idx]);
  const auto temp_vec2 = vec::Vectorized<T>::loadu(&vec2[idx]);
  *tail_sum = fmadd(*tail_sum, temp_vec1, temp_vec2);
}

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

static_assert(
    (vec::Vectorized<Half>::size() & (vec::Vectorized<Half>::size() - 1)) == 0,
    "Below code expects power-of-2 vector register size!");

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
  const auto len_aligned_vec = len & ~(vec::Vectorized<Half>::size() - 1); \
  for (int j = len_aligned; j < len_aligned_vec; j += vec::Vectorized<Half>::size()) { \
    dot_with_fp32_arith_vectorized_tail_inner_loop##bfdot_suffix(vec1, vec2, &tail_sum, j); \
  }                                                                     \
  reduced_sum += reduce(tail_sum);                                      \
                                                                        \
  /* Second-tier tail fixup: handle all workloads. */                   \
  for (const auto j : c10::irange(len_aligned_vec, len)) {                         \
    /* Attempting to use Half here caused multiple test failures; */    \
    /* using float to unbreak. (Suspect we need a scalar FMA.) */       \
    float x1 = vec1[j];                                                 \
    float x2 = vec2[j];                                                 \
    reduced_sum += x1 * x2;                                             \
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

float fp16_dot_with_fp32_arith(const Half* vec1, const Half* vec2, int64_t len) {
  return dot_with_fp32_arith_no_bfdot(vec1, vec2, len);
}

void fp16_gemv_trans_fp32_arith_by_dot_products(const int m, const int n, const Half* a, const int64_t lda, const Half *x, const float beta, Half* y, int incy) {
  if (beta == 0.0f) {
    parallel_for(0, n, 1, [&](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        y[i * incy] = fp16_dot_with_fp32_arith(x, a + lda * i, m);
      }
    });
  } else if (beta == 1.0f) {
    parallel_for(0, n, 1, [&](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        // We need to accumulate in fp32; y[i * incy] += ... gets wrong results.
        y[i * incy] = static_cast<float>(y[i * incy]) + fp16_dot_with_fp32_arith(x, a + lda * i, m);
      }
    });
  } else {
    parallel_for(0, n, 1, [&](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        y[i * incy] = beta * y[i * incy] + fp16_dot_with_fp32_arith(x, a + lda * i, m);
      }
    });
  }
}

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
    const int incy) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(incx == 1 && alpha == 1.0);
#if !defined(__aarch64__) || defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
  if (at::globalContext().allowFP16ReductionCPU()) {
    fp16_gemv_trans_fp16_arith_by_dot_products(m, n, a, lda, x, beta, y, incy);
    return;
  }
#endif
  fp16_gemv_trans_fp32_arith_by_dot_products(m, n, a, lda, x, beta, y, incy);
}

float bf16_dot_with_fp32_arith(const at::BFloat16* vec1, const at::BFloat16* vec2, int64_t len) {
#if COMPILER_SUPPORTS_BF16_TARGET
  if (cpuinfo_has_arm_bf16()) {
    return dot_with_fp32_arith_bfdot(vec1, vec2, len);
  } else
#endif // COMPILER_SUPPORTS_BF16_TARGET
  {
    return dot_with_fp32_arith_no_bfdot(vec1, vec2, len);
  }
}

void bf16_gemv_trans_fp32_arith_by_dot_products(const int m, const int n, const at::BFloat16* a, const int64_t lda, const at::BFloat16 *x, at::BFloat16* y, int incy) {
  parallel_for(0, n, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      y[i * incy] = bf16_dot_with_fp32_arith(x, a + lda * i, m);
    }
  });
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
  bf16_gemv_trans_fp32_arith_by_dot_products(m, n, a, lda, x, y, incy);
}

float fp16_dot(
  const int64_t n,
  const at::Half* x,
  const int64_t incx,
  const at::Half* y,
  const int64_t incy) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(incx == 1 && incy == 1);
  return fp16_dot_with_fp32_arith(x, y, n);
}

float bf16_dot(
  const int64_t n,
  const at::BFloat16* x,
  const int64_t incx,
  const at::BFloat16* y,
  const int64_t incy) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(incx == 1 && incy == 1);
  return bf16_dot_with_fp32_arith(x, y, n);
}

#endif // !defined(C10_MOBILE)
} // namespace CPU_CAPABILITY

#if !defined(C10_MOBILE)
REGISTER_DISPATCH(fp16_gemv_trans_stub, &fp16_gemv_trans)
REGISTER_DISPATCH(bf16_gemv_trans_stub, &bf16_gemv_trans)
REGISTER_DISPATCH(fp16_dot_stub, &fp16_dot)
REGISTER_DISPATCH(bf16_dot_stub, &bf16_dot)
#endif //!defined(C10_MOBILE)

} // namespace at::native
