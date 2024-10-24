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
constexpr auto kF32ElementsPerIteration = kF32RegistersPerIteration * kF32ElementsPerRegister;;

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
    for (int i = 0; i < offset; ++i) {
      x[i] = x[i] + x[offset + i];
    }
  });
  const auto [t0, t1] = vec::convert_half_float(x[0]);
#if defined(__aarch64__)
  return vaddvq_f32(t0 + t1);
#else
  return vec::vec_reduce_all<float>(
      std::plus<vec::Vectorized<float>>(),
      t0 + t1);
#endif
}

float fp16_dot_with_fp16_arith(const Half* x, const Half* a, int len) {
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
static void fp16_gemv_trans_fp16_arith_by_dot_products(const int m, const int n, const Half* a, const int lda, const Half *x, const float beta, Half* y, int incy) {
  if (beta == 0.0f) {
    parallel_for(0, n, 1, [&](int begin, int end) {
      for (int i = begin; i < end; ++i) {
        y[i * incy] = fp16_dot_with_fp16_arith(x, a + lda * i, m);
      }
    });
  } else if (beta == 1.0f) {
    parallel_for(0, n, 1, [&](int begin, int end) {
      for (int i = begin; i < end; ++i) {
        y[i * incy] += fp16_dot_with_fp16_arith(x, a + lda * i, m);
      }
    });
  } else {
    parallel_for(0, n, 1, [&](int begin, int end) {
      for (int i = begin; i < end; ++i) {
        y[i * incy] = beta * y[i * incy] + fp16_dot_with_fp16_arith(x, a + lda * i, m);
      }
    });
  }
}

#endif // !defined(__aarch64__) || defined( __ARM_FEATURE_FP16_SCALAR_ARITHMETIC)

float reduce(vec::Vectorized<float> x) {
#if defined(__aarch64__)
  return vaddvq_f32(x);
#else
  return vec::vec_reduce_all<float>(
      std::plus<vec::Vectorized<float>>(),
      x);
#endif
}

float reduce(vec::VectorizedN<float, kF32RegistersPerIteration>& x) {
  int offset = kF32RegistersPerIteration;
  c10::ForcedUnroll<IntegerLog2(kF32RegistersPerIteration)>{}([&offset, &x](auto idx) {
    offset /= 2;
    for (int i = 0; i < offset; ++i) {
      x[i] = x[i] + x[offset + i];
    }
  });
  return reduce(x[0]);
}

C10_ALWAYS_INLINE void dot_with_fp32_arith_main_inner_loop_no_bfdot(
  const Half* vec1,
  const Half* vec2,
  vec::VectorizedN<float, kF32RegistersPerIteration>& sum,
  int registerPairIndex) {
  // Load a pair of f32 registers at a time.
  const auto temp_vec1 = vec::Vectorized<Half>::loadu(&vec1[registerPairIndex * vec::Vectorized<Half>::size()]);
  const auto temp_vec2 = vec::Vectorized<Half>::loadu(&vec2[registerPairIndex * vec::Vectorized<Half>::size()]);

#ifdef __aarch64__
  const auto [result_low, result_high] = vec::fmadd(temp_vec1, temp_vec2, sum[2 * registerPairIndex], sum[2 * registerPairIndex + 1]);
#else
  const auto [temp_vec1_low, temp_vec1_high] = convert_half_float(temp_vec1);
  const auto [temp_vec2_low, temp_vec2_high] = convert_half_float(temp_vec2);
  const auto result_low = vec::fmadd(temp_vec1_low, temp_vec2_low, sum[2 * registerPairIndex]);
  const auto result_high = vec::fmadd(temp_vec1_high, temp_vec2_high, sum[2 * registerPairIndex + 1]);
#endif
  sum[2 * registerPairIndex] = result_low;
  sum[2 * registerPairIndex + 1] = result_high;
}

// Return a + b_low * c_low + b_high * c_high
vec::Vectorized<float> f32_dot_f16(vec::Vectorized<float> a, vec::Vectorized<Half> b, vec::Vectorized<Half> c) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_FML)
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

C10_ALWAYS_INLINE void dot_with_fp32_arith_vectorized_tail_inner_loop_no_bfdot(
    const Half* vec1,
    const Half* vec2,
    vec::Vectorized<float>* tail_sum,
    int idx) {
  const auto temp_vec1 = vec::Vectorized<Half>::loadu(&vec1[idx]);
  const auto temp_vec2 = vec::Vectorized<Half>::loadu(&vec2[idx]);
  *tail_sum = f32_dot_f16(*tail_sum, temp_vec1, temp_vec2);
}

// NOTE: This and some below code is unfortunately duplicated with
// BlasKernel.cpp for now, but will be consolidated when BF16 moves
// here as well.
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

template <typename T>
struct half_to_float16 {
  using type = T;
};


#ifdef __aarch64__
template <>
struct half_to_float16<Half> {
  using type = float16_t;
};
#endif

template <typename T>
using half_to_float16_t = typename half_to_float16<T>::type;

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
  for (int j = len_aligned_vec; j < len; ++j) {                         \
    /* We use half_to_float16_t here because changing to Half was */    \
    /* causing arithmetic to at fp16 precision, but the necessary */    \
    /* necessary behavior to pass python test/test_mps.py -k */         \
    /* test_output_grad_match_nn_functional_linear_cpu_float16 is */    \
    /* fp32. (I'm not sure exactly why this fixes it.) */               \
    half_to_float16_t<std::decay_t<decltype(vec1[j])>> x1 = vec1[j];    \
    half_to_float16_t<std::decay_t<decltype(vec2[j])>> x2 = vec2[j];    \
    reduced_sum += x1 * x2;                                             \
  }                                                                     \
  return reduced_sum

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

// On my Apple M1 Macbook (which is ARM v8.5 and thus has the
// instructions f32_fma_{low,high}_f16 is targeting), this kernel has
// equivalent performance to the fp16-native kernel.
void fp16_gemv_trans_fp32_arith_by_dot_products(const int m, const int n, const Half* a, const int lda, const Half *x, const float beta, Half* y, int incy) {
  if (beta == 0.0f) {
    parallel_for(0, n, 1, [&](int begin, int end) {
      for (int i = begin; i < end; ++i) {
        y[i * incy] = fp16_dot_with_fp32_arith(x, a + lda * i, m);
      }
    });
  } else if (beta == 1.0f) {
    parallel_for(0, n, 1, [&](int begin, int end) {
      for (int i = begin; i < end; ++i) {
        // We need to accumulate in fp32; y[i * incy] += ... gets wrong results.
        y[i * incy] = static_cast<float>(y[i * incy]) + fp16_dot_with_fp32_arith(x, a + lda * i, m);
      }
    });
  } else {
    parallel_for(0, n, 1, [&](int begin, int end) {
      for (int i = begin; i < end; ++i) {
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
    return fp16_gemv_trans_fp16_arith_by_dot_products(m, n, a, lda, x, beta, y, incy);
  }
#endif
  return fp16_gemv_trans_fp32_arith_by_dot_products(m, n, a, lda, x, beta, y, incy);
}
#endif // !defined(C10_MOBILE)

} // namespace CPU_CAPABILITY

#if !defined(C10_MOBILE)
// NOTE: we don't *need* to go through dispatch for the ARM-only
// implementation right now, but we will need it when we cover x86.
REGISTER_DISPATCH(fp16_dot_with_fp32_arith_stub, &fp16_dot_with_fp32_arith);
REGISTER_DISPATCH(fp16_gemv_trans_stub, &fp16_gemv_trans);
#else
#endif // defined(__aarch64__) && !defined(C10_MOBILE)

} // namespace at::native
