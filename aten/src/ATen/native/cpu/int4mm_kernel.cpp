#include <type_traits>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/int_mm_kernel.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>
#include <c10/util/Unroll.h>

#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-but-set-variable")
namespace at::native {

namespace {

inline bool is_block_start(int index, int BLOCK_SIZE) {
  return !(index & (BLOCK_SIZE -1));
}

#if (defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2)) && !defined(_MSC_VER)
// convert 16x int4 to int8, handle 64 bits at a time
// used in avx2 and avx512
inline __m128i conver_int4_to_int8(const uint8_t* data) {
  __m128i tmp = _mm_loadu_si64((const __m128i*)data);
  __m128i bytes = _mm_cvtepu8_epi16(tmp);
  const __m128i lowMask = _mm_set1_epi8(0xF);
  __m128i high = _mm_andnot_si128(lowMask, bytes);
  __m128i low = _mm_and_si128(lowMask, bytes);
  high = _mm_slli_epi16(high, 4);
  bytes = _mm_or_si128(low, high);
  return bytes;
}
#endif

#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)

// A block : {BLOCK_M, BLOCK_K}, lda = K
// B block : {BLOCK_K, BLOCK_N / 2}, ldb = BLOCK_N / 2
// C block : {BLOCK_M, BLOCK_N}, ldc = N
//
// ScaleAndZeros block : {1, BLOCK_N, 2}
//
template <int BLOCK_M, int BLOCK_N>
inline void tinygemm_kernel(
    const BFloat16* RESTRICT A,
    const uint8_t* RESTRICT B,
    const BFloat16* RESTRICT ScaleAndZeros,
    BFloat16* RESTRICT C,
    int lda,
    int ldb,
    int ldc,
    int K,
    int BLOCK_K) {

  constexpr int ROWS = BLOCK_M;
  constexpr int COLS = BLOCK_N / 16;

  const int PREFETCH_SIZE_K = 16 * 4;
  const int PREFETCH_SIZE_KB = (PREFETCH_SIZE_K + BLOCK_K - 1) / BLOCK_K;

  // number of blocks on K
  const int KB = K / BLOCK_K;

  __m512 va;
  __m512 vb[COLS];
  __m512 vc[ROWS * COLS];
  __m512 scale[COLS];
  __m512 zero[COLS];

  // Lookup table to de-quantize int4 values to bf16.
  // Values are dequantized as truly int4 [-8, 7] range;
  //
  // dequant = (bf16(int4_value) * bf16_scale) + bf16_zero
  //
  static const __m512 lut = _mm512_set_ps(
      7.0f, 6.0f, 5.0f, 4.0f,
      3.0f, 2.0f, 1.0f, 0.0f,
      -1.0f, -2.0f, -3.0f, -4.0f,
      -5.0f, -6.0f, -7.0f, -8.0f);

  // index for transpose
  static const __m512i idx1 = _mm512_set_epi32(
      30, 28, 26, 24, 22, 20, 18, 16,
      14, 12, 10, 8, 6, 4, 2, 0);
  static const __m512i idx2 = _mm512_set_epi32(
      31, 29, 27, 25, 23, 21, 19, 17,
      15, 13, 11, 9, 7, 5, 3, 1);

  // load scale and zero point
  auto load_scale_and_zeros = [&](int i, int _kb) {
    // load 2x bfloat16 vector
    __m512i t = _mm512_loadu_si512((__m512i*)(ScaleAndZeros + _kb * ldc * 2 + 32 * i));
    if (_kb + PREFETCH_SIZE_KB < KB) {
      _mm_prefetch(ScaleAndZeros + (_kb + PREFETCH_SIZE_KB) * ldc * 2 + 32 * i, _MM_HINT_T0);
    }

    // convert to 2x f32 vector
    __m512 a, b;
    vec::cvtbf16_fp32(t, a, b);

    // transpose scale_and_zero from {16, 2} to {2, 16}
    // inputs:
    //   a: {s0, z0, s1, z1, ..., s7, z7}
    //   b: {s8, z8, s9, z9, ..., s15, z15}
    // output:
    //   scale: {s0, s1, s2, ..., s15}
    //   zero:  {z0, z1, z2, ..., z15}
    scale[i] = _mm512_mask_permutex2var_ps(a, 0xffff, idx1, b);
    zero[i] = _mm512_mask_permutex2var_ps(a, 0xffff, idx2, b);
  };

  auto loadc = [&](auto i) {
    vc[i] = _mm512_setzero_ps();
  };
  c10::ForcedUnroll<ROWS * COLS>{}(loadc);

  auto compute = [&, COLS](auto i, int k) {
    constexpr  int row = i / COLS;
    constexpr  int col = i % COLS;

    if constexpr (col == 0) {
      float aa = static_cast<float>(A[row * lda + k]);
      if (k + PREFETCH_SIZE_K < K) {
        _mm_prefetch(A + row * lda + k + PREFETCH_SIZE_K, _MM_HINT_T0);
      }
      va = _mm512_set1_ps(aa);
    }

    if constexpr (row == 0) {
      if constexpr (COLS == 4) {
        // when BLOCK_N = 64, handle each row at a time
        // to reduce de-quantize overhead.
        if constexpr (col == 0) {
          __m256i b4 = _mm256_loadu_si256((__m256i*)(B + k * ldb));
          if (k + PREFETCH_SIZE_K < K) {
            _mm_prefetch(B + (k + PREFETCH_SIZE_K) * ldb, _MM_HINT_T0);
          }

          __m512i b32 = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(b4));
          vb[0] = _mm512_permutexvar_ps(b32, lut);
          vb[0] = _mm512_fmadd_ps(vb[0], scale[0], zero[0]);
          vb[2] = _mm512_permutexvar_ps(_mm512_srli_epi32(b32, 4), lut);
          vb[2] = _mm512_fmadd_ps(vb[2], scale[2], zero[2]);

          b32 = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(b4, 1));
          vb[1] = _mm512_permutexvar_ps(b32, lut);
          vb[1] = _mm512_fmadd_ps(vb[1], scale[1], zero[1]);
          vb[3] = _mm512_permutexvar_ps(_mm512_srli_epi32(b32, 4), lut);
          vb[3] = _mm512_fmadd_ps(vb[3], scale[3], zero[3]);
        }
      } else {
        __m128i b8 = conver_int4_to_int8(B + k * ldb + col * 8);
        __m512i b32 = _mm512_cvtepu8_epi32(b8);
        vb[col] = _mm512_permutexvar_ps(b32, lut);
        vb[col] = _mm512_fmadd_ps(vb[col], scale[col], zero[col]);
      }
    }

    constexpr int idx = row * COLS + col;
    vc[idx] = _mm512_fmadd_ps(va, vb[col], vc[idx]);
  };

  for (int k = 0, kb = 0; k < K; ++k) {
    if (is_block_start(k, BLOCK_K)) {
      c10::ForcedUnroll<COLS>{}(load_scale_and_zeros, kb++);
    }
    c10::ForcedUnroll<ROWS * COLS>{}(compute, k);
  }

  //store to C
  auto storec = [&, COLS](auto i) {
    constexpr int row = i / COLS;
    constexpr int col = i % COLS;
    if constexpr (COLS == 4) {
      // when BLOCK_N = 64, handle each row at a time
      // to reduce `cvtfp32_bf16` overhead.
      if constexpr (col == 0) {
        __m512i c01 = vec::cvtfp32_bf16(vc[row * 4 + 0], vc[row * 4 + 1]);
        __m512i c23 = vec::cvtfp32_bf16(vc[row * 4 + 2], vc[row * 4 + 3]);
        _mm512_storeu_si512((__m512i*)(C + row * ldc + 0 * 32), c01);
        _mm512_storeu_si512((__m512i*)(C + row * ldc + 1 * 32), c23);
      }
    } else {
      __m256i ci = vec::cvtfp32_bf16(vc[i]);
      _mm256_storeu_si256((__m256i*)(C + row * ldc + col * 16), ci);
    }
  };
  c10::ForcedUnroll<ROWS * COLS>{}(storec);
}

#elif defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)

template <int BLOCK_M, int BLOCK_N>
inline void tinygemm_kernel(
    const BFloat16* RESTRICT A,
    const uint8_t* RESTRICT B,
    const BFloat16* RESTRICT ScaleAndZeros,
    BFloat16* RESTRICT C,
    int lda,
    int ldb,
    int ldc,
    int K,
    int BLOCK_K) {

  constexpr int ROWS = BLOCK_M;
  constexpr int COLS = BLOCK_N / 8;

  const int PREFETCH_SIZE_K = 16 * 4;
  const int PREFETCH_SIZE_KB = (PREFETCH_SIZE_K + BLOCK_K - 1) / BLOCK_K;

  // number of blocks on K
  const int KB = K / BLOCK_K;

  __m256 va;
  __m256 vb[COLS];
  __m256 vc[ROWS * COLS];
  __m256 scale[COLS];
  __m256 zero[COLS];

  static const __m256i idx1 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

  // offset to shift from range [0, 15] to [-8, 7]
  const __m256 offset = _mm256_set1_ps(-8.0f);

  // load scale and zero point
  auto load_scale_and_zeros = [&](int i, int _kb) {
    // load 2x bfloat16 vector
    __m256i t = _mm256_loadu_si256((__m256i*)(ScaleAndZeros + _kb * ldc * 2 + 16 * i));
    if (_kb + PREFETCH_SIZE_KB < KB) {
      _mm_prefetch(ScaleAndZeros + (_kb + PREFETCH_SIZE_KB) * ldc * 2 + 16 * i, _MM_HINT_T0);
    }

    // convert to 2x f32 vector
    __m256 a, b;
    vec::cvtbf16_fp32(t, a, b);

    // transpose scale_and_zero from {8, 2} to {2, 8}
    // inputs:
    //   a: {s0, z0, s1, z1, s2, z2, s3, z3}
    //   b: {s4, z4, s5, z5, s6, z6, s7, z7}
    // output:
    //   scale: {s0, s1, s2, s3, s4, s5, s6, s7}
    //   zero:  {z0, z1, z2, z3, z4, z5, z6, z7}
    a = _mm256_permutevar8x32_ps(a, idx1);
    b = _mm256_permutevar8x32_ps(b, idx1);
    scale[i] = _mm256_permute2f128_ps(a, b, 0b0100000);
    zero[i] = _mm256_permute2f128_ps(a, b, 0b0110001);

    // zero = -8 * scale + zero
    zero[i] = _mm256_fmadd_ps(scale[i], offset, zero[i]);
  };

  auto loadc = [&](auto i) {
    vc[i] = _mm256_setzero_ps();
  };
  c10::ForcedUnroll<ROWS * COLS>{}(loadc);

  auto compute = [&, COLS](auto i, int k) {
    constexpr int row = i / COLS;
    constexpr int col = i % COLS;

    if constexpr (col == 0) {
      float aa = static_cast<float>(A[row * lda + k]);
      if (k + PREFETCH_SIZE_K < K) {
        _mm_prefetch(A + row * lda + k + PREFETCH_SIZE_K, _MM_HINT_T0);
      }
      va = _mm256_set1_ps(aa);
    }

    if constexpr (row == 0) {
      if constexpr (COLS == 4) {
        // when BLOCK_N = 32, handle each row at a time
        if constexpr (col == 0) {
          __m256i mask = _mm256_set1_epi32(0xF);
          __m128i b4 = _mm_loadu_si128((__m128i*)(B + k * ldb));
          if (k + PREFETCH_SIZE_K < K) {
            _mm_prefetch(B + (k + PREFETCH_SIZE_K) * ldb, _MM_HINT_T0);
          }

          __m256i b32 = _mm256_cvtepu8_epi32(b4);
          vb[0] = _mm256_cvtepi32_ps(_mm256_and_si256(b32, mask));
          vb[0] = _mm256_fmadd_ps(vb[0], scale[0], zero[0]);
          vb[2] = _mm256_cvtepi32_ps(_mm256_srli_epi32(b32, 4));
          vb[2] = _mm256_fmadd_ps(vb[2], scale[2], zero[2]);

          b32 = _mm256_cvtepu8_epi32(_mm_shuffle_epi32(b4, _MM_SHUFFLE(3, 2, 3, 2)));
          vb[1] = _mm256_cvtepi32_ps(_mm256_and_si256(b32, mask));
          vb[1] = _mm256_fmadd_ps(vb[1], scale[1], zero[1]);
          vb[3] = _mm256_cvtepi32_ps(_mm256_srli_epi32(b32, 4));
          vb[3] = _mm256_fmadd_ps(vb[3], scale[3], zero[3]);
        }
      } else {
        if constexpr (col % 2 == 0) {
          // de-quantize per 64 bits (16x int4)
          __m128i b8 = conver_int4_to_int8(B + k * ldb + col * 4);
          __m128i b8_val0 = _mm_set1_epi64x(_mm_extract_epi64(b8, 0));
          __m128i b8_val1 = _mm_set1_epi64x(_mm_extract_epi64(b8, 1));
          if (k + PREFETCH_SIZE_K < K) {
            _mm_prefetch(B + (k + PREFETCH_SIZE_K) * ldb + col * 4, _MM_HINT_T0);
          }

          vb[col] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b8_val0));
          vb[col] = _mm256_fmadd_ps(vb[col], scale[col], zero[col]);
          vb[col + 1] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b8_val1));
          vb[col + 1] = _mm256_fmadd_ps(vb[col + 1], scale[col + 1], zero[col + 1]);
        }
      }
    }

    constexpr int idx = row * COLS + col;
    vc[idx] = _mm256_fmadd_ps(va, vb[col], vc[idx]);
  };
  for (int k = 0, kb = 0; k < K; ++k) {
    if (is_block_start(k, BLOCK_K)) {
        c10::ForcedUnroll<COLS>{}(load_scale_and_zeros, kb++);
    }
    c10::ForcedUnroll<ROWS * COLS>{}(compute, k);
  }

  // store to C
  auto storec = [&](auto i) {
    constexpr int row = i / COLS;
    constexpr int col = i % COLS;
    if constexpr (col % 2 == 0) {
      __m256i ci = vec::cvtfp32_bf16(vc[row * COLS + col], vc[row * COLS + col + 1]);
      _mm256_storeu_si256((__m256i*)(C + row * ldc + col * 8), ci);
    }
  };
  c10::ForcedUnroll<ROWS * COLS>{}(storec);
}

#endif

#if !defined(C10_MOBILE) && defined(__aarch64__)
#include <arm_neon.h>

inline float32x4x2_t load_as_float32x4x2(const Half* ptr) {
  float16x4x2_t f16_val = vld2_f16(reinterpret_cast<const float16_t *>(ptr));
  auto val_low = vcvt_f32_f16(f16_val.val[0]);
  auto val_high = vcvt_f32_f16(f16_val.val[1]);
  return {val_low, val_high};
}

inline void store_float32x4(Half* ptr, float32x4_t val) {
    vst1_f16(reinterpret_cast<float16_t*>(ptr), vcvt_f16_f32(val));
}

inline float32x4x2_t load_as_float32x4x2(const BFloat16* ptr) {
  int32x4_t shift = vdupq_n_s32(16);
  uint16x4x2_t u16_val = vld2_u16(reinterpret_cast<const uint16_t *>(ptr));
  uint32x4_t int_low = vmovl_u16(u16_val.val[0]);
  uint32x4_t int_high = vmovl_u16(u16_val.val[1]);
  return {vreinterpretq_f32_u32(vshlq_u32(int_low, shift)), vreinterpretq_f32_u32(vshlq_u32(int_high, shift))};
}

inline void store_float32x4(BFloat16* ptr, float32x4_t val) {
    int32x4_t shift = vdupq_n_s32(-16);
    uint32x4_t uint32_val = vshlq_u32(vreinterpretq_u32_f32(val), shift);
    vst1_u16(reinterpret_cast<uint16_t*>(ptr), vmovn_u32(uint32_val));
}

inline float32x4x2_t load_as_float32x4x2(const float* ptr) {
  return vld2q_f32(ptr);
}

inline void store_float32x4(float* ptr, float32x4_t val) {
    vst1q_f32(ptr, val);
}

template <int BLOCK_M, int BLOCK_N, typename T>
inline void tinygemm_kernel_(
    const T* RESTRICT A,
    const uint8_t* RESTRICT B,
    const T* RESTRICT ScaleAndZeros,
    T* RESTRICT C,
    int lda,
    int ldb,
    int ldc,
    int K,
    int BLOCK_K) {
  int16_t shift_vals[4] = {0, -4, -8, -12};
  int16x4_t shifts = vld1_s16(shift_vals);
  int16x4_t offs = vdup_n_s16(8);
  uint16x4_t mask = vdup_n_u16(0x0F);
  for (const auto m : c10::irange(BLOCK_M)) {
    for (int n = 0; n < BLOCK_N; n+= 16) {
      float32x4_t c_val[4];
      float32x4_t scales[4], zeros[4];
      c10::ForcedUnroll<4>{}([&](auto i) {
          c_val[i] = vdupq_n_f32(0.0);
      });
      for (const auto k : c10::irange(K)) {
        const auto a_val = vdupq_n_f32(static_cast<float>(A[m * lda + k]));
        if (is_block_start(k, BLOCK_K)) {
          int kb = k / BLOCK_K;
          c10::ForcedUnroll<4>{}([&](auto i) {
            auto scales_and_zeros = load_as_float32x4x2(ScaleAndZeros + kb * ldc * 2 + n * 2 + i * 8);
            scales[i] = scales_and_zeros.val[0];
            zeros[i] = scales_and_zeros.val[1];
          });
        }
        c10::ForcedUnroll<4>{}([&](auto i) {
          uint16_t b_pack = reinterpret_cast<const uint16_t*>(B + k * ldb + n / 2)[i];
          uint16x4_t b_masked = vand_u16(vshl_u16(vdup_n_u16(b_pack), shifts), mask);
          int16x4_t b_ints = vsub_s16(vreinterpret_s16_u16(b_masked), offs);
          float32x4_t b_vals = vcvtq_f32_s32(vmovl_s16(b_ints));
          b_vals = vaddq_f32(zeros[i], vmulq_f32(scales[i], b_vals));
          c_val[i] = vfmaq_f32(c_val[i], b_vals, a_val);
        });
      }
      c10::ForcedUnroll<4>{}([&](auto i) {
        store_float32x4(C + m * ldc + n + i * 4, c_val[i]);
      });
    }
  }
}

template <int BLOCK_M, int BLOCK_N>
inline void tinygemm_kernel(
    const Half* RESTRICT A,
    const uint8_t* RESTRICT B,
    const Half* RESTRICT ScaleAndZeros,
    Half* RESTRICT C,
    int lda,
    int ldb,
    int ldc,
    int K,
    int BLOCK_K) {
  tinygemm_kernel_<BLOCK_M, BLOCK_N>(A, B, ScaleAndZeros, C, lda, ldb, ldc, K, BLOCK_K);
}

template <int BLOCK_M, int BLOCK_N>
inline void tinygemm_kernel(
    const BFloat16* RESTRICT A,
    const uint8_t* RESTRICT B,
    const BFloat16* RESTRICT ScaleAndZeros,
    BFloat16* RESTRICT C,
    int lda,
    int ldb,
    int ldc,
    int K,
    int BLOCK_K) {
  tinygemm_kernel_<BLOCK_M, BLOCK_N>(A, B, ScaleAndZeros, C, lda, ldb, ldc, K, BLOCK_K);
}

template <int BLOCK_M, int BLOCK_N>
inline void tinygemm_kernel(
    const float* RESTRICT A,
    const uint8_t* RESTRICT B,
    const float* RESTRICT ScaleAndZeros,
    float* RESTRICT C,
    int lda,
    int ldb,
    int ldc,
    int K,
    int BLOCK_K) {
  tinygemm_kernel_<BLOCK_M, BLOCK_N>(A, B, ScaleAndZeros, C, lda, ldb, ldc, K, BLOCK_K);
}
#endif

template<int BLOCK_N>
inline float convert_int4_to_float(const uint8_t* b, int n) {
  static constexpr float lut[16] = {
    -8.0f, -7.0f, -6.0f, -5.0f,
    -4.0f, -3.0f, -2.0f, -1.0f,
    0.0f, 1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f, 7.0f
  };
  int index;
#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)
  if constexpr (BLOCK_N == 64) {
    const int nb = n/BLOCK_N;
    n -= nb*BLOCK_N;
    if (n < 32) {
      auto val = b[nb * BLOCK_N / 2 + n];
      index = val & 0x0f;
    } else {
      auto val = b[nb * BLOCK_N / 2 + (n - 32)];
      index = val >> 4;
    }
  } else
#elif defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)
  if constexpr (BLOCK_N == 32) {
    const int nb = n/BLOCK_N;
    n -= nb*BLOCK_N;
    if (n < 16) {
      auto val = b[nb * BLOCK_N / 2 + n];
      index = val & 0x0f;
    } else {
      auto val = b[nb * BLOCK_N / 2 + (n - 16)];
      index = val >> 4;
    }
  } else
#endif
  {
    const auto is_even = (n & 1) == 0;
    auto val = b[n/2];
    index = is_even ? (val & 0x0F) : (val >> 4);
  }
  return lut[index];
}

// non-vectorized version
template <int BLOCK_M, int BLOCK_N, typename T>
inline void tinygemm_kernel(
    const T* RESTRICT A,
    const uint8_t* RESTRICT B,
    const T* RESTRICT ScaleAndZeros,
    T* RESTRICT C,
    int lda,
    int ldb,
    int ldc,
    int K,
    int BLOCK_K) {

  for (const auto m : c10::irange(BLOCK_M)) {
    for (const auto n : c10::irange(BLOCK_N)) {
      float c_val = 0;
      for (const auto k : c10::irange(K)) {
        int kb = k / BLOCK_K;
        const auto scale = static_cast<float>(ScaleAndZeros[kb * ldc * 2 + n * 2]);
        const auto zero = static_cast<float>(ScaleAndZeros[kb * ldc * 2 + n * 2 + 1]);
        const auto a_val = static_cast<float>(A[m * lda + k]);
        float b_val = convert_int4_to_float<BLOCK_N>(B + k *ldb, n);
        b_val = b_val * scale + zero;

        c_val += a_val * b_val;
      }
      C[m * ldc + n] = c_val;
    }
  }
}


#define LAUNCH_TINYGEMM_KERNEL(MB_SIZE, NB_SIZE)                 \
  tinygemm_kernel<MB_SIZE, NB_SIZE>(                             \
      A_ptr, B_ptr, S_ptr, C_ptr,                                \
      K, NB_SIZE / 2, N, K, BLOCK_K);

#define LAUNCH_TINYGEMM_NB_SIZE(MB_SIZE)                         \
  switch (nb_size) {                                             \
    case 16:                                                     \
      LAUNCH_TINYGEMM_KERNEL(MB_SIZE, 16);                       \
      break;                                                     \
    case 32:                                                     \
      LAUNCH_TINYGEMM_KERNEL(MB_SIZE, 32);                       \
      break;                                                     \
    case 48:                                                     \
      LAUNCH_TINYGEMM_KERNEL(MB_SIZE, 48);                       \
      break;                                                     \
    case 64:                                                     \
      LAUNCH_TINYGEMM_KERNEL(MB_SIZE, 64);                       \
      break;                                                     \
    default:                                                     \
      TORCH_CHECK(false, "Unsupported n block size: ", nb_size); \
      break;                                                     \
  }

// NB: int4 weight pack (with BLOCK_N 64)
//   weight (int32): {N/64, 64, K}
//   packed (uint8): {N/64, K, 32}
//
// 1. avx512 packed format:
//   When N is 64, to do 256-bit unpacking at a time, we pack Lane0 with Lane2,
//   Lane1 with Lane3 since we can only do shift on a 128-bit basis.
//
//   weight:
//     [Lane0] N0...15:  {a00, a01, a02, ...}
//     [Lane1] N16...31: {a10, a11, a12, ...}
//     [Lane2] N32...47: {a20, a21, a22, ...}
//     [Lane3] N48...63: {a30, a31, a32, ...}
//
//  packed:
//     [Lane02] N0...31:  {a20|a00, a21|a01, a22|a02, ...}
//     [Lane13] N32...63: {a30|a10, a31|a11, a32|a12, ...}
//
//  Note: when N is 16, 32 or 48, pack with 64-bit format.
//
// 2. avx2 packed format:
//   When N is 32, to do 128-bit unpacking at a time.
//
//   weight:
//     [Lane0] N0...15:  { a0,  a1,  a2, ...}
//     [Lane1] N16...32: {a16, a17, a18, ...}
//
//  packed:
//    [Lane01] N0...32: {a16|a0, a17|a1, a18|a2, ...}
//
//  Note: When N is 16, pack with 64-bit format
//
// 3 non-vectorized packed format:
//   Do 64-bit unpacking at a time.
//
//   weight: {a0, a1, a2, a3, ..., a14, a15}
//   packed: {a1|a0, a3, a2, ..., a15|a14}
//
void weight_to_int4pack_kernel(
    const Tensor& weight_packed,
    const Tensor& weight,
    int N, int K) {

  auto weight_packed_data = reinterpret_cast<uint8_t*>(weight_packed.data_ptr());
  const auto weight_data = weight.data_ptr<uint8_t>();

  // 64 for avx512 and 32 for avx2/non-vectorized
  constexpr int BLOCK_N = vec::Vectorized<float>::size() * 4;
  const int NB =  (N + BLOCK_N - 1) / BLOCK_N;
  int K_div_2 = K / 2;

  // parallel on NB blocks
  at::parallel_for(0, NB, 0, [&](int begin, int end) {
    for (const auto i : c10::irange(begin, end)) {
      int nb_size = std::min(BLOCK_N, N - i * BLOCK_N);

      const uint8_t* src = weight_data + i * BLOCK_N * K_div_2;
      uint8_t* dst = weight_packed_data + i * K * BLOCK_N / 2;
      for (const auto k : c10::irange(K_div_2)) {
#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)
        if (nb_size == BLOCK_N) {
          for (const auto d : c10::irange(16)) {
            uint8_t val0 = src[(d + 0) * K_div_2 + k];
            uint8_t val1 = src[(d + 16) * K_div_2 + k];
            uint8_t val2 = src[(d + 32) * K_div_2 + k];
            uint8_t val3 = src[(d + 48) * K_div_2 + k];

            uint8_t packed02_0 = (val2 & 0xF0) | ((val0 & 0xF0) >> 4);
            uint8_t packed13_0 = (val3 & 0xF0) | ((val1 & 0xF0) >> 4);
            uint8_t packed02_1 = ((val2 & 0xF) << 4) | (val0 & 0xF);
            uint8_t packed13_1 = ((val3 & 0xF) << 4) | (val1 & 0xF);

            dst[k * 2 * 32 + d] = packed02_0;
            dst[k * 2 * 32 + 16 + d] = packed13_0;
            dst[(k * 2 + 1) * 32 + d] = packed02_1;
            dst[(k * 2 + 1) * 32 + 16 + d] = packed13_1;
          }
        } else {
          // for nb_size 16, 32, 48
          for (int n = 0; n < nb_size; n += 2) {
            uint8_t val0 = src[n * K_div_2 + k];
            uint8_t val1 = src[n * K_div_2 + K_div_2 + k];

            uint8_t packed_0 = ((val1 & 0xF0)) | ((val0 & 0xF0) >> 4);
            uint8_t packed_1 = ((val1 & 0xF) << 4) | (val0 & 0xF);
            dst[k * 2 * nb_size / 2 + n / 2] = packed_0;
            dst[(k * 2 + 1) * nb_size / 2 + n / 2] = packed_1;
          }
        }
#elif defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)
        if (nb_size == BLOCK_N) {
          // for nb_size 32
          for (const auto d : c10::irange(16)) {
            uint8_t val0 = src[(d + 0) * K_div_2 + k];
            uint8_t val1 = src[(d + 16) * K_div_2 + k];

            uint8_t packed01_0 = ((val1 & 0xF0) | ((val0 & 0xF0) >> 4));
            uint8_t packed01_1 = ((val1 & 0xF) << 4) | (val0 & 0xF);
            dst[k * 2 * 16 + d] = packed01_0;
            dst[(k * 2 + 1) * 16 + d] = packed01_1;
          }
        } else {
          // for nb_size 16
          for (int n = 0; n < nb_size; n += 2) {
            int32_t val0 = src[n * K_div_2 + k];
            int32_t val1 = src[n * K_div_2 + K_div_2 + k];

            uint8_t packed_0 = ((val1 & 0xF0)) | ((val0 & 0xF0) >> 4);
            uint8_t packed_1 = ((val1 & 0xF) << 4) | (val0 & 0xF);
            dst[k * 2 * nb_size / 2 + n / 2] = packed_0;
            dst[(k * 2 + 1) * nb_size / 2 + n / 2] = packed_1;
          }
        }
#else
        for (int n = 0; n < nb_size; n += 2) {
          uint8_t val0 = src[n * K_div_2 + k];
          uint8_t val1 = src[n * K_div_2 + K_div_2 + k];

          uint8_t packed_0 = ((val1 & 0xF0)) | ((val0 & 0xF0) >> 4);
          uint8_t packed_1 = ((val1 & 0xF) << 4) | (val0 & 0xF);
          dst[k * 2 * nb_size / 2 + n / 2] = packed_0;
          dst[(k * 2 + 1) * nb_size / 2 + n / 2] = packed_1;
        }
#endif
      }
    }
  });
}

template<typename T>
void int4pack_mm_kernel_(
    const Tensor& C,
    const Tensor& A,
    const Tensor& B,
    int qGroupSize,
    const Tensor& qScaleAndZeros,
    int N, int K) {

  const auto* A_data = A.const_data_ptr<T>();
  const auto* B_data = reinterpret_cast<const uint8_t*>(B.const_data_ptr());
  auto* C_data = C.data_ptr<T>();
  const auto* S_data = qScaleAndZeros.const_data_ptr<T>();

  int M = A.size(0);

  constexpr int BLOCK_M = 4;
  // 64 for avx512 and 32 for avx2/non-vectorized
  constexpr int BLOCK_N = vec::Vectorized<float>::size() * 4;
  // 32, 64, 128, 256
  const int BLOCK_K = qGroupSize;

  const int MB = (M + BLOCK_M - 1) / BLOCK_M;
  const int NB = (N + BLOCK_N - 1) / BLOCK_N;

  at::parallel_for(0, MB * NB, 0, [&](int begin, int end) {
    int mb{0}, nb{0};
    data_index_init(begin, mb, MB, nb, NB);

    for ([[maybe_unused]] const auto i : c10::irange(begin, end)) {
      int mb_start = mb * BLOCK_M;
      int mb_size = std::min(BLOCK_M, M - mb_start);
      int nb_start = nb * BLOCK_N;
      int nb_size = std::min(BLOCK_N, N - nb_start);

      const auto* A_ptr = A_data + mb_start * K;
      const auto* B_ptr = B_data + nb_start * K / 2;
      const auto* S_ptr = S_data + nb_start * 2;
      auto* C_ptr = C_data + mb_start * N + nb_start;

      switch (mb_size) {
        case 1:
          LAUNCH_TINYGEMM_NB_SIZE(1);
          break;
        case 2:
          LAUNCH_TINYGEMM_NB_SIZE(2);
          break;
        case 3:
          LAUNCH_TINYGEMM_NB_SIZE(3);
          break;
        case 4:
          LAUNCH_TINYGEMM_NB_SIZE(4);
          break;
        default:
          TORCH_CHECK(false, "Unsupported m block size: ", mb_size);
      }

      // move to the next index
      data_index_step(mb, MB, nb, NB);
    }
  });
}

void int4pack_mm_kernel(
    const Tensor& C,
    const Tensor& A,
    const Tensor& B,
    int qGroupSize,
    const Tensor& qScaleAndZeros,
    int N, int K) {
  if (C.scalar_type() == kBFloat16) {
    int4pack_mm_kernel_<BFloat16>(C, A, B, qGroupSize, qScaleAndZeros, N, K);
  } else if (C.scalar_type() == kHalf) {
    int4pack_mm_kernel_<Half>(C, A, B, qGroupSize, qScaleAndZeros, N, K);
  } else {
    int4pack_mm_kernel_<float>(C, A, B, qGroupSize, qScaleAndZeros, N, K);
  }
}

} // anonymous namespace

ALSO_REGISTER_AVX512_DISPATCH(weight_to_int4pack_stub, &weight_to_int4pack_kernel)
ALSO_REGISTER_AVX512_DISPATCH(int4pack_mm_stub, &int4pack_mm_kernel)

} // at::native
C10_DIAGNOSTIC_POP()
