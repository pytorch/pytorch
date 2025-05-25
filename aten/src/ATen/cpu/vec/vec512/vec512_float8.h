#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#if (defined(CPU_CAPABILITY_AVX512))
#define SLEEF_STATIC_LIBS
#include <sleef.h>
#endif

namespace at::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)

static inline void cvtfp8e4m3_fp32(const __m128i& a, __m512& o) {
  // Zero Extend
  __m512i x = _mm512_cvtepu8_epi32(a);
  __m512i val = _mm512_and_epi32(
      _mm512_slli_epi32(x, 24), _mm512_set1_epi32(0x7FFFFFFF)); // nonsign_val
  __m512i mant =
      _mm512_and_si512(x, _mm512_set1_epi32(0x07)); // mantissa = x & 0x07
  __m512i exp = _mm512_and_si512(
      _mm512_srli_epi32(x, 3),
      _mm512_set1_epi32(0x0F)); // exp = (x >> 3) & 0x0F
  __m512i sign =
      _mm512_and_si512(x, _mm512_set1_epi32(0x80)); // sign = x & 0x80
  __m512i _zeros = _mm512_setzero_si512();

  // --- Step 1: Calculate the renorm_shift
  __m512i renorm_shift = _zeros;
  // Denorm case (exp == 0 && mant != 0) ---
  __mmask16 denormal_mask = _mm512_cmpeq_epi32_mask(exp, _zeros) &
      _mm512_cmpneq_epi32_mask(mant, _zeros);
  if (denormal_mask) {
    // An alternative solution is as what scalar did in
    // pytorch/c10/util/Float8_e4m3fn.h To count the num of leading zeros, since
    // here we know the unsigned denorm value has zero sign and exp which is 5
    // leading zeros, we need to count the leading zero of mant (3bit) which may
    // done through table lookup for example: const uint8_t lz_table[8] = {3, 2,
    // 1, 1, 0, 0, 0, 0}; num_leading_zero = lz_table[mant] + 5;

    __m512i _ones = _mm512_set1_epi32(1);
    __m512i _twos = _mm512_set1_epi32(2);
    __m512i _threes = _mm512_set1_epi32(3);

    // Default leading zero number for denorm value is 1 = 5 - 4
    __m512i denorm_renorm_shift = _ones;
    // For mant 001, leading zero number is 3 = 7 -4
    __mmask16 leading_Zero_mask = _mm512_cmpeq_epi32_mask(mant, _ones);
    denorm_renorm_shift =
        _mm512_mask_mov_epi32(denorm_renorm_shift, leading_Zero_mask, _threes);
    // For mant 010 and 011, leading zero number is 2 = 6 -4
    leading_Zero_mask = _mm512_cmpeq_epi32_mask(mant, _twos);
    denorm_renorm_shift =
        _mm512_mask_mov_epi32(denorm_renorm_shift, leading_Zero_mask, _twos);
    leading_Zero_mask = _mm512_cmpeq_epi32_mask(mant, _threes);
    denorm_renorm_shift =
        _mm512_mask_mov_epi32(denorm_renorm_shift, leading_Zero_mask, _twos);

    renorm_shift =
        _mm512_mask_mov_epi32(renorm_shift, denormal_mask, denorm_renorm_shift);
  }

  // --- Step 2: calculate norm and denorm ---
  __m512i norm_shifted =
      _mm512_srli_epi32(_mm512_sllv_epi32(val, renorm_shift), 4);
  // exponent bias adjustment: (0x78 - renorm_shift) << 23
  __m512i exp_bias = _mm512_slli_epi32(
      _mm512_sub_epi32(_mm512_set1_epi32(0x78), renorm_shift), 23);
  val = _mm512_add_epi32(norm_shifted, exp_bias);

  // --- Step 3: Nan case (exp == 0xF && mant == 0x07) ---
  __mmask16 nan_mask = _mm512_cmpeq_epi32_mask(exp, _mm512_set1_epi32(0xF)) &
      _mm512_cmpeq_epi32_mask(mant, _mm512_set1_epi32(0x07));
  if (nan_mask) {
    const __m512i nan_values = _mm512_set1_epi32(0x7FC00000);
    val = _mm512_mask_mov_epi32(val, nan_mask, nan_values);
  }

  // --- Step 4: Zero case (exp == 0x00 && mant == 0x00) ---
  __mmask16 zero_mask = _mm512_cmpeq_epi32_mask(exp, _zeros) &
      _mm512_cmpeq_epi32_mask(mant, _zeros);
  if (zero_mask) {
    val = _mm512_mask_mov_epi32(val, zero_mask, _zeros);
  }

  // --- Step 5: OR with sign (sign bit << 24 to get to bit 31) ---
  val = _mm512_or_si512(val, _mm512_slli_epi32(sign, 24));

  o = _mm512_castsi512_ps(val);
}

static inline __m128i cvtfp32_fp8e4m3(const __m512& src) {
  // cvt 16x32 from fp32 to fp8 e4m3
  const __m512i sign_mask = _mm512_set1_epi32(0x80000000);
  const __m512i fp8_max = _mm512_set1_epi32(UINT32_C(1087) << 20);
  const __m512i denorm_thresh = _mm512_set1_epi32(UINT32_C(121) << 23);
  const __m512i denorm_mask = _mm512_set1_epi32(UINT32_C(141) << 23);
  const __m512i bias_part1 = _mm512_set1_epi32((uint32_t)(7 - 127) << 23);
  const __m512i rounding_bias = _mm512_set1_epi32(0x7FFFF);
  __m512i f_bits = _mm512_castps_si512(src);
  // Extract and save sign
  __m512i sign = _mm512_and_epi32(f_bits, sign_mask);
  f_bits = _mm512_xor_epi32(f_bits, sign);

  // Prepare result containers
  __m512i result = _mm512_setzero_si512();

  // Step 1: Handle case of overflow
  // (f_bits >= fp8_max): set result = 0x7f
  __mmask16 overflow_mask = _mm512_cmpge_epu32_mask(f_bits, fp8_max);
  if (overflow_mask) {
    result = _mm512_mask_set1_epi32(result, overflow_mask, 0x7f);
  }

  // Step 2: Handle small numbers (denormals)
  // Small numbers (f_bits < denorm_thresh)
  __mmask16 denorm_thresh_mask = _mm512_cmplt_epu32_mask(f_bits, denorm_thresh);

  if (denorm_thresh_mask) {
    __m512 small_input = _mm512_castsi512_ps(f_bits);
    __m512 small_denorm =
        _mm512_add_ps(small_input, _mm512_castsi512_ps(denorm_mask));
    __m512i small_denorm_bits = _mm512_castps_si512(small_denorm);
    __m512i small_result = _mm512_sub_epi32(small_denorm_bits, denorm_mask);
    result = _mm512_mask_mov_epi32(result, denorm_thresh_mask, small_result);
  }

  // Step 3: Handle normal numbers
  __mmask16 normal_mask = ~(overflow_mask | denorm_thresh_mask);

  if (normal_mask) {
    // mant_odd = (f_bits >> 20) & 1
    __m512i mant_odd =
        _mm512_and_epi32(_mm512_srli_epi32(f_bits, 20), _mm512_set1_epi32(1));
    // f_bits += bias_part1 + rounding_bias
    __m512i rounded = _mm512_add_epi32(f_bits, bias_part1);
    rounded = _mm512_add_epi32(rounded, rounding_bias);
    // Add mant_odd
    rounded = _mm512_add_epi32(rounded, mant_odd);
    // Shift right by 20 bits
    __m512i normal_result = _mm512_srli_epi32(rounded, 20);
    result = _mm512_mask_mov_epi32(result, normal_mask, normal_result);
  }

  // Merge back the sign
  __m512i sign_shifted = _mm512_srli_epi32(sign, 24);
  result = _mm512_or_epi32(result, sign_shifted);

  // Now result is 16 x 32-bit integers, but we only need 8-bit for each
  __m512i packed = _mm512_and_si512(result, _mm512_set1_epi32(0xFF));

  // Narrow 32-bit integers to 8-bit
  return _mm512_cvtepi32_epi8(packed);
}

static inline float fp8e4m3_to_fp32_scalar(uint8_t val) {
  __m512i v = _mm512_set1_epi8(val);
  __m128i v_128 = _mm512_castsi512_si128(v);
  __m512 o;
  cvtfp8e4m3_fp32(v_128, o);
  return _mm512_cvtss_f32(o);
}

static inline uint8_t fp32_to_fp8e4m3_scalar(float val) {
  __m512 v = _mm512_set1_ps(val);
  __m128i o = cvtfp32_fp8e4m3(v);
  return static_cast<std::uint8_t>(_mm_cvtsi128_si32(o));
}

static inline void cvtfp8e5m2_fp32(const __m128i& a, __m512& o) {
  __m256i a_256 = _mm256_castsi128_si256(a);
  __m512i a_512 = _mm512_cvtepu8_epi16(a_256);
  a_512 = _mm512_slli_epi16(a_512, 8);
  a_256 = _mm512_castsi512_si256(a_512);
  cvtfp16_fp32(a_256, o);
}

static inline __m128i cvtfp32_fp8e5m2(const __m512& src) {
  constexpr uint32_t fp32_inf = UINT32_C(255) << 23;
  constexpr uint32_t fp8_max = UINT32_C(143) << 23;
  constexpr uint32_t denorm_mask = UINT32_C(134) << 23;

  // Cvt to bits
  __m512i input_bits = _mm512_castps_si512(src);
  __m512i result = _mm512_setzero_si512();

  // Get the sign
  __m512i sign = _mm512_and_si512(input_bits, _mm512_set1_epi32(0x80000000));

  // Get the unsigned input
  input_bits = _mm512_xor_si512(input_bits, sign);

  // Calculate the mask for inf, nan and denorm
  __mmask16 greater_than_fp8_max =
      _mm512_cmpge_epi32_mask(input_bits, _mm512_set1_epi32(fp8_max));
  __mmask16 greater_than_fp32_inf =
      _mm512_cmpgt_epi32_mask(input_bits, _mm512_set1_epi32(fp32_inf));
  __mmask16 less_than_normal = _mm512_cmpgt_epi32_mask(
      _mm512_set1_epi32((UINT32_C(113) << 23)), input_bits);
  __m512i temp_bits_for_denorm = _mm512_setzero_si512();
  if (less_than_normal) {
    __m512i denorm_mask_512i = _mm512_set1_epi32(denorm_mask);
    temp_bits_for_denorm = _mm512_castps_si512(_mm512_add_ps(
        _mm512_castsi512_ps(input_bits),
        _mm512_castsi512_ps(denorm_mask_512i)));
    temp_bits_for_denorm =
        _mm512_sub_epi32(temp_bits_for_denorm, denorm_mask_512i);
  }

  // Step 1: Norm Val
  __m512i mant_odd_mask =
      _mm512_and_epi32(_mm512_srli_epi32(input_bits, 21), _mm512_set1_epi32(1));
  input_bits = _mm512_add_epi32(
      input_bits, _mm512_set1_epi32(((uint32_t)(15 - 127) << 23) + 0xFFFFF));
  input_bits = _mm512_add_epi32(input_bits, mant_odd_mask);
  result = _mm512_srli_epi32(input_bits, 21);

  // Step 2: INF and NAN
  if (greater_than_fp8_max) {
    result = _mm512_mask_mov_epi32(
        result, greater_than_fp8_max, _mm512_set1_epi8(0x7C));
    if (greater_than_fp32_inf) {
      result = _mm512_mask_mov_epi32(
          result, greater_than_fp32_inf, _mm512_set1_epi8(0x7F));
    }
  }

  // Step 3: Denorm val
  if (less_than_normal) {
    result =
        _mm512_mask_mov_epi32(result, less_than_normal, temp_bits_for_denorm);
  }

  // Step 4: restore sign
  result = _mm512_or_si512(result, _mm512_srli_epi32(sign, 24));

  return _mm512_cvtepi32_epi8(result);
}

static inline float fp8e5m2_to_fp32_scalar(uint8_t val) {
  __m512i v = _mm512_set1_epi8(val);
  __m128i v_128 = _mm512_castsi512_si128(v);
  __m512 o;
  cvtfp8e5m2_fp32(v_128, o);
  return _mm512_cvtss_f32(o);
}

static inline uint8_t fp32_to_fp8e5m2_scalar(float val) {
  __m512 v = _mm512_set1_ps(val);
  __m128i o = cvtfp32_fp8e5m2(v);
  return static_cast<std::uint8_t>(_mm_cvtsi128_si32(o));
}

template <typename T>
class Vectorizedf8 {
  static_assert(
      std::integral_constant < bool,
      std::is_same_v<T, at::Float8_e4m3fn> || std::is_same_v < T,
      at::Float8_e5m2 >> ::value,
      "Support only float8 e4m3.");

 private:
  __m512i values;
  template <typename Op, typename VectorizedType>
  Vectorized<T> inline binary_compare(const VectorizedType& b, Op op) const {
    __m512 a0, a1, a2, a3;
    __m512 b0, b1, b2, b3;
    __m512 o0, o1, o2, o3;
    if constexpr (std::is_same_v<T, c10::Float8_e4m3fn>) {
      cvtfp8e4m3_fp32(_mm512_extracti32x4_epi32(values, 0), a0);
      cvtfp8e4m3_fp32(_mm512_extracti32x4_epi32(b.values, 0), b0);
      cvtfp8e4m3_fp32(_mm512_extracti32x4_epi32(values, 1), a1);
      cvtfp8e4m3_fp32(_mm512_extracti32x4_epi32(b.values, 1), b1);
      cvtfp8e4m3_fp32(_mm512_extracti32x4_epi32(values, 2), a2);
      cvtfp8e4m3_fp32(_mm512_extracti32x4_epi32(b.values, 2), b2);
      cvtfp8e4m3_fp32(_mm512_extracti32x4_epi32(values, 3), a3);
      cvtfp8e4m3_fp32(_mm512_extracti32x4_epi32(b.values, 3), b3);
    } else {
      cvtfp8e5m2_fp32(_mm512_extracti32x4_epi32(values, 0), a0);
      cvtfp8e5m2_fp32(_mm512_extracti32x4_epi32(b.values, 0), b0);
      cvtfp8e5m2_fp32(_mm512_extracti32x4_epi32(values, 1), a1);
      cvtfp8e5m2_fp32(_mm512_extracti32x4_epi32(b.values, 1), b1);
      cvtfp8e5m2_fp32(_mm512_extracti32x4_epi32(values, 2), a2);
      cvtfp8e5m2_fp32(_mm512_extracti32x4_epi32(b.values, 2), b2);
      cvtfp8e5m2_fp32(_mm512_extracti32x4_epi32(values, 3), a3);
      cvtfp8e5m2_fp32(_mm512_extracti32x4_epi32(b.values, 3), b3);
    }

    o0 = op(a0, b0);
    o1 = op(a1, b1);
    o2 = op(a2, b2);
    o3 = op(a3, b3);
    __m128i o128_0, o128_1, o128_2, o128_3;
    if constexpr (std::is_same_v<T, c10::Float8_e4m3fn>) {
      o128_0 = cvtfp32_fp8e4m3(o0);
      o128_1 = cvtfp32_fp8e4m3(o1);
      o128_2 = cvtfp32_fp8e4m3(o2);
      o128_3 = cvtfp32_fp8e4m3(o3);
    } else {
      o128_0 = cvtfp32_fp8e5m2(o0);
      o128_1 = cvtfp32_fp8e5m2(o1);
      o128_2 = cvtfp32_fp8e5m2(o2);
      o128_3 = cvtfp32_fp8e5m2(o3);
    }

    __m512i result = _mm512_setzero_si512();
    result = _mm512_inserti32x4(result, o128_0, 0);
    result = _mm512_inserti32x4(result, o128_1, 1);
    result = _mm512_inserti32x4(result, o128_2, 2);
    result = _mm512_inserti32x4(result, o128_3, 3);

    return result;
  }

 public:
  using value_type = uint8_t;
  using size_type = int;
  static constexpr size_type size() {
    return 64;
  }
  Vectorizedf8() {}
  Vectorizedf8(__m512i v) : values(v) {}
  Vectorizedf8(T val) {
    value_type uw = val.x;
    values = _mm512_set1_epi8(uw);
  }
  operator __m512i() const {
    return values;
  }
  T& operator[](int idx) = delete;
  const T& operator[](int idx) const = delete;
  static Vectorized<T> loadu(const void* ptr, int16_t count = size()) {
    if (count == size()) {
      return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
    } else if (count == 16) {
      // Fast path if only load element number of 16
      __m128i input_128 =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
      return _mm512_castsi128_si512(input_128);
    } else {
      __mmask64 mask = (1ULL << count) - 1;
      return _mm512_maskz_loadu_epi8(mask, ptr);
    }
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), values);
    } else if (count > 0) {
      if (count == 16) {
        // Fast path if only store element number of 16
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(ptr), _mm512_castsi512_si128(values));
      } else {
        __mmask64 mask = (1ULL << count) - 1;
        _mm512_mask_storeu_epi8(ptr, mask, values);
      }
    }
  }

  Vectorized<T> abs() const {
    return _mm512_andnot_si512(_mm512_set1_epi8(0x80), values);
  }

  Vectorized<T> inline operator==(const Vectorizedf8<T>& other) const {
    return binary_compare(other, [](__m512 x, __m512 y) {
      auto zero_vec = _mm512_set1_epi32(0);
      auto cmp = _mm512_cmp_ps_mask(x, y, _CMP_EQ_OQ);
      return _mm512_castsi512_ps(
          _mm512_mask_set1_epi32(zero_vec, cmp, 0xFFFFFFFF));
    });
  }

  Vectorized<T> inline operator!=(const Vectorizedf8<T>& other) const {
    return binary_compare(other, [](__m512 x, __m512 y) {
      auto zero_vec = _mm512_set1_epi32(0);
      auto cmp = _mm512_cmp_ps_mask(x, y, _CMP_NEQ_UQ);
      return _mm512_castsi512_ps(
          _mm512_mask_set1_epi32(zero_vec, cmp, 0xFFFFFFFF));
    });
  }

  Vectorized<T> inline operator>(const Vectorizedf8<T>& other) const {
    return binary_compare(other, [](__m512 x, __m512 y) {
      auto zero_vec = _mm512_set1_epi32(0);
      auto cmp = _mm512_cmp_ps_mask(x, y, _CMP_GT_OQ);
      return _mm512_castsi512_ps(
          _mm512_mask_set1_epi32(zero_vec, cmp, 0xFFFFFFFF));
    });
  }

  Vectorized<T> inline operator>=(const Vectorizedf8<T>& other) const {
    return binary_compare(other, [](__m512 x, __m512 y) {
      auto zero_vec = _mm512_set1_epi32(0);
      auto cmp = _mm512_cmp_ps_mask(x, y, _CMP_GE_OQ);
      return _mm512_castsi512_ps(
          _mm512_mask_set1_epi32(zero_vec, cmp, 0xFFFFFFFF));
    });
  }

  Vectorized<T> inline operator<(const Vectorizedf8<T>& other) const {
    return binary_compare(other, [](__m512 x, __m512 y) {
      auto zero_vec = _mm512_set1_epi32(0);
      auto cmp = _mm512_cmp_ps_mask(x, y, _CMP_LT_OQ);
      return _mm512_castsi512_ps(
          _mm512_mask_set1_epi32(zero_vec, cmp, 0xFFFFFFFF));
    });
  }

  Vectorized<T> inline operator<=(const Vectorizedf8<T>& other) const {
    return binary_compare(other, [](__m512 x, __m512 y) {
      auto zero_vec = _mm512_set1_epi32(0);
      auto cmp = _mm512_cmp_ps_mask(x, y, _CMP_LE_OQ);
      return _mm512_castsi512_ps(
          _mm512_mask_set1_epi32(zero_vec, cmp, 0xFFFFFFFF));
    });
  }
};

template <>
class Vectorized<Float8_e4m3fn> : public Vectorizedf8<Float8_e4m3fn> {
 public:
  using Vectorizedf8::Vectorizedf8;

  using value_type = Float8_e4m3fn;

  Vectorized<Float8_e4m3fn> eq(const Vectorized<Float8_e4m3fn>& other) const;
  Vectorized<Float8_e4m3fn> ne(const Vectorized<Float8_e4m3fn>& other) const;
  Vectorized<Float8_e4m3fn> gt(const Vectorized<Float8_e4m3fn>& other) const;
  Vectorized<Float8_e4m3fn> ge(const Vectorized<Float8_e4m3fn>& other) const;
  Vectorized<Float8_e4m3fn> lt(const Vectorized<Float8_e4m3fn>& other) const;
  Vectorized<Float8_e4m3fn> le(const Vectorized<Float8_e4m3fn>& other) const;
};

template <
    typename T,
    typename Op,
    std::enable_if_t<
        std::is_same_v<T, c10::Float8_e4m3fn> ||
            std::is_same_v<T, c10::Float8_e5m2>,
        int> = 0>
static inline Vectorized<T> binary_fp8_op_as_fp32(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    Op op) {
  __m512 a0, a1, a2, a3;
  __m512 b0, b1, b2, b3;
  __m512 o0, o1, o2, o3;
  if constexpr (std::is_same_v<T, c10::Float8_e4m3fn>) {
    cvtfp8e4m3_fp32(_mm512_extracti32x4_epi32(a, 0), a0);
    cvtfp8e4m3_fp32(_mm512_extracti32x4_epi32(b, 0), b0);
    cvtfp8e4m3_fp32(_mm512_extracti32x4_epi32(a, 1), a1);
    cvtfp8e4m3_fp32(_mm512_extracti32x4_epi32(b, 1), b1);
    cvtfp8e4m3_fp32(_mm512_extracti32x4_epi32(a, 2), a2);
    cvtfp8e4m3_fp32(_mm512_extracti32x4_epi32(b, 2), b2);
    cvtfp8e4m3_fp32(_mm512_extracti32x4_epi32(a, 3), a3);
    cvtfp8e4m3_fp32(_mm512_extracti32x4_epi32(b, 3), b3);
  } else {
    cvtfp8e5m2_fp32(_mm512_extracti32x4_epi32(a, 0), a0);
    cvtfp8e5m2_fp32(_mm512_extracti32x4_epi32(b, 0), b0);
    cvtfp8e5m2_fp32(_mm512_extracti32x4_epi32(a, 1), a1);
    cvtfp8e5m2_fp32(_mm512_extracti32x4_epi32(b, 1), b1);
    cvtfp8e5m2_fp32(_mm512_extracti32x4_epi32(a, 2), a2);
    cvtfp8e5m2_fp32(_mm512_extracti32x4_epi32(b, 2), b2);
    cvtfp8e5m2_fp32(_mm512_extracti32x4_epi32(a, 3), a3);
    cvtfp8e5m2_fp32(_mm512_extracti32x4_epi32(b, 3), b3);
  }
  o0 = op(a0, b0);
  o1 = op(a1, b1);
  o2 = op(a2, b2);
  o3 = op(a3, b3);

  __m128i o128_0, o128_1, o128_2, o128_3;
  if constexpr (std::is_same_v<T, c10::Float8_e4m3fn>) {
    o128_0 = cvtfp32_fp8e4m3(o0);
    o128_1 = cvtfp32_fp8e4m3(o1);
    o128_2 = cvtfp32_fp8e4m3(o2);
    o128_3 = cvtfp32_fp8e4m3(o3);
  } else {
    o128_0 = cvtfp32_fp8e5m2(o0);
    o128_1 = cvtfp32_fp8e5m2(o1);
    o128_2 = cvtfp32_fp8e5m2(o2);
    o128_3 = cvtfp32_fp8e5m2(o3);
  }

  __m512i result = _mm512_setzero_si512();
  result = _mm512_inserti32x4(result, o128_0, 0);
  result = _mm512_inserti32x4(result, o128_1, 1);
  result = _mm512_inserti32x4(result, o128_2, 2);
  result = _mm512_inserti32x4(result, o128_3, 3);

  return result;
}

// Refer to
// https://github.com/pytorch/pytorch/pull/153364#discussion_r2086509353 FP8 +,
// -, *, /, planed to be deleted in the future and here is just to make compiler
// happy
Vectorized<Float8_e4m3fn> inline operator+(
    const Vectorized<Float8_e4m3fn>& a,
    const Vectorized<Float8_e4m3fn>& b) {
  return binary_fp8_op_as_fp32(a, b, [](const __m512& x, const __m512& y) {
    return _mm512_add_ps(x, y);
  });
}

Vectorized<Float8_e4m3fn> inline operator-(
    const Vectorized<Float8_e4m3fn>& a,
    const Vectorized<Float8_e4m3fn>& b) {
  return binary_fp8_op_as_fp32(a, b, [](const __m512& x, const __m512& y) {
    return _mm512_sub_ps(x, y);
  });
}

Vectorized<Float8_e4m3fn> inline operator*(
    const Vectorized<Float8_e4m3fn>& a,
    const Vectorized<Float8_e4m3fn>& b) {
  return binary_fp8_op_as_fp32(a, b, [](const __m512& x, const __m512& y) {
    return _mm512_mul_ps(x, y);
  });
}

Vectorized<Float8_e4m3fn> inline operator/(
    const Vectorized<Float8_e4m3fn>& a,
    const Vectorized<Float8_e4m3fn>& b) {
  return binary_fp8_op_as_fp32(a, b, [](const __m512& x, const __m512& y) {
    return _mm512_div_ps(x, y);
  });
}

Vectorized<Float8_e4m3fn> inline operator&(
    const Vectorized<Float8_e4m3fn>& a,
    const Vectorized<Float8_e4m3fn>& b) {
  return _mm512_and_si512(a, b);
}

inline Vectorized<Float8_e4m3fn> Vectorized<Float8_e4m3fn>::eq(
    const Vectorized<Float8_e4m3fn>& other) const {
  return (*this == other) & Vectorized<Float8_e4m3fn>(1.0f);
}

inline Vectorized<Float8_e4m3fn> Vectorized<Float8_e4m3fn>::ne(
    const Vectorized<Float8_e4m3fn>& other) const {
  return (*this == other) & Vectorized<Float8_e4m3fn>(1.0f);
}

inline Vectorized<Float8_e4m3fn> Vectorized<Float8_e4m3fn>::gt(
    const Vectorized<Float8_e4m3fn>& other) const {
  return (*this > other) & Vectorized<Float8_e4m3fn>(1.0f);
}

inline Vectorized<Float8_e4m3fn> Vectorized<Float8_e4m3fn>::ge(
    const Vectorized<Float8_e4m3fn>& other) const {
  return (*this >= other) & Vectorized<Float8_e4m3fn>(1.0f);
}

inline Vectorized<Float8_e4m3fn> Vectorized<Float8_e4m3fn>::lt(
    const Vectorized<Float8_e4m3fn>& other) const {
  return (*this < other) & Vectorized<Float8_e4m3fn>(1.0f);
}

inline Vectorized<Float8_e4m3fn> Vectorized<Float8_e4m3fn>::le(
    const Vectorized<Float8_e4m3fn>& other) const {
  return (*this <= other) & Vectorized<Float8_e4m3fn>(1.0f);
}

template <>
class Vectorized<Float8_e5m2> : public Vectorizedf8<Float8_e5m2> {
 public:
  using Vectorizedf8::Vectorizedf8;

  using value_type = Float8_e5m2;

  Vectorized<Float8_e5m2> eq(const Vectorized<Float8_e5m2>& other) const;
  Vectorized<Float8_e5m2> ne(const Vectorized<Float8_e5m2>& other) const;
  Vectorized<Float8_e5m2> gt(const Vectorized<Float8_e5m2>& other) const;
  Vectorized<Float8_e5m2> ge(const Vectorized<Float8_e5m2>& other) const;
  Vectorized<Float8_e5m2> lt(const Vectorized<Float8_e5m2>& other) const;
  Vectorized<Float8_e5m2> le(const Vectorized<Float8_e5m2>& other) const;
};

// Refer to
// https://github.com/pytorch/pytorch/pull/153364#discussion_r2086509353 FP8 +,
// -, *, /, planed to be deleted in the future and here is just to make compiler
// happy
Vectorized<Float8_e5m2> inline operator+(
    const Vectorized<Float8_e5m2>& a,
    const Vectorized<Float8_e5m2>& b) {
  return binary_fp8_op_as_fp32(a, b, [](const __m512& x, const __m512& y) {
    return _mm512_add_ps(x, y);
  });
}

Vectorized<Float8_e5m2> inline operator-(
    const Vectorized<Float8_e5m2>& a,
    const Vectorized<Float8_e5m2>& b) {
  return binary_fp8_op_as_fp32(a, b, [](const __m512& x, const __m512& y) {
    return _mm512_sub_ps(x, y);
  });
}

Vectorized<Float8_e5m2> inline operator*(
    const Vectorized<Float8_e5m2>& a,
    const Vectorized<Float8_e5m2>& b) {
  return binary_fp8_op_as_fp32(a, b, [](const __m512& x, const __m512& y) {
    return _mm512_mul_ps(x, y);
  });
}

Vectorized<Float8_e5m2> inline operator/(
    const Vectorized<Float8_e5m2>& a,
    const Vectorized<Float8_e5m2>& b) {
  return binary_fp8_op_as_fp32(a, b, [](const __m512& x, const __m512& y) {
    return _mm512_div_ps(x, y);
  });
}

Vectorized<Float8_e5m2> inline operator&(
    const Vectorized<Float8_e5m2>& a,
    const Vectorized<Float8_e5m2>& b) {
  return _mm512_and_si512(a, b);
}

inline Vectorized<Float8_e5m2> Vectorized<Float8_e5m2>::eq(
    const Vectorized<Float8_e5m2>& other) const {
  return (*this == other) & Vectorized<Float8_e5m2>(1.0f);
}

inline Vectorized<Float8_e5m2> Vectorized<Float8_e5m2>::ne(
    const Vectorized<Float8_e5m2>& other) const {
  return (*this == other) & Vectorized<Float8_e5m2>(1.0f);
}

inline Vectorized<Float8_e5m2> Vectorized<Float8_e5m2>::gt(
    const Vectorized<Float8_e5m2>& other) const {
  return (*this > other) & Vectorized<Float8_e5m2>(1.0f);
}

inline Vectorized<Float8_e5m2> Vectorized<Float8_e5m2>::ge(
    const Vectorized<Float8_e5m2>& other) const {
  return (*this >= other) & Vectorized<Float8_e5m2>(1.0f);
}

inline Vectorized<Float8_e5m2> Vectorized<Float8_e5m2>::lt(
    const Vectorized<Float8_e5m2>& other) const {
  return (*this < other) & Vectorized<Float8_e5m2>(1.0f);
}

inline Vectorized<Float8_e5m2> Vectorized<Float8_e5m2>::le(
    const Vectorized<Float8_e5m2>& other) const {
  return (*this <= other) & Vectorized<Float8_e5m2>(1.0f);
}

#endif

} // namespace CPU_CAPABILITY
} // namespace at::vec
