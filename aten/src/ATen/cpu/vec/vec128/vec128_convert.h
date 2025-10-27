#pragma once
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec_convert.h>

namespace at::vec {
inline namespace CPU_CAPABILITY {
#if (defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256))

// Enable auto-vectorization for GCC-13+ and clang-17+
// GCC-12 has a bug: gcc.gnu.org/bugzilla/show_bug.cgi?id=117001
#if __GNUC__ > 12 || (defined(__clang__) && (__clang_major__ >= 17))

template <typename from_type, typename to_type>
inline void convertImpl(
    const from_type* __restrict src,
    to_type* __restrict dst,
    int64_t n) {
  uint64_t len = static_cast<uint64_t>(n);
  for (uint64_t i = 0; i < len; i++) {
    dst[i] = static_cast<to_type>(src[i]);
  }
}

#define CONVERT_TEMPLATE(from_type, to_type)                           \
  template <>                                                          \
  inline void convert(const from_type* src, to_type* dst, int64_t n) { \
    return convertImpl<from_type, to_type>(src, dst, n);               \
  }

CONVERT_TEMPLATE(uint8_t, uint8_t)
CONVERT_TEMPLATE(uint8_t, int8_t)
CONVERT_TEMPLATE(uint8_t, int16_t)
CONVERT_TEMPLATE(uint8_t, int32_t)
CONVERT_TEMPLATE(uint8_t, int64_t)
CONVERT_TEMPLATE(uint8_t, float)
CONVERT_TEMPLATE(uint8_t, double)
CONVERT_TEMPLATE(int8_t, uint8_t)
CONVERT_TEMPLATE(int8_t, int8_t)
CONVERT_TEMPLATE(int8_t, int16_t)
CONVERT_TEMPLATE(int8_t, int32_t)
CONVERT_TEMPLATE(int8_t, int64_t)
CONVERT_TEMPLATE(int8_t, float)
CONVERT_TEMPLATE(int8_t, double)
CONVERT_TEMPLATE(int16_t, uint8_t)
CONVERT_TEMPLATE(int16_t, int8_t)
CONVERT_TEMPLATE(int16_t, int16_t)
CONVERT_TEMPLATE(int16_t, int32_t)
CONVERT_TEMPLATE(int16_t, int64_t)
CONVERT_TEMPLATE(int16_t, float)
CONVERT_TEMPLATE(int16_t, double)
CONVERT_TEMPLATE(int32_t, uint8_t)
CONVERT_TEMPLATE(int32_t, int8_t)
CONVERT_TEMPLATE(int32_t, int16_t)
CONVERT_TEMPLATE(int32_t, int32_t)
CONVERT_TEMPLATE(int32_t, int64_t)
CONVERT_TEMPLATE(int32_t, float)
CONVERT_TEMPLATE(int32_t, double)
CONVERT_TEMPLATE(int64_t, uint8_t)
CONVERT_TEMPLATE(int64_t, int8_t)
CONVERT_TEMPLATE(int64_t, int16_t)
CONVERT_TEMPLATE(int64_t, int32_t)
CONVERT_TEMPLATE(int64_t, int64_t)
CONVERT_TEMPLATE(int64_t, float)
CONVERT_TEMPLATE(int64_t, double)
CONVERT_TEMPLATE(float, uint8_t)
CONVERT_TEMPLATE(float, int8_t)
CONVERT_TEMPLATE(float, int16_t)
CONVERT_TEMPLATE(float, int32_t)
CONVERT_TEMPLATE(float, int64_t)
CONVERT_TEMPLATE(float, float)
CONVERT_TEMPLATE(float, double)
CONVERT_TEMPLATE(double, uint8_t)
CONVERT_TEMPLATE(double, int8_t)
CONVERT_TEMPLATE(double, int16_t)
CONVERT_TEMPLATE(double, int32_t)
CONVERT_TEMPLATE(double, int64_t)
CONVERT_TEMPLATE(double, float)
CONVERT_TEMPLATE(double, double)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
CONVERT_TEMPLATE(float16_t, uint8_t)
CONVERT_TEMPLATE(float16_t, int8_t)
CONVERT_TEMPLATE(float16_t, int16_t)
CONVERT_TEMPLATE(float16_t, int32_t)
CONVERT_TEMPLATE(float16_t, int64_t)
CONVERT_TEMPLATE(float16_t, float16_t)
CONVERT_TEMPLATE(float16_t, float)
CONVERT_TEMPLATE(float16_t, double)
CONVERT_TEMPLATE(uint8_t, float16_t)
CONVERT_TEMPLATE(int8_t, float16_t)
CONVERT_TEMPLATE(int16_t, float16_t)
CONVERT_TEMPLATE(int32_t, float16_t)
CONVERT_TEMPLATE(int64_t, float16_t)
CONVERT_TEMPLATE(float, float16_t)
CONVERT_TEMPLATE(double, float16_t)
#endif
#ifdef __ARM_FEATURE_BF16
CONVERT_TEMPLATE(bfloat16_t, uint8_t)
CONVERT_TEMPLATE(bfloat16_t, int8_t)
CONVERT_TEMPLATE(bfloat16_t, int16_t)
CONVERT_TEMPLATE(bfloat16_t, int32_t)
CONVERT_TEMPLATE(bfloat16_t, int64_t)
CONVERT_TEMPLATE(bfloat16_t, bfloat16_t)
CONVERT_TEMPLATE(bfloat16_t, float)
CONVERT_TEMPLATE(bfloat16_t, double)
CONVERT_TEMPLATE(uint8_t, bfloat16_t)
CONVERT_TEMPLATE(int8_t, bfloat16_t)
CONVERT_TEMPLATE(int16_t, bfloat16_t)
CONVERT_TEMPLATE(int32_t, bfloat16_t)
CONVERT_TEMPLATE(int64_t, bfloat16_t)
CONVERT_TEMPLATE(float, bfloat16_t)
CONVERT_TEMPLATE(double, bfloat16_t)
#endif

#endif

template <typename src_t>
struct VecConvert<
    float,
    1,
    src_t,
    1,
    typename std::enable_if_t<is_8bit_integer_v<src_t>, void>> {
  static inline VectorizedN<float, 1> apply(const VectorizedN<src_t, 1>& src) {
    return convert_int8_half_register_to_float(src[0]);
  }
};
template <typename src_t>
struct VecConvert<
    float,
    2,
    src_t,
    1,
    typename std::enable_if_t<is_8bit_integer_v<src_t>, void>> {
  static inline VectorizedN<float, 2> apply(const VectorizedN<src_t, 1>& src) {
    const auto [v0, v1] = convert_int8_to_float(src[0]);
    return VectorizedN<float, 2>(v0, v1);
  }
};

template <>
struct VecConvert<float, 2, BFloat16, 1> {
  static inline VectorizedN<float, 2> apply(
      const VectorizedN<BFloat16, 1>& src) {
    VectorizedN<float, 2> result;
    uint16x8_t u16_8 = vld1q_u16(reinterpret_cast<const uint16_t*>(&src[0]));
    auto u16_low1 = vget_low_u16(u16_8);
    auto u16_high1 = vget_high_u16(u16_8);
    float32x4_t f32x4_0 =
        vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(u16_low1), 16));
    float32x4_t f32x4_1 =
        vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(u16_high1), 16));
    result[0] = f32x4_0;
    result[1] = f32x4_1;
    return result;
  }
};
// Half register to full register.
template <>
struct VecConvert<float, 1, BFloat16, 1> {
  static inline VectorizedN<float, 1> apply(
      const VectorizedN<BFloat16, 1>& src) {
    VectorizedN<float, 1> result;
    uint16x4_t u16_8 = vld1_u16(reinterpret_cast<const uint16_t*>(&src[0]));
    float32x4_t f32x4_0 =
        vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(u16_8), 16));
    result[0] = f32x4_0;
    return result;
  }
};

#endif // defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256)
} // namespace CPU_CAPABILITY
} // namespace at::vec
