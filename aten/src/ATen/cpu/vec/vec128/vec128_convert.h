#pragma once
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec_convert.h>

namespace at::vec {
inline namespace CPU_CAPABILITY {
#if (defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256))
template <typename src_t>
struct VecConvert<
    float,
    1,
    src_t,
    1,
    typename std::enable_if_t<is_8bit_integer_v<src_t>,
        void>> {
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
    typename std::enable_if_t<is_8bit_integer_v<src_t>,
        void>> {
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
    float32x4_t f32x4_0 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(u16_low1), 16));
    float32x4_t f32x4_1 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(u16_high1), 16));
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
    float32x4_t f32x4_0 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(u16_8), 16));
    result[0] = f32x4_0;
    return result;
  }
};

#endif // defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256)
} // namespace CPU_CAPABILITY
} // namespace at::vec
