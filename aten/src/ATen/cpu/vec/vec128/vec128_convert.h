#pragma once
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec_convert.h>

namespace at::vec {
inline namespace CPU_CAPABILITY {
#if (defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256))

// Enable auto-vectorization for clang-17+
// GCC-12 has a bug: gcc.gnu.org/bugzilla/show_bug.cgi?id=117001
#if defined(__clang__) && (__clang_major__ >= 17)

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

template <typename to_type>
inline void convertFromBool(
    const bool* __restrict src,
    to_type* __restrict dst,
    int64_t n) {
  const uint8_t* srcPtr = reinterpret_cast<const uint8_t*>(src);
  uint64_t len = static_cast<uint64_t>(n);
  for (uint64_t i = 0; i < len; i++) {
    dst[i] = srcPtr[i] != 0 ? static_cast<to_type>(1) : static_cast<to_type>(0);
  }
}

template <typename from_type>
inline void convertToBool(
    const from_type* __restrict src,
    bool* __restrict dst,
    int64_t n) {
  uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dst);
  uint64_t len = static_cast<uint64_t>(n);
  for (uint64_t i = 0; i < len; i++) {
    dstPtr[i] = src[i] != static_cast<from_type>(0) ? 1 : 0;
  }
}

#define CONVERT_TEMPLATE(from_type, to_type)                           \
  template <>                                                          \
  inline void convert(const from_type* src, to_type* dst, int64_t n) { \
    return convertImpl<from_type, to_type>(src, dst, n);               \
  }

#define CONVERT_FROM_BOOL_TEMPLATE(to_type)                       \
  inline void convert(const bool* src, to_type* dst, int64_t n) { \
    return convertFromBool<to_type>(src, dst, n);                 \
  }

#define CONVERT_TO_BOOL_TEMPLATE(from_type)                         \
  inline void convert(const from_type* src, bool* dst, int64_t n) { \
    return convertToBool<from_type>(src, dst, n);                   \
  }

CONVERT_TEMPLATE(uint8_t, uint8_t)
CONVERT_TEMPLATE(uint8_t, int8_t)
CONVERT_TEMPLATE(uint8_t, int16_t)
CONVERT_TEMPLATE(uint8_t, int32_t)
CONVERT_TEMPLATE(uint8_t, int64_t)
CONVERT_TEMPLATE(uint8_t, float)
CONVERT_TEMPLATE(uint8_t, double)
CONVERT_TO_BOOL_TEMPLATE(uint8_t)
CONVERT_TEMPLATE(int8_t, uint8_t)
CONVERT_TEMPLATE(int8_t, int8_t)
CONVERT_TEMPLATE(int8_t, int16_t)
CONVERT_TEMPLATE(int8_t, int32_t)
CONVERT_TEMPLATE(int8_t, int64_t)
CONVERT_TEMPLATE(int8_t, float)
CONVERT_TEMPLATE(int8_t, double)
CONVERT_TO_BOOL_TEMPLATE(int8_t)
CONVERT_TEMPLATE(int16_t, uint8_t)
CONVERT_TEMPLATE(int16_t, int8_t)
CONVERT_TEMPLATE(int16_t, int16_t)
CONVERT_TEMPLATE(int16_t, int32_t)
CONVERT_TEMPLATE(int16_t, int64_t)
CONVERT_TEMPLATE(int16_t, float)
CONVERT_TEMPLATE(int16_t, double)
CONVERT_TO_BOOL_TEMPLATE(int16_t)
CONVERT_TEMPLATE(int32_t, uint8_t)
CONVERT_TEMPLATE(int32_t, int8_t)
CONVERT_TEMPLATE(int32_t, int16_t)
CONVERT_TEMPLATE(int32_t, int32_t)
CONVERT_TEMPLATE(int32_t, int64_t)
CONVERT_TEMPLATE(int32_t, float)
CONVERT_TEMPLATE(int32_t, double)
CONVERT_TO_BOOL_TEMPLATE(int32_t)
CONVERT_TEMPLATE(int64_t, uint8_t)
CONVERT_TEMPLATE(int64_t, int8_t)
CONVERT_TEMPLATE(int64_t, int16_t)
CONVERT_TEMPLATE(int64_t, int32_t)
CONVERT_TEMPLATE(int64_t, int64_t)
CONVERT_TEMPLATE(int64_t, float)
CONVERT_TEMPLATE(int64_t, double)
CONVERT_TO_BOOL_TEMPLATE(int64_t)
CONVERT_TEMPLATE(float, uint8_t)
CONVERT_TEMPLATE(float, int8_t)
CONVERT_TEMPLATE(float, int16_t)
CONVERT_TEMPLATE(float, int32_t)
CONVERT_TEMPLATE(float, int64_t)
CONVERT_TEMPLATE(float, float)
CONVERT_TEMPLATE(float, double)
CONVERT_TO_BOOL_TEMPLATE(float)
CONVERT_TEMPLATE(double, uint8_t)
CONVERT_TEMPLATE(double, int8_t)
CONVERT_TEMPLATE(double, int16_t)
CONVERT_TEMPLATE(double, int32_t)
CONVERT_TEMPLATE(double, int64_t)
CONVERT_TEMPLATE(double, float)
CONVERT_TEMPLATE(double, double)
CONVERT_TO_BOOL_TEMPLATE(double)
CONVERT_FROM_BOOL_TEMPLATE(uint8_t)
CONVERT_FROM_BOOL_TEMPLATE(int8_t)
CONVERT_FROM_BOOL_TEMPLATE(int16_t)
CONVERT_FROM_BOOL_TEMPLATE(int32_t)
CONVERT_FROM_BOOL_TEMPLATE(int64_t)
CONVERT_FROM_BOOL_TEMPLATE(float)
CONVERT_FROM_BOOL_TEMPLATE(double)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#define CONVERT_FROM_FP16_TEMPLATE(to_type)                            \
  template <>                                                          \
  inline void convert(const at::Half* src, to_type* dst, int64_t n) {  \
    const float16_t* srcPtr = reinterpret_cast<const float16_t*>(src); \
    return convertImpl<float16_t, to_type>(srcPtr, dst, n);            \
  }

#define CONVERT_TO_FP16_TEMPLATE(from_type)                             \
  template <>                                                           \
  inline void convert(const from_type* src, at::Half* dst, int64_t n) { \
    float16_t* dstPtr = reinterpret_cast<float16_t*>(dst);              \
    return convertImpl<from_type, float16_t>(src, dstPtr, n);           \
  }

CONVERT_FROM_FP16_TEMPLATE(uint8_t)
CONVERT_FROM_FP16_TEMPLATE(int8_t)
CONVERT_FROM_FP16_TEMPLATE(int16_t)
CONVERT_FROM_FP16_TEMPLATE(int32_t)
CONVERT_FROM_FP16_TEMPLATE(int64_t)
CONVERT_FROM_FP16_TEMPLATE(float16_t)
CONVERT_FROM_FP16_TEMPLATE(float)
CONVERT_FROM_FP16_TEMPLATE(double)
CONVERT_TO_FP16_TEMPLATE(uint8_t)
CONVERT_TO_FP16_TEMPLATE(int8_t)
CONVERT_TO_FP16_TEMPLATE(int16_t)
CONVERT_TO_FP16_TEMPLATE(int32_t)
CONVERT_TO_FP16_TEMPLATE(int64_t)
CONVERT_TO_FP16_TEMPLATE(float)
CONVERT_TO_FP16_TEMPLATE(double)

inline void convertBoolToFp16Impl(
    const bool* __restrict src,
    at::Half* __restrict dst,
    int64_t n) {
  const uint8_t* srcPtr = reinterpret_cast<const uint8_t*>(src);
  float16_t* dstPtr = reinterpret_cast<float16_t*>(dst);
  uint64_t len = static_cast<uint64_t>(n);
  for (uint64_t i = 0; i < len; i++) {
    dstPtr[i] = srcPtr[i] != 0 ? 1.0 : 0;
  }
}

template <>
inline void convert(const bool* src, at::Half* dst, int64_t n) {
  return convertBoolToFp16Impl(src, dst, n);
}

inline void convertFp16ToBoolImpl(
    const at::Half* __restrict src,
    bool* __restrict dst,
    int64_t n) {
  const float16_t* srcPtr = reinterpret_cast<const float16_t*>(src);
  uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dst);
  uint64_t len = static_cast<uint64_t>(n);
  for (uint64_t i = 0; i < len; i++) {
    dstPtr[i] = srcPtr[i] != 0.0 ? 1 : 0;
  }
}

template <>
inline void convert(const at::Half* src, bool* dst, int64_t n) {
  return convertFp16ToBoolImpl(src, dst, n);
}

#endif

template <typename to_type>
inline void convertFromBf16Impl(
    const c10::BFloat16* __restrict src,
    to_type* __restrict dst,
    int64_t n) {
  const uint16_t* srcPtr = reinterpret_cast<const uint16_t*>(src);
  uint64_t len = static_cast<uint64_t>(n);
  for (uint64_t i = 0; i < len; i++) {
    uint32_t tmp = static_cast<uint32_t>(srcPtr[i]) << 16;
    float tmpF;
    __builtin_memcpy(&tmpF, &tmp, sizeof(float));
    dst[i] = static_cast<to_type>(tmpF);
  }
}
#define CONVERT_FROM_BF16_TEMPLATE(to_type)                                \
  template <>                                                              \
  inline void convert(const c10::BFloat16* src, to_type* dst, int64_t n) { \
    return convertFromBf16Impl<to_type>(src, dst, n);                      \
  }

CONVERT_FROM_BF16_TEMPLATE(uint8_t)
CONVERT_FROM_BF16_TEMPLATE(int8_t)
CONVERT_FROM_BF16_TEMPLATE(int16_t)
CONVERT_FROM_BF16_TEMPLATE(int32_t)
CONVERT_FROM_BF16_TEMPLATE(int64_t)
CONVERT_FROM_BF16_TEMPLATE(float)
CONVERT_FROM_BF16_TEMPLATE(double)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
CONVERT_FROM_BF16_TEMPLATE(float16_t)
#endif

#ifdef __ARM_FEATURE_BF16

// clang-[17, 20] crashes when autovectorizing static cast to bf16
// Below is a workaround to have some vectorization
// Works decently well for smaller int types
template <typename from_type>
inline void convertToBf16Impl(
    const from_type* __restrict src,
    c10::BFloat16* __restrict dst,
    uint64_t n) {
  bfloat16_t* dstPtr = reinterpret_cast<bfloat16_t*>(dst);
  uint64_t loopBound = n - (n % 16);
  uint64_t i = 0;
  for (; i < loopBound; i += 16) {
    float32x4_t a, b, c, d;
    a[0] = static_cast<float>(src[i]);
    a[1] = static_cast<float>(src[i + 1]);
    a[2] = static_cast<float>(src[i + 2]);
    a[3] = static_cast<float>(src[i + 3]);
    b[0] = static_cast<float>(src[i + 4]);
    b[1] = static_cast<float>(src[i + 5]);
    b[2] = static_cast<float>(src[i + 6]);
    b[3] = static_cast<float>(src[i + 7]);
    c[0] = static_cast<float>(src[i + 8]);
    c[1] = static_cast<float>(src[i + 9]);
    c[2] = static_cast<float>(src[i + 10]);
    c[3] = static_cast<float>(src[i + 11]);
    d[0] = static_cast<float>(src[i + 12]);
    d[1] = static_cast<float>(src[i + 13]);
    d[2] = static_cast<float>(src[i + 14]);
    d[3] = static_cast<float>(src[i + 15]);

    vst1q_bf16(dstPtr + i, vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(a), b));
    vst1q_bf16(dstPtr + i + 8, vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(c), d));
  }

#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
  for (; i < n; i++) {
    float a = static_cast<float>(src[i]);
    dstPtr[i] = vcvth_bf16_f32(a);
  }
}

#define CONVERT_TO_BF16_TEMPLATE(from_type)                                  \
  template <>                                                                \
  inline void convert(const from_type* src, c10::BFloat16* dst, int64_t n) { \
    return convertToBf16Impl<from_type>(src, dst, n);                        \
  }

CONVERT_TO_BF16_TEMPLATE(uint8_t)
CONVERT_TO_BF16_TEMPLATE(int8_t)
CONVERT_TO_BF16_TEMPLATE(int16_t)
CONVERT_TO_BF16_TEMPLATE(int32_t)

#endif

inline void convertBoolToBfloat16Impl(
    const bool* __restrict src,
    c10::BFloat16* __restrict dst,
    int64_t n) {
  const uint8_t* srcPtr = reinterpret_cast<const uint8_t*>(src);
  uint16_t* dstPtr = reinterpret_cast<uint16_t*>(dst);
  uint64_t len = static_cast<uint64_t>(n);
  constexpr uint16_t kBf16One = 0x3f80; // 1.0 in bfloat16
  for (uint64_t i = 0; i < len; i++) {
    dstPtr[i] = srcPtr[i] != 0 ? kBf16One : 0;
  }
}

template <>
inline void convert(const bool* src, c10::BFloat16* dst, int64_t n) {
  return convertBoolToBfloat16Impl(src, dst, n);
}

inline void convertBfloat16ToBoolImpl(
    const c10::BFloat16* __restrict src,
    bool* __restrict dst,
    int64_t n) {
  uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dst);
  const uint16_t* srcPtr = reinterpret_cast<const uint16_t*>(src);
  uint64_t len = static_cast<uint64_t>(n);
  for (uint64_t i = 0; i < len; i++) {
    // Check if all non-sign bits are 0
    bool isBf16Zero = (srcPtr[i] & 0x7fff) == 0;
    dstPtr[i] = isBf16Zero ? 0 : 1;
  }
}

template <>
inline void convert(const c10::BFloat16* src, bool* dst, int64_t n) {
  return convertBfloat16ToBoolImpl(src, dst, n);
}

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
