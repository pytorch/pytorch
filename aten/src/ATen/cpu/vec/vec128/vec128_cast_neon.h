#pragma once
// Bit-preserving casts between the 128-bit NEON Vectorized specializations.

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec128/vec128_bfloat16_neon.h>
#include <ATen/cpu/vec/vec128/vec128_double_neon.h>
#include <ATen/cpu/vec/vec128/vec128_float_neon.h>
#include <ATen/cpu/vec/vec128/vec128_half_neon.h>
#include <ATen/cpu/vec/vec128/vec128_int_aarch64.h>
#include <ATen/cpu/vec/vec128/vec128_uint_aarch64.h>
#include <ATen/cpu/vec/vec_base.h>

namespace at::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {
#if (defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256))

// vreinterpretq is a register rename, so these replace the stack round trip
// that CastImpl in vec_base.h otherwise falls back to. Only the types with a
// single-register NEON specialization are covered; uint16_t/uint32_t/uint64_t
// have no Vectorized specialization here and keep the generic path.
#define CAST_TEMPLATE(dst_t, dst_prefix, src_t, src_prefix)                   \
  template <>                                                                 \
  inline Vectorized<dst_t> cast<dst_t, src_t>(const Vectorized<src_t>& src) { \
    return vreinterpretq_##dst_prefix##_##src_prefix(src);                    \
  }

// There is no vreinterpretq from a type to itself, so the diagonal is left to
// CastImpl<T, T>, which already returns the source unchanged.
#define CAST_TEMPLATE_BIDIRECTIONAL(t1_t, t1_prefix, t2_t, t2_prefix) \
  CAST_TEMPLATE(t1_t, t1_prefix, t2_t, t2_prefix)                     \
  CAST_TEMPLATE(t2_t, t2_prefix, t1_t, t1_prefix)

CAST_TEMPLATE_BIDIRECTIONAL(int8_t, s8, int16_t, s16)
CAST_TEMPLATE_BIDIRECTIONAL(int8_t, s8, int32_t, s32)
CAST_TEMPLATE_BIDIRECTIONAL(int8_t, s8, int64_t, s64)
CAST_TEMPLATE_BIDIRECTIONAL(int8_t, s8, uint8_t, u8)
CAST_TEMPLATE_BIDIRECTIONAL(int8_t, s8, float, f32)
CAST_TEMPLATE_BIDIRECTIONAL(int8_t, s8, double, f64)
CAST_TEMPLATE_BIDIRECTIONAL(int16_t, s16, int32_t, s32)
CAST_TEMPLATE_BIDIRECTIONAL(int16_t, s16, int64_t, s64)
CAST_TEMPLATE_BIDIRECTIONAL(int16_t, s16, uint8_t, u8)
CAST_TEMPLATE_BIDIRECTIONAL(int16_t, s16, float, f32)
CAST_TEMPLATE_BIDIRECTIONAL(int16_t, s16, double, f64)
CAST_TEMPLATE_BIDIRECTIONAL(int32_t, s32, int64_t, s64)
CAST_TEMPLATE_BIDIRECTIONAL(int32_t, s32, uint8_t, u8)
CAST_TEMPLATE_BIDIRECTIONAL(int32_t, s32, float, f32)
CAST_TEMPLATE_BIDIRECTIONAL(int32_t, s32, double, f64)
CAST_TEMPLATE_BIDIRECTIONAL(int64_t, s64, uint8_t, u8)
CAST_TEMPLATE_BIDIRECTIONAL(int64_t, s64, float, f32)
CAST_TEMPLATE_BIDIRECTIONAL(int64_t, s64, double, f64)
CAST_TEMPLATE_BIDIRECTIONAL(uint8_t, u8, float, f32)
CAST_TEMPLATE_BIDIRECTIONAL(uint8_t, u8, double, f64)
CAST_TEMPLATE_BIDIRECTIONAL(float, f32, double, f64)

// bf16/fp16 vec classes are not available for C10_MOBILE
#if !defined(C10_MOBILE)

CAST_TEMPLATE_BIDIRECTIONAL(c10::Half, f16, int8_t, s8)
CAST_TEMPLATE_BIDIRECTIONAL(c10::Half, f16, int16_t, s16)
CAST_TEMPLATE_BIDIRECTIONAL(c10::Half, f16, int32_t, s32)
CAST_TEMPLATE_BIDIRECTIONAL(c10::Half, f16, int64_t, s64)
CAST_TEMPLATE_BIDIRECTIONAL(c10::Half, f16, uint8_t, u8)
CAST_TEMPLATE_BIDIRECTIONAL(c10::Half, f16, float, f32)
CAST_TEMPLATE_BIDIRECTIONAL(c10::Half, f16, double, f64)

// bfloat16x8_t only exists on a target with BF16 support; at_bfloat16x8_t is
// uint16x8_t otherwise. Routing through u16 lets the existing shim collapse
// the second hop to nothing on whichever of the two the target provides.
#define CAST_BF16_TEMPLATE(other_t, other_prefix)                            \
  template <>                                                                \
  inline Vectorized<c10::BFloat16> cast<c10::BFloat16, other_t>(             \
      const Vectorized<other_t>& src) {                                      \
    return at_vreinterpretq_bf16_u16(vreinterpretq_u16_##other_prefix(src)); \
  }                                                                          \
  template <>                                                                \
  inline Vectorized<other_t> cast<other_t, c10::BFloat16>(                   \
      const Vectorized<c10::BFloat16>& src) {                                \
    return vreinterpretq_##other_prefix##_u16(                               \
        at_vreinterpretq_u16_bf16(static_cast<at_bfloat16x8_t>(src)));       \
  }

CAST_BF16_TEMPLATE(int8_t, s8)
CAST_BF16_TEMPLATE(int16_t, s16)
CAST_BF16_TEMPLATE(int32_t, s32)
CAST_BF16_TEMPLATE(int64_t, s64)
CAST_BF16_TEMPLATE(uint8_t, u8)
CAST_BF16_TEMPLATE(float, f32)
CAST_BF16_TEMPLATE(double, f64)
CAST_BF16_TEMPLATE(c10::Half, f16)

#endif // !defined(C10_MOBILE)

#endif // defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256)
} // namespace CPU_CAPABILITY
} // namespace at::vec
