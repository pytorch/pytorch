#pragma once

#include <ATen/cpu/vec/intrinsics.h>

#include <ATen/cpu/vec/vec_base.h>

#if defined(CPU_CAPABILITY_SVE256)

// Define the data type of VLS(vector-length specific).
typedef svbool_t vls_pred_t
    __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svint8_t vls_int8_t
    __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svint16_t vls_int16_t
    __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svint32_t vls_int32_t
    __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svint64_t vls_int64_t
    __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svuint8_t vls_uint8_t
    __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svuint16_t vls_uint16_t
    __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svuint32_t vls_uint32_t
    __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svuint64_t vls_uint64_t
    __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svfloat16_t vls_float16_t
    __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svbfloat16_t vls_bfloat16_t
    __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svfloat32_t vls_float32_t
    __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svfloat64_t vls_float64_t
    __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));

#define ptrue svptrue_b8()
#define ZERO_S8 svdup_n_s8(0)
#define ZERO_S16 svdup_n_s16(0)
#define ZERO_S32 svdup_n_s32(0)
#define ZERO_S64 svdup_n_s64(0)
#define ZERO_U8 svdup_n_u8(0)
#define ZERO_U16 svdup_n_u16(0)
#define ZERO_U32 svdup_n_u32(0)
#define ZERO_U64 svdup_n_u64(0)
#define ZERO_F16 svdup_n_f16(0.f)
#define ZERO_F32 svdup_n_f32(0.f)
#define ZERO_F64 svdup_n_f64(0.0)
#define ONE_S8 svdup_n_s8(1)
#define ONE_S16 svdup_n_s16(1)
#define ONE_S32 svdup_n_s32(1)
#define ONE_S64 svdup_n_s64(1)
#define ONE_U8 svdup_n_u8(1)
#define ONE_U16 svdup_n_u16(1)
#define ONE_U32 svdup_n_u32(1)
#define ONE_U64 svdup_n_u64(1)
#define ONE_F16 svdup_n_f16(1.f)
#define ONE_BF16 svdup_n_bf16(1.f)
#define ONE_F32 svdup_n_f32(1.f)
#define ONE_F64 svdup_n_f64(1.0)
#define ALL_S8_TRUE_MASK svdup_n_s8(0xff)
#define ALL_S8_FALSE_MASK svdup_n_s8(0x0)
#define ALL_S16_TRUE_MASK svdup_n_s16(0xffff)
#define ALL_S16_FALSE_MASK svdup_n_s16(0x0)
#define ALL_S32_TRUE_MASK svdup_n_s32(0xffffffff)
#define ALL_S32_FALSE_MASK svdup_n_s32(0x0)
#define ALL_S64_TRUE_MASK svdup_n_s64(0xffffffffffffffff)
#define ALL_S64_FALSE_MASK svdup_n_s64(0x0)
#define ALL_U8_TRUE_MASK svdup_n_u8(0x01)
#define ALL_U8_FALSE_MASK svdup_n_u8(0x00)
#define ALL_F16_TRUE_MASK svreinterpret_f16_s16(ALL_S16_TRUE_MASK)
#define ALL_F16_FALSE_MASK svreinterpret_f16_s16(ALL_S16_FALSE_MASK)
#define ALL_BF16_TRUE_MASK svreinterpret_bf16_s16(ALL_S16_TRUE_MASK)
#define ALL_BF16_FALSE_MASK svreinterpret_bf16_s16(ALL_S16_FALSE_MASK)
#define ALL_F32_TRUE_MASK svreinterpret_f32_s32(ALL_S32_TRUE_MASK)
#define ALL_F32_FALSE_MASK svreinterpret_f32_s32(ALL_S32_FALSE_MASK)
#define ALL_F64_TRUE_MASK svreinterpret_f64_s64(ALL_S64_TRUE_MASK)
#define ALL_F64_FALSE_MASK svreinterpret_f64_s64(ALL_S64_FALSE_MASK)

namespace at::vec {
inline namespace CPU_CAPABILITY {

// VecBool<T> on SVE: a predicate with one element per lane of Vectorized<T>.
// It is a class only so that it can have operators, which svbool_t does not
// get on every compiler; it converts to and from svbool_t implicitly.
template <typename T>
struct SveVecBool {
  vls_pred_t p;

  SveVecBool(svbool_t p) : p(p) {}
  operator svbool_t() const {
    return p;
  }
  // The lanes of the matching Vectorized<T>: ~ must not set the bits of a
  // predicate that do not belong to any lane.
  static svbool_t all() {
    if constexpr (sizeof(T) == 1) {
      return svptrue_b8();
    } else if constexpr (sizeof(T) == 2) {
      return svptrue_b16();
    } else if constexpr (sizeof(T) == 4) {
      return svptrue_b32();
    } else {
      return svptrue_b64();
    }
  }
  // Bit i is set when lane i is false. The false lanes are packed to one
  // predicate bit per byte and expanded to 0/1 bytes. Multiplying each 64-bit
  // element by 0x0102040810204080 gathers its 8 bytes into its top byte, and
  // the table lookup moves those top bytes into the low 64 bits.
  int64_t zero_mask() const {
    static_assert(VECTOR_WIDTH <= 64);
    svbool_t z = ~*this;
    if constexpr (sizeof(T) == 8) {
      z = svuzp1_b32(z, svpfalse_b());
    }
    if constexpr (sizeof(T) >= 4) {
      z = svuzp1_b16(z, svpfalse_b());
    }
    if constexpr (sizeof(T) >= 2) {
      z = svuzp1_b8(z, svpfalse_b());
    }
    const svuint64_t gathered = svmul_n_u64_x(
        svptrue_b64(),
        svreinterpret_u64_u8(svdup_n_u8_z(z, 1)),
        0x0102040810204080);
    const vls_uint64_t packed = svreinterpret_u64_u8(
        svtbl_u8(svreinterpret_u8_u64(gathered), svindex_u8(7, 8)));
    return packed[0];
  }
};

template <typename T>
inline SveVecBool<T> operator&(SveVecBool<T> a, SveVecBool<T> b) {
  return svand_b_z(SveVecBool<T>::all(), a, b);
}
template <typename T>
inline SveVecBool<T> operator|(SveVecBool<T> a, SveVecBool<T> b) {
  return svorr_b_z(SveVecBool<T>::all(), a, b);
}
template <typename T>
inline SveVecBool<T> operator^(SveVecBool<T> a, SveVecBool<T> b) {
  return sveor_b_z(SveVecBool<T>::all(), a, b);
}
template <typename T>
inline SveVecBool<T> operator~(SveVecBool<T> a) {
  return svnot_b_z(SveVecBool<T>::all(), a);
}

} // namespace CPU_CAPABILITY
} // namespace at::vec

#endif // defined(CPU_CAPABILITY_SVE256)
