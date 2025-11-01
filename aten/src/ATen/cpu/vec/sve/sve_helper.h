#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <c10/macros/Macros.h>
#include <cstdint>

#include <ATen/cpu/vec/vec_base.h>

#if defined(__aarch64__) &&                     \
    (defined(AT_BUILD_ARM_VEC256_WITH_SLEEF) || \
     defined(AT_BUILD_ARM_VECSVE_WITH_SLEEF))
#define SLEEF_STATIC_LIBS
#include <sleef.h>
#define USE_SLEEF(sleef_code, non_sleef_code) sleef_code
#else
#define USE_SLEEF(sleef_code, non_sleef_code) non_sleef_code
#endif

#if defined(CPU_CAPABILITY_SVE)

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

#endif // defined(CPU_CAPABILITY_SVE)
