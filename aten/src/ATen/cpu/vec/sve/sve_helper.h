#pragma once

#include <ATen/cpu/vec/intrinsics.h>

#include <ATen/cpu/vec/vec_base.h>

#if defined(CPU_CAPABILITY_SVE)
template<typename T>
inline uint64_t svcnt();
template<>
inline uint64_t svcnt<int8_t>() { return svcntb(); }
template<>
inline uint64_t svcnt<int16_t>() { return svcnth(); }
template<>
inline uint64_t svcnt<int32_t>() { return svcntw(); }
template<>
inline uint64_t svcnt<int64_t>() { return svcntd(); }
template<>
inline uint64_t svcnt<uint8_t>() { return svcntb(); }
template<>
inline uint64_t svcnt<uint16_t>() { return svcnth(); }
template<>
inline uint64_t svcnt<uint32_t>() { return svcntw(); }
template<>
inline uint64_t svcnt<uint64_t>() { return svcntd(); }
template<>
inline uint64_t svcnt<float16_t>() { return svcnth(); }
template<>
inline uint64_t svcnt<float>() { return svcntw(); }
template<>
inline uint64_t svcnt<double>() { return svcntd(); }

// Define the data type of VLS(vector-length specific).
typedef svbool_t vls_pred_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svint8_t vls_int8_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svint16_t vls_int16_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svint32_t vls_int32_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svint64_t vls_int64_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svuint8_t vls_uint8_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svuint16_t vls_uint16_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svuint32_t vls_uint32_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svuint64_t vls_uint64_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svfloat16_t vls_float16_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svfloat32_t vls_float32_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));
typedef svfloat64_t vls_float64_t __attribute__((arm_sve_vector_bits(VECTOR_WIDTH * 8)));

// Define all true predicate.
const vls_pred_t const_ptrue = svptrue_b8();

// Define constant for the value of zero.
const vls_int8_t const_zero_s8 = svdup_n_s8(0);
const vls_int16_t const_zero_s16 = svdup_n_s16(0);
const vls_int32_t const_zero_s32 = svdup_n_s32(0);
const vls_int64_t const_zero_s64 = svdup_n_s64(0);
const vls_uint8_t const_zero_u8 = svdup_n_u8(0);
const vls_uint16_t const_zero_u16 = svdup_n_u16(0);
const vls_uint32_t const_zero_u32 = svdup_n_u32(0);
const vls_uint64_t const_zero_u64 = svdup_n_u64(0);
const vls_float16_t const_zero_f16 = svdup_n_f16(0.f);
const vls_float32_t const_zero_f32 = svdup_n_f32(0.f);
const vls_float64_t const_zero_f64 = svdup_n_f64(0.0);

// Define constant for the value of one.
const vls_int8_t const_one_s8 = svdup_n_s8(1);
const vls_int16_t const_one_s16 = svdup_n_s16(1);
const vls_int32_t const_one_s32 = svdup_n_s32(1);
const vls_int64_t const_one_s64 = svdup_n_s64(1);
const vls_uint8_t const_one_u8 = svdup_n_u8(1);
const vls_uint16_t const_one_u16 = svdup_n_u16(1);
const vls_uint32_t const_one_u32 = svdup_n_u32(1);
const vls_uint64_t const_one_u64 = svdup_n_u64(1);
const vls_float16_t const_one_f16 = svdup_n_f16(1.f);
const vls_float32_t const_one_f32 = svdup_n_f32(1.f);
const vls_float64_t const_one_f64 = svdup_n_f64(1.0);

// Define constant for the value of mask.
const vls_int8_t const_all_s8_true_mask = svdup_n_s8(0xff);
const vls_int8_t const_all_s8_false_mask = svdup_n_s8(0x0);
const vls_int16_t const_all_s16_true_mask = svdup_n_s16(0xffff);
const vls_int16_t const_all_s16_false_mask = svdup_n_s16(0x0);
const vls_int32_t const_all_s32_true_mask = svdup_n_s32(0xffffffff);
const vls_int32_t const_all_s32_false_mask = svdup_n_s32(0x0);
const vls_int64_t const_all_s64_true_mask = svdup_n_s64(0xffffffffffffffff);
const vls_int64_t const_all_s64_false_mask = svdup_n_s64(0x0);
const vls_uint8_t const_all_u8_true_mask = svdup_n_u8(0x01);
const vls_uint8_t const_all_u8_false_mask = svdup_n_u8(0x00);
const vls_float16_t const_all_f16_true_mask = svreinterpret_f16_s16(const_all_s16_true_mask);
const vls_float16_t const_all_f16_false_mask = svreinterpret_f16_s16(const_all_s16_false_mask);
const vls_float32_t const_all_f32_true_mask = svreinterpret_f32_s32(const_all_s32_true_mask);
const vls_float32_t const_all_f32_false_mask = svreinterpret_f32_s32(const_all_s32_false_mask);
const vls_float64_t const_all_f64_true_mask = svreinterpret_f64_s64(const_all_s64_true_mask);
const vls_float64_t const_all_f64_false_mask = svreinterpret_f64_s64(const_all_s64_false_mask);

#define ptrue const_ptrue
#define ZERO_S8 const_zero_s8
#define ZERO_S16 const_zero_s16
#define ZERO_S32 const_zero_s32
#define ZERO_S64 const_zero_s64
#define ZERO_U8 const_zero_u8
#define ZERO_U16 const_zero_u16
#define ZERO_U32 const_zero_u32
#define ZERO_U64 const_zero_u64
#define ZERO_F16 const_zero_f16
#define ZERO_F32 const_zero_f32
#define ZERO_F64 const_zero_f64
#define ONE_S8 const_one_s8
#define ONE_S16 const_one_s16
#define ONE_S32 const_one_s32
#define ONE_S64 const_one_s64
#define ONE_U8 const_one_u8
#define ONE_U16 const_one_u16
#define ONE_U32 const_one_u32
#define ONE_U64 const_one_u64
#define ONE_F16 const_one_f16
#define ONE_F32 const_one_f32
#define ONE_F64 const_one_f64
#define ALL_S8_TRUE_MASK const_all_s8_true_mask
#define ALL_S8_FALSE_MASK const_all_s8_false_mask
#define ALL_S16_TRUE_MASK const_all_s16_true_mask
#define ALL_S16_FALSE_MASK const_all_s16_false_mask
#define ALL_S32_TRUE_MASK const_all_s32_true_mask
#define ALL_S32_FALSE_MASK const_all_s32_false_mask
#define ALL_S64_TRUE_MASK const_all_s64_true_mask
#define ALL_S64_FALSE_MASK const_all_s64_false_mask
#define ALL_U8_TRUE_MASK const_all_u8_true_mask
#define ALL_U8_FALSE_MASK const_all_u8_false_mask
#define ALL_F16_TRUE_MASK const_all_f16_true_mask
#define ALL_F16_FALSE_MASK const_all_f16_false_mask
#define ALL_F32_TRUE_MASK const_all_f32_true_mask
#define ALL_F32_FALSE_MASK const_all_f32_false_mask
#define ALL_F64_TRUE_MASK const_all_f64_true_mask
#define ALL_F64_FALSE_MASK const_all_f64_false_mask

#endif // defined(CPU_CAPABILITY_SVE)
