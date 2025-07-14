#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>

#if defined(CPU_CAPABILITY_RVV)

typedef int8_t fixed_vint8m2_t[CONFIG_VLMAX / sizeof (int8_t)];
typedef int16_t fixed_vint16m2_t[CONFIG_VLMAX / sizeof (int16_t)];
typedef int32_t fixed_vint32m2_t[CONFIG_VLMAX / sizeof (int32_t)];
typedef int64_t fixed_vint64m2_t[CONFIG_VLMAX / sizeof (int64_t)];

typedef uint8_t fixed_vuint8m2_t[CONFIG_VLMAX / sizeof (uint8_t)];
typedef uint16_t fixed_vuint16m2_t[CONFIG_VLMAX / sizeof (uint16_t)];
typedef uint32_t fixed_vuint32m2_t[CONFIG_VLMAX / sizeof (uint32_t)];
typedef uint64_t fixed_vuint64m2_t[CONFIG_VLMAX / sizeof (uint64_t)];

typedef float fixed_vfloat32m2_t[CONFIG_VLMAX / sizeof (float)];
typedef double fixed_vfloat64m2_t[CONFIG_VLMAX / sizeof (double)];

#define VFLOAT32_VL  (CONFIG_VLMAX_BITS / 32)
#define VQINT8_VL  (CONFIG_VLMAX_BITS / 8)
#define VQUINT8_VL  (CONFIG_VLMAX_BITS / 8)
#define VQINT32_VL  (CONFIG_VLMAX_BITS / 32)

#endif // defined(CPU_CAPABILITY_RVV)
