#pragma once
// ARM NEON uses 128-bit vector registers.

#include <ATen/cpu/vec/intrinsics.h>

#ifdef __aarch64__
#if defined(CPU_CAPABILITY_SVE) && !defined(CPU_CAPABILITY_SVE256)
// SVE128: include SVE common headers (VLS at 128 bits)
#include <ATen/cpu/vec/sve/vec_common_sve.h>
#elif !defined(CPU_CAPABILITY_SVE)
// NEON path
#include <ATen/cpu/vec/vec128/vec128_bfloat16_neon.h>
#include <ATen/cpu/vec/vec128/vec128_double_neon.h>
#include <ATen/cpu/vec/vec128/vec128_float_neon.h>
#include <ATen/cpu/vec/vec128/vec128_half_neon.h>
#include <ATen/cpu/vec/vec128/vec128_int_aarch64.h>
#include <ATen/cpu/vec/vec128/vec128_uint_aarch64.h>
#endif

#include <ATen/cpu/vec/vec128/vec128_convert.h>
#endif
