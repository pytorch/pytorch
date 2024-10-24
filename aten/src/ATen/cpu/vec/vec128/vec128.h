#pragma once
// ARM NEON uses 128-bit vector registers.

#include <ATen/cpu/vec/intrinsics.h>

#if !defined(CPU_CAPABILITY_SVE)
#include <ATen/cpu/vec/vec128/vec128_float_neon.h>
#include <ATen/cpu/vec/vec128/vec128_half_neon.h>
#endif
