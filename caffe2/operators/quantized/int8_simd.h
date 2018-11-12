#pragma once

// We want to allow 128-bit wide SIMD if either NEON is available (as
// detected by GEMMLOWP_NEON), or whether SSE4.2 and Clang is
// available (in which case we will use the neon_sse.h library to
// share source between the two implementations). We use SSE4.2 to
// ensure we can use the full neon2sse library, and we use Clang as
// GCC has issues correctly compiling some parts of the neon2sse
// library.

// Otherwise, the INT8_NEON_SIMD variable will be undefined.

#include "gemmlowp/fixedpoint/fixedpoint.h"
#include "gemmlowp/public/gemmlowp.h"

#ifdef GEMMLOWP_NEON
#define INT8_NEON_SIMD
#endif

#if defined(__SSE4_2__) && defined(__clang__)
#include "NEON_2_SSE.h"
#define INT8_NEON_SIMD
#endif
