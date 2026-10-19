#pragma once
#include <c10/metal/common.h>

C10_METAL_CONSTEXPR uint32_t BN_MAX_THREADS = 1024;
// Per-channel Welford partial (mean, M2, count), stored as packed_float3
C10_METAL_CONSTEXPR uint32_t BN_PARTIAL_SIZE = 3;
enum { BN_MEAN, BN_SCALE, BN_SHIFT, BN_INVSTD, BN_VAR, BN_NUM_STATS };
