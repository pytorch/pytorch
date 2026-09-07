#pragma once

#include <c10/metal/common.h>

// The rms_norm kernels read this many elements per thread in unrolled loops.
// Inherited from the MLX kernel these were adapted from.
C10_METAL_CONSTEXPR uint32_t N_READS = 4;

// Above this normalized size a threadgroup cannot cover a row in one pass, so
// the looped variants stride over it instead. The host also falls back to the
// looped variant when a pipeline cannot field the threads a row would need.
C10_METAL_CONSTEXPR uint32_t LOOPED_LIMIT = 1024 * N_READS;

// The grad_weight partial reduction walks n_row_blocks rows for each of N
// columns. One thread per column starves at small N (N=64 would leave 64
// threads reducing 1024 rows each), so threads tile as REDUCE_COLS columns x
// REDUCE_SLICES row slices: lanes stay contiguous in N so the reads coalesce.
C10_METAL_CONSTEXPR uint32_t REDUCE_COLS = 32;
C10_METAL_CONSTEXPR uint32_t REDUCE_SLICES = 32;
