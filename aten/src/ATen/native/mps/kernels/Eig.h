#pragma once
#include <c10/metal/common.h>

// Largest matrix the threadgroup-memory solver can hold: three n-by-n complex
// scratch matrices must fit in 32KB of threadgroup memory. It is also the SIMD
// width, since each matrix is solved by one SIMD group with a lane per row.
C10_METAL_CONSTEXPR int kEigMaxDim = 32;

struct EigParams {
  int n;
  bool compute_vectors;
};
