#pragma once
#include <c10/metal/common.h>

// Largest matrix the threadgroup-memory solver can hold: two n-by-n complex
// scratch matrices must fit in 32KB of threadgroup memory.
C10_METAL_CONSTEXPR int kEigMaxDim = 32;

struct EigParams {
  int n;
  int compute_vectors;
};
