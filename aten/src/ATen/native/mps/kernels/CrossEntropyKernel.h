#pragma once
#include <c10/metal/common.h>

// Parameters for the fused 2D cross-entropy Metal kernels.
//   has_weight: 0/1 flag; when set a per-class weight buffer (and the
//               precomputed sum_w buffer) is bound and consulted.
struct CrossEntropyParams {
  uint32_t vocab_size;
  uint32_t batch_size;
  int32_t ignore_index;
  uint32_t has_weight;
  float label_smoothing;
};
