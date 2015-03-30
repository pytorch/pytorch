#include "THCApply.cuh"

// Implementation of copyIgnoringOverlaps, defined after pointwiseApply2.
void THCudaTensor_copyIgnoringOverlaps(THCState* state,
                                       THCudaTensor* dst,
                                       THCudaTensor* src) {
  THCudaTensor_pointwiseApply2(state, dst, src, CopyOp<float>(),
                               ReadOnly, // ignore overwrites
                               ReadOnly);
}
