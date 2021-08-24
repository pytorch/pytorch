#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/THCUNN.h"
#else

#include <ATen/core/Reduction.h>
#include <ATen/Generator.h>

TORCH_CUDA_CU_API void THNN_(MultiMarginCriterion_updateOutput)(
    THCState* state,
    THCTensor* input,
    THCIndexTensor* target,
    THCTensor* output,
    int64_t reduction,
    int p,
    THCTensor* weights, // [OPTIONAL]
    accreal margin);

TORCH_CUDA_CU_API void THNN_(MultiMarginCriterion_updateGradInput)(
    THCState* state,
    THCTensor* input,
    THCIndexTensor* target,
    THCTensor* gradOutput,
    THCTensor* gradInput,
    int64_t reduction,
    int p,
    THCTensor* weights, // [OPTIONAL]
    accreal margin);

#endif
