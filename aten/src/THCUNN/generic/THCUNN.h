#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/THCUNN.h"
#else

#include <ATen/core/Reduction.h>
#include <ATen/Generator.h>

TORCH_CUDA_CU_API void THNN_(ClassNLLCriterion_updateOutput)(
    THCState* state,
    THCTensor* input,
    THCIndexTensor* target,
    THCTensor* output,
    int64_t reduction,
    THCTensor* weights, // [OPTIONAL]
    THCTensor* total_weight,
    int64_t ignore_index);

TORCH_CUDA_CU_API void THNN_(ClassNLLCriterion_updateGradInput)(
    THCState* state,
    THCTensor* input,
    THCIndexTensor* target,
    THCTensor* gradOutput,
    THCTensor* gradInput,
    int64_t reduction,
    THCTensor* weights, // [OPTIONAL]
    THCTensor* total_weight,
    int64_t ignore_index);

TORCH_CUDA_CU_API void THNN_(GatedLinear_updateOutput)(
    THCState* state,
    THCTensor* input,
    THCTensor* output,
    int dim);

TORCH_CUDA_CU_API void THNN_(GatedLinear_updateGradInput)(
    THCState* state,
    THCTensor* input,
    THCTensor* gradOutput,
    THCTensor* gradInput,
    int dim);

TORCH_CUDA_CU_API void THNN_(LogSigmoid_updateOutput)(
    THCState* state,
    THCTensor* input,
    THCTensor* output,
    THCTensor* buffer);

TORCH_CUDA_CU_API void THNN_(LogSigmoid_updateGradInput)(
    THCState* state,
    THCTensor* input,
    THCTensor* gradOutput,
    THCTensor* gradInput,
    THCTensor* buffer);

TORCH_CUDA_CU_API void THNN_(MultiLabelMarginCriterion_updateOutput)(
    THCState* state,
    THCTensor* input,
    THCIndexTensor* target,
    THCTensor* output,
    THCTensor* is_target,
    int64_t reduction);

TORCH_CUDA_CU_API void THNN_(MultiLabelMarginCriterion_updateGradInput)(
    THCState* state,
    THCTensor* input,
    THCIndexTensor* target,
    THCTensor* gradOutput,
    THCTensor* gradInput,
    THCTensor* is_target,
    int64_t reduction);

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

TORCH_CUDA_CU_API void THNN_(SpatialClassNLLCriterion_updateOutput)(
    THCState* state,
    THCTensor* input,
    THCIndexTensor* target,
    THCTensor* output,
    int64_t reduction,
    THCTensor* weights, // [OPTIONAL]
    THCTensor* total_weight,
    int64_t ignore_index);

TORCH_CUDA_CU_API void THNN_(SpatialClassNLLCriterion_updateGradInput)(
    THCState* state,
    THCTensor* input,
    THCIndexTensor* target,
    THCTensor* gradOutput,
    THCTensor* gradInput,
    int64_t reduction,
    THCTensor* weights, // [OPTIONAL]
    THCTensor* total_weight,
    int64_t ignore_index);

TORCH_CUDA_CU_API void THNN_(SpatialConvolutionMM_updateOutput)(
    THCState* state,
    THCTensor* input,
    THCTensor* output,
    THCTensor* weight,
    THCTensor* bias, // [OPTIONAL]
    THCTensor* columns,
    THCTensor* ones,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH);

TORCH_CUDA_CU_API void THNN_(SpatialConvolutionMM_updateGradInput)(
    THCState* state,
    THCTensor* input,
    THCTensor* gradOutput,
    THCTensor* gradInput,
    THCTensor* weight,
    THCTensor* columns,
    THCTensor* ones,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH);

TORCH_CUDA_CU_API void THNN_(SpatialConvolutionMM_accGradParameters)(
    THCState* state,
    THCTensor* input,
    THCTensor* gradOutput,
    THCTensor* gradWeight,
    THCTensor* gradBias, // [OPTIONAL]
    THCTensor* columns,
    THCTensor* ones,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    accreal scale);

TORCH_CUDA_CU_API void THNN_(SpatialDepthwiseConvolution_updateOutput)(
    THCState* state,
    THCTensor* input,
    THCTensor* output,
    THCTensor* weight,
    THCTensor* bias, // [OPTIONAL]
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH);

TORCH_CUDA_CU_API void THNN_(SpatialDepthwiseConvolution_updateGradInput)(
    THCState* state,
    THCTensor* input,
    THCTensor* gradOutput,
    THCTensor* gradInput,
    THCTensor* weight,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH);

TORCH_CUDA_CU_API void THNN_(SpatialDepthwiseConvolution_accGradParameters)(
    THCState* state,
    THCTensor* input,
    THCTensor* gradOutput,
    THCTensor* gradWeight,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int dilationW,
    int dilationH);

TORCH_CUDA_CU_API void THNN_(RReLU_updateOutput)(
    THCState* state,
    THCTensor* input,
    THCTensor* output,
    THCTensor* noise,
    double lower,
    double upper,
    bool train,
    bool inplace,
    c10::optional<at::Generator> generator);

TORCH_CUDA_CU_API void THNN_(RReLU_updateGradInput)(
    THCState* state,
    THCTensor* input,
    THCTensor* gradOutput,
    THCTensor* gradInput,
    THCTensor* noise,
    double lower,
    double upper,
    bool train,
    bool inplace);
#endif
