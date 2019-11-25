#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/THCUNN.h"
#else

#include <ATen/core/Reduction.h>

THC_API void THNN_(BCECriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *output,
                  int64_t reduction,
                  THCTensor *weights);         // [OPTIONAL]

THC_API void THNN_(BCECriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int64_t reduction,
                  THCTensor *weights);         // [OPTIONAL]

THC_API void THNN_(ClassNLLCriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCIndexTensor *target,
                  THCTensor *output,
                  int64_t reduction,
                  THCTensor *weights,       // [OPTIONAL]
                  THCTensor *total_weight,
                  int64_t ignore_index);

THC_API void THNN_(ClassNLLCriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCIndexTensor *target,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int64_t reduction,
                  THCTensor *weights,       // [OPTIONAL]
                  THCTensor *total_weight,
                  int64_t ignore_index);

THC_API void THNN_(ELU_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  accreal alpha,
                  accreal scale,
                  accreal input_scale,
                  bool inplace);

THC_API void THNN_(ELU_updateGradInput)(
                  THCState *state,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *output,
                  accreal alpha,
                  accreal scale,
                  accreal input_scale);

THC_API void THNN_(HardTanh_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  accreal min_val,
                  accreal max_val,
                  bool inplace);

THC_API void THNN_(HardTanh_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  accreal min_val,
                  accreal max_val,
                  bool inplace);

THC_API void THNN_(GatedLinear_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int dim);

THC_API void THNN_(GatedLinear_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int dim);

THC_API void THNN_(LeakyReLU_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  accreal negval,
                  bool inplace);

THC_API void THNN_(LeakyReLU_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  accreal negval,
                  bool inplace);

THC_API void THNN_(LogSigmoid_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *buffer);

THC_API void THNN_(LogSigmoid_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *buffer);

THC_API void THNN_(MultiLabelMarginCriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCIndexTensor *target,
                  THCTensor *output,
                  THCTensor *is_target,
                  int64_t reduction);

THC_API void THNN_(MultiLabelMarginCriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCIndexTensor *target,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *is_target,
                  int64_t reduction);

THC_API void THNN_(MultiMarginCriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCIndexTensor *target,
                  THCTensor *output,
                  int64_t reduction,
                  int p,
                  THCTensor *weights,           // [OPTIONAL]
                  accreal margin);

THC_API void THNN_(MultiMarginCriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCIndexTensor *target,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int64_t reduction,
                  int p,
                  THCTensor *weights,           // [OPTIONAL]
                  accreal margin);

THC_API void THNN_(SpatialClassNLLCriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCIndexTensor *target,
                  THCTensor *output,
                  int64_t reduction,
                  THCTensor *weights,       // [OPTIONAL]
                  THCTensor *total_weight,
                  int64_t ignore_index);

THC_API void THNN_(SpatialClassNLLCriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCIndexTensor *target,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int64_t reduction,
                  THCTensor *weights,       // [OPTIONAL]
                  THCTensor *total_weight,
                  int64_t ignore_index);

THC_API void THNN_(SpatialConvolutionMM_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias,              // [OPTIONAL]
                  THCTensor *columns,
                  THCTensor *ones,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH);

THC_API void THNN_(SpatialConvolutionMM_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *weight,
                  THCTensor *columns,
                  THCTensor *ones,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH);

THC_API void THNN_(SpatialConvolutionMM_accGradParameters)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,          // [OPTIONAL]
                  THCTensor *columns,
                  THCTensor *ones,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  accreal scale);

THC_API void THNN_(SpatialDepthwiseConvolution_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias,              // [OPTIONAL]
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int dilationW, int dilationH);

THC_API void THNN_(SpatialDepthwiseConvolution_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *weight,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int dilationW, int dilationH);

THC_API void THNN_(SpatialDepthwiseConvolution_accGradParameters)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradWeight,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int dilationW, int dilationH);

THC_API void THNN_(RReLU_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *noise,
                  double lower,
                  double upper,
                  bool train,
                  bool inplace,
                  void *generator);

THC_API void THNN_(RReLU_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *noise,
                  double lower,
                  double upper,
                  bool train,
                  bool inplace);

THC_API void THNN_(SoftMarginCriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *output,
                  int64_t reduction);

THC_API void THNN_(SoftMarginCriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int64_t reduction);

THC_API void THNN_(SoftPlus_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  accreal beta,
                  accreal threshold);

THC_API void THNN_(SoftPlus_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *output,
                  accreal beta,
                  accreal threshold);

THC_API void THNN_(SoftShrink_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  accreal lambda);

THC_API void THNN_(SoftShrink_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  accreal lambda);

THC_API void THNN_(Tanh_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output);

THC_API void THNN_(Tanh_updateGradInput)(
                  THCState *state,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *output);

#endif
