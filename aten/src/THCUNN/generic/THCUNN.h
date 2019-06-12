#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCUNN.h"
#else

#include <ATen/core/Reduction.h>

THC_API void THNN_(Abs_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output);

THC_API void THNN_(Abs_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput);

THC_API void THNN_(AbsCriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *output,
                  int64_t reduction);

THC_API void THNN_(AbsCriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int64_t reduction);

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

THC_API void THNN_(DistKLDivCriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *output,
                  int64_t reduction);

THC_API void THNN_(DistKLDivCriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int64_t reduction);

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

THC_API void THNN_(FeatureLPPooling_updateOutput)(
                  THCState* state,
                  THCTensor* inputTH,
                  THCTensor* outputTH,
                  accreal power,
                  int width,
                  int stride,
                  bool batchMode);

THC_API void THNN_(FeatureLPPooling_updateGradInput)(
                  THCState* state,
                  THCTensor* gradOutputTH,
                  THCTensor* inputTH,
                  THCTensor* outputTH,
                  THCTensor* gradInputTH,
                  accreal power,
                  int width,
                  int stride,
                  bool batchMode);

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

THC_API void THNN_(Im2Col_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int64_t kH, int64_t kW,
                  int64_t dH, int64_t dW,
                  int64_t padH, int64_t padW,
                  int64_t sH, int64_t sW);

THC_API void THNN_(Im2Col_updateGradInput)(
                  THCState *state,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int64_t inputHeight, int64_t inputWidth,
                  int64_t kH, int64_t kW,
                  int64_t dH, int64_t dW,
                  int64_t padH, int64_t padW,
                  int64_t sH, int64_t sW);

THC_API void THNN_(Col2Im_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int64_t outputHeight, int64_t outputWidth,
                  int64_t kH, int64_t kW,
                  int64_t dH, int64_t dW,
                  int64_t padH, int64_t padW,
                  int64_t sH, int64_t sW);

 THC_API void THNN_(Col2Im_updateGradInput)(
                  THCState *state,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int64_t kH, int64_t kW,
                  int64_t dH, int64_t dW,
                  int64_t padH, int64_t padW,
                  int64_t sH, int64_t sW);

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

THC_API void THNN_(LookupTable_accGradParameters)(
                  THCState *state,
                  THCIndexTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradWeight,
                  THCIndexTensor *count,
                  THCIndexTensor *sorted,       // [OPTIONAL]
                  THCIndexTensor *indices,      // [OPTIONAL]
                  bool scaleGradByFreq,
                  int paddingValue,
                  accreal scale);

THC_API void THNN_(LookupTable_renorm)(
                  THCState *state,
                  THCIndexTensor *idx,
                  THCTensor *weight,
                  accreal maxNorm,
                  accreal normType);

THC_API void THNN_(LookupTableBag_updateOutput)(
           THCState *state,
           THCIndexTensor *input,
           THCIndexTensor *offsets,
           THCTensor *weight,
           THCTensor *output,
           THCIndexTensor *offset2bag,
	   int mode,
           THCIndexTensor *seq_length);       // [OPTIONAL]

THC_API void THNN_(LookupTableBag_accGradParameters)(
           THCState *state,
           THCIndexTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradWeight,
           THCIndexTensor *offset2bag,
           THCIndexTensor *count,
           THCIndexTensor *sortedIndices,
           THCIndexTensor *origIndices,
           bool scaleGradByFreq,
	   int mode,
	   THCIndexTensor *seq_length,        // [OPTIONAL]
           accreal scale_);

THC_API void THNN_(L1Cost_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output);

THC_API void THNN_(L1Cost_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,       // [OPTIONAL]
                  THCTensor *gradInput);

THC_API void THNN_(MarginCriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *output,
                  bool sizeAverage,
                  accreal margin);

THC_API void THNN_(MarginCriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *gradInput,
                  bool sizeAverage,
                  accreal margin);

THC_API void THNN_(MSECriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *output,
                  int64_t reduction);

THC_API void THNN_(MSECriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int64_t reduction);

THC_API void THNN_(MultiLabelMarginCriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCIndexTensor *target,
                  THCTensor *output,
                  THCTensor *istarget,
                  int64_t reduction);

THC_API void THNN_(MultiLabelMarginCriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCIndexTensor *target,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *istarget,
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
THC_API void THNN_(SmoothL1Criterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *output,
                  int64_t reduction);

THC_API void THNN_(SmoothL1Criterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int64_t reduction);

THC_API void THNN_(SparseLinear_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias);

THC_API void THNN_(SparseLinear_accGradParameters)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,
                  THCTensor *weight,
                  THCTensor *bias,
                  accreal weightDecay,
                  accreal scale);

THC_API void THNN_(SparseLinear_legacyUpdateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias);

THC_API void THNN_(SparseLinear_legacyAccGradParameters)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,
                  THCTensor *weight,
                  THCTensor *bias,
                  accreal weightDecay,
                  accreal scale);

THC_API void THNN_(SparseLinear_zeroGradParameters)(
                  THCState *state,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,
                  THCTensor *lastInput);

THC_API void THNN_(SparseLinear_updateParameters)(
                  THCState *state,
                  THCTensor *weight,
                  THCTensor *bias,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,
                  THCTensor *lastInput,
                  accreal learningRate);

THC_API void THNN_(IndexLinear_updateOutput)(
                  THCState *state,
                  THCIndexTensor *keys,
                  int64_t keysOffset,
                  THCTensor *values,
                  THCIndexTensor *sizes,
                  THCIndexTensor *cumSumSizes,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias,
                  THCTensor *normalizedValues,
                  int   train);

THC_API void THNN_(IndexLinear_accGradParameters)(
                  THCState *state,
                  THCIndexTensor *keys,
                  int64_t keysOffset,
                  THCTensor *values,
                  THCIndexTensor *sizes,
                  THCIndexTensor *cumSumSizes,
                  THCTensor *gradOutput,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,
                  THCTensor *weight,
                  THCTensor *bias,
                  THCTensor* valuesBuffer,
                  accreal weightDecay,
                  accreal scale);

THC_API void THNN_(IndexLinear_accUpdateGradParameters)(
                  THCState *state,
                  THCIndexTensor *keys,
                  int64_t keysOffset,
                  THCTensor *values,
                  THCIndexTensor *sizes,
                  THCIndexTensor *cumSumSizes,
                  THCTensor *gradOutput,
                  THCTensor *weight,
                  THCTensor *bias,
                  accreal weightDecay,
                  accreal scale);

THC_API void THNN_(IndexLinear_updateParameters)(
                  THCState *state,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,
                  THCTensor *weight,
                  THCTensor *bias,
                  THCIndexTensor *runningKeys,
                  THCIndexTensor *cumSumSizes,
                  int64_t keysOffset,
                  accreal weightDecay,
                  accreal learningRate);

THC_API void THNN_(SpatialAdaptiveMaxPooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCIndexTensor *indices,
                  int osizeW,
                  int osizeH);

THC_API void THNN_(SpatialAdaptiveMaxPooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCIndexTensor *indices);

THC_API void THNN_(SpatialAdaptiveAveragePooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int osizeW,
                  int osizeH);

THC_API void THNN_(SpatialAdaptiveAveragePooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput);

THC_API void THNN_(SpatialAveragePooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  bool ceil_mode,
                  bool count_include_pad);

THC_API void THNN_(SpatialAveragePooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  bool ceil_mode,
                  bool count_include_pad);

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

THC_API void THNN_(SpatialConvolutionLocal_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias,
                  THCTensor *finput,
                  THCTensor *fgradInput,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int64_t inputWidth, int64_t inputHeight,
                  int64_t outputWidth, int64_t outputHeight);

THC_API void THNN_(SpatialConvolutionLocal_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *weight,
                  THCTensor *finput,
                  THCTensor *fgradInput,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int64_t inputWidth, int64_t inputHeight,
                  int64_t outputWidth, int64_t outputHeight);

THC_API void THNN_(SpatialConvolutionLocal_accGradParameters)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,
                  THCTensor *finput,
                  THCTensor *fgradInput,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int64_t inputWidth, int64_t inputHeight,
                  int64_t outputWidth, int64_t outputHeight,
                  accreal scale);

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

THC_API void THNN_(SpatialCrossMapLRN_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *scale,
                  int size,
                  accreal alpha,
                  accreal beta,
                  accreal k);

THC_API void THNN_(SpatialCrossMapLRN_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *scale,
                  THCTensor *output,
                  int size,
                  accreal alpha,
                  accreal beta,
                  accreal k);

THC_API void THNN_(SpatialDilatedConvolution_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias,            // [OPTIONAL]
                  THCTensor *columns,
                  THCTensor *ones,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int dilationW, int dilationH);

THC_API void THNN_(SpatialDilatedConvolution_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *weight,
                  THCTensor *columns,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int dilationW, int dilationH);

THC_API void THNN_(SpatialDilatedConvolution_accGradParameters)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,        // [OPTIONAL]
                  THCTensor *columns,
                  THCTensor *ones,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int dilationW, int dilationH,
                  accreal scale);

THC_API void THNN_(SpatialFullDilatedConvolution_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias,          // [OPTIONAL]
                  THCTensor *columns,
                  THCTensor *ones,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int dilationW, int dilationH,
                  int adjW, int adjH);

THC_API void THNN_(SpatialFullDilatedConvolution_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *weight,
                  THCTensor *columns,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int dilationW, int dilationH,
                  int adjW, int adjH);

THC_API void THNN_(SpatialFullDilatedConvolution_accGradParameters)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,     // [OPTIONAL]
                  THCTensor *columns,
                  THCTensor *ones,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int dilationW, int dilationH,
                  int adjW, int adjH,
                  accreal scale);

THC_API void THNN_(SpatialDilatedMaxPooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCIndexTensor *indices,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int dilationW, int dilationH,
                  bool ceil_mode);

THC_API void THNN_(SpatialDilatedMaxPooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCIndexTensor *indices,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int dilationW, int dilationH,
                  bool ceil_mode);

THC_API void THNN_(SpatialFractionalMaxPooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int outputW, int outputH,
                  int poolSizeW, int poolSizeH,
                  THCIndexTensor *indices,
                  THCTensor *randomSamples);

THC_API void THNN_(SpatialFractionalMaxPooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int outputW, int outputH,
                  int poolSizeW, int poolSizeH,
                  THCIndexTensor *indices);

THC_API void THNN_(SpatialFullConvolution_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias,          // [OPTIONAL]
                  THCTensor *columns,
                  THCTensor *ones,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int adjW, int adjH);

THC_API void THNN_(SpatialFullConvolution_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *weight,
                  THCTensor *columns,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int adjW, int adjH);

THC_API void THNN_(SpatialFullConvolution_accGradParameters)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,     // [OPTIONAL]
                  THCTensor *columns,
                  THCTensor *ones,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int adjW, int adjH,
                  accreal scale);

THC_API void THNN_(SpatialMaxPooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCIndexTensor *indices,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  bool ceil_mode);

THC_API void THNN_(SpatialMaxPooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCIndexTensor *indices,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  bool ceil_mode);

THC_API void THNN_(SpatialMaxUnpooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCIndexTensor *indices,
                  int owidth, int oheight);

THC_API void THNN_(SpatialMaxUnpooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCIndexTensor *indices,
                  int owidth, int oheight);

THC_API void THNN_(SpatialReflectionPadding_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int padL, int padR,
                  int padT, int padB);

THC_API void THNN_(SpatialReflectionPadding_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int padL, int padR,
                  int padT, int padB);

THC_API void THNN_(SpatialReplicationPadding_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int padL, int padR,
                  int padT, int padB);

THC_API void THNN_(SpatialReplicationPadding_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int padL, int padR,
                  int padT, int padB);

THC_API void THNN_(SpatialSubSampling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias,
                  int kW, int kH,
                  int dW, int dH);

THC_API void THNN_(SpatialSubSampling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *weight,
                  int kW, int kH,
                  int dW, int dH);

THC_API void THNN_(SpatialSubSampling_accGradParameters)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,
                  int kW, int kH,
                  int dW, int dH,
                  accreal scale);

THC_API void THNN_(SpatialUpSamplingBilinear_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int outputHeight,
                  int outputWidth,
                  bool align_corners);

THC_API void THNN_(SpatialUpSamplingBilinear_updateGradInput)(
                  THCState *state,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int nbatch,
                  int nchannels,
                  int inputHeight,
                  int inputWidth,
                  int outputHeight,
                  int outputWidth,
                  bool align_corners);

THC_API void THNN_(SpatialUpSamplingNearest_updateGradInput)(
                  THCState *state,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int nbatch,
                  int nchannels,
                  int inputHeight,
                  int inputWidth,
                  int outputHeight,
                  int outputWidth);

THC_API void THNN_(SpatialUpSamplingNearest_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int outputHeight,
                  int outputWidth);

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

THC_API void THNN_(Sigmoid_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output);

THC_API void THNN_(Sigmoid_updateGradInput)(
                  THCState *state,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *output);

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

THC_API void THNN_(Square_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output);

THC_API void THNN_(Square_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput);

THC_API void THNN_(Sqrt_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  accreal eps);

THC_API void THNN_(Sqrt_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *output);

THC_API void THNN_(Tanh_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output);

THC_API void THNN_(Tanh_updateGradInput)(
                  THCState *state,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *output);

THC_API void THNN_(TemporalConvolution_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias,
                  int kW, int dW,
                  int inputFrameSize,
                  int outputFrameSize);

THC_API void THNN_(TemporalConvolution_updateGradInput)(
                  THCState* state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *weight,
                  int kW, int dW);

THC_API void THNN_(TemporalConvolution_accGradParameters)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,
                  int kW, int dW,
                  accreal scale);

THC_API void THNN_(TemporalMaxPooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCIndexTensor *indices,
                  int kW, int dW);

THC_API void THNN_(TemporalMaxPooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCIndexTensor *indices,
                  int kW, int dW);

THC_API void THNN_(TemporalRowConvolution_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias,          // [OPTIONAL]
                  THCTensor *finput,
                  THCTensor *fgradInput,
                  int kW,
                  int dW,
                  int padW,
                  bool featFirst);

THC_API void THNN_(TemporalRowConvolution_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *weight,
                  THCTensor *finput,
                  THCTensor *fgradInput,
                  int kW,
                  int dW,
                  int padW,
                  bool featFirst);

THC_API void THNN_(TemporalRowConvolution_accGradParameters)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,
                  THCTensor *finput,
                  THCTensor *fgradInput,
                  int kW,
                  int dW,
                  int padW,
                  bool featFirst,
                  accreal scale);

THC_API void THNN_(TemporalReflectionPadding_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int padL, int padR);

THC_API void THNN_(TemporalReflectionPadding_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int padL, int padR);

THC_API void THNN_(TemporalReplicationPadding_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int padL, int padR);

THC_API void THNN_(TemporalReplicationPadding_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int padL, int padR);

THC_API void THNN_(TemporalUpSamplingLinear_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int outputWidth,
                  bool align_corners);

THC_API void THNN_(TemporalUpSamplingLinear_updateGradInput)(
                  THCState *state,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int nbatch,
                  int nchannels,
                  int inputWidth,
                  int outputWidth,
                  bool align_corners);

THC_API void THNN_(TemporalUpSamplingNearest_updateGradInput)(
                  THCState *state,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int nbatch,
                  int nchannels,
                  int inputWidth,
                  int outputWidth);

THC_API void THNN_(TemporalUpSamplingNearest_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int outputWidth);

THC_API void THNN_(VolumetricAveragePooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int kT, int kW, int kH,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  bool ceil_mode,
                  bool count_include_pad);

THC_API void THNN_(VolumetricAveragePooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int kT, int kW, int kH,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  bool ceil_mode,
                  bool count_include_pad);

// VolumetricConvolution is legacy and purposefully not bound by ATen
THC_API void THNN_(VolumetricConvolution_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias,         // [OPTIONAL]
                  THCTensor *finput,
                  THCTensor *fgradInput,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH);

THC_API void THNN_(VolumetricConvolution_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *weight,
                  THCTensor *finput,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH);

THC_API void THNN_(VolumetricConvolution_accGradParameters)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,     // [OPTIONAL]
                  THCTensor *finput,
                  THCTensor *fgradInput,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  accreal scale);

THC_API void THNN_(VolumetricDilatedConvolution_updateOutput)(
                  THCState *state,
                  THCTensor  *input,
                  THCTensor  *output,
                  THCTensor  *weight,
                  THCTensor  *bias,        // [OPTIONAL]
                  THCTensor  *columns,
                  THCTensor  *ones,
                  int kT, int kW, int kH,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  int dilationT, int dilationW, int dilationH);

THC_API void THNN_(VolumetricDilatedConvolution_updateGradInput)(
                  THCState *state,
                  THCTensor  *input,
                  THCTensor  *gradOutput,
                  THCTensor  *gradInput,
                  THCTensor  *weight,
                  THCTensor  *columns,
                  int kT, int kW, int kH,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  int dilationT, int dilationW, int dilationH);

THC_API void THNN_(VolumetricDilatedConvolution_accGradParameters)(
                  THCState *state,
                  THCTensor  *input,
                  THCTensor  *gradOutput,
                  THCTensor  *gradWeight,
                  THCTensor  *gradBias,    // [OPTIONAL]
                  THCTensor  *columns,
                  THCTensor  *ones,
                  int kT, int kW, int kH,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  int dilationT, int dilationW, int dilationH,
                  accreal scale);

THC_API void THNN_(VolumetricFullDilatedConvolution_updateOutput)(
                  THCState *state,
                  THCTensor  *input,
                  THCTensor  *output,
                  THCTensor  *weight,
                  THCTensor  *bias,        // [OPTIONAL]
                  THCTensor  *finput,
                  THCTensor  *fgradInput,
                  int kT, int kW, int kH,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  int dilationT, int dilationW, int dilationH,
                  int adjT, int adjW, int adjH);

THC_API void THNN_(VolumetricFullDilatedConvolution_updateGradInput)(
                  THCState *state,
                  THCTensor  *input,
                  THCTensor  *gradOutput,
                  THCTensor  *gradInput,
                  THCTensor  *weight,
                  THCTensor  *finput,
                  THCTensor  *fgradInput,
                  int kT, int kW, int kH,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  int dilationT, int dilationW, int dilationH,
                  int adjT, int adjW, int adjH);

THC_API void THNN_(VolumetricFullDilatedConvolution_accGradParameters)(
                  THCState *state,
                  THCTensor  *input,
                  THCTensor  *gradOutput,
                  THCTensor  *gradWeight,  // [OPTIONAL]
                  THCTensor  *gradBias,    // [OPTIONAL]
                  THCTensor  *finput,
                  THCTensor  *fgradInput,
                  int kT, int kW, int kH,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  int dilationT, int dilationW, int dilationH,
                  int adjT, int adjW, int adjH,
                  accreal scale);

THC_API void THNN_(VolumetricDilatedMaxPooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCIndexTensor *indices,
                  int kT, int kW, int kH,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  int dilationT, int dilationW, int dilationH,
                  bool ceilMode);

THC_API void THNN_(VolumetricDilatedMaxPooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCIndexTensor *indices,
                  int kT, int kW, int kH,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  int dilationT, int dilationW, int dilationH,
                  bool ceilMode);

THC_API void THNN_(VolumetricFractionalMaxPooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int outputT, int outputW, int outputH,
                  int poolSizeT, int poolSizeW, int poolSizeH,
                  THCIndexTensor *indices,
                  THCTensor *randomSamples);

THC_API void THNN_(VolumetricFractionalMaxPooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int outputT, int outputW, int outputH,
                  int poolSizeT, int poolSizeW, int poolSizeH,
                  THCIndexTensor *indices);

THC_API void THNN_(VolumetricFullConvolution_updateOutput)(
                  THCState *state,
                  THCTensor  *input,
                  THCTensor  *output,
                  THCTensor  *weight,
                  THCTensor  *bias,        // [OPTIONAL]
                  THCTensor  *finput,
                  THCTensor  *fgradInput,
                  int kT, int kW, int kH,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  int adjT, int adjW, int adjH);

THC_API void THNN_(VolumetricFullConvolution_updateGradInput)(
                  THCState *state,
                  THCTensor  *input,
                  THCTensor  *gradOutput,
                  THCTensor  *gradInput,
                  THCTensor  *weight,
                  THCTensor  *finput,
                  THCTensor  *fgradInput,
                  int kT, int kW, int kH,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  int adjT, int adjW, int adjH);

THC_API void THNN_(VolumetricFullConvolution_accGradParameters)(
                  THCState *state,
                  THCTensor  *input,
                  THCTensor  *gradOutput,
                  THCTensor  *gradWeight,  // [OPTIONAL]
                  THCTensor  *gradBias,    // [OPTIONAL]
                  THCTensor  *finput,
                  THCTensor  *fgradInput,
                  int kT, int kW, int kH,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  int adjT, int adjW, int adjH,
                  accreal scale);

THC_API void THNN_(VolumetricMaxPooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCIndexTensor *indices,
                  int kT, int kW, int kH,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  bool ceilMode);

THC_API void THNN_(VolumetricMaxPooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCIndexTensor *indices,
                  int kT, int kW, int kH,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  bool ceilMode);

THC_API void THNN_(VolumetricMaxUnpooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCIndexTensor *indices,
                  int outputTime, int outputWidth, int outputHeight,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH);

THC_API void THNN_(VolumetricMaxUnpooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCIndexTensor *indices,
                  int outputTime, int outputWidth, int outputHeight,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH);

THC_API void THNN_(VolumetricAdaptiveMaxPooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCIndexTensor *indices,
                  int osizeT,
                  int osizeW,
                  int osizeH);

THC_API void THNN_(VolumetricAdaptiveMaxPooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCIndexTensor *indices);

THC_API void THNN_(VolumetricAdaptiveAveragePooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int osizeT,
                  int osizeW,
                  int osizeH);

THC_API void THNN_(VolumetricAdaptiveAveragePooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput);

THC_API void THNN_(VolumetricReplicationPadding_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int pleft, int pright,
                  int ptop, int pbottom,
                  int pfront, int pback);

THC_API void THNN_(VolumetricReplicationPadding_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int pleft, int pright,
                  int ptop, int pbottom,
                  int pfront, int pback);

THC_API void THNN_(VolumetricUpSamplingNearest_updateGradInput)(
                  THCState *state,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int nbatch,
                  int nchannels,
                  int inputDepth,
                  int inputHeight,
                  int inputWidth,
                  int outputDepth,
                  int outputHeight,
                  int outputWidth);

THC_API void THNN_(VolumetricUpSamplingNearest_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int outputDepth,
                  int outputHeight,
                  int outputWidth);

THC_API void THNN_(VolumetricUpSamplingTrilinear_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int outputDepth,
                  int outputHeight,
                  int outputWidth,
                  bool align_corners);

THC_API void THNN_(VolumetricUpSamplingTrilinear_updateGradInput)(
                  THCState *state,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int nbatch,
                  int nchannels,
                  int inputDepth,
                  int inputHeight,
                  int inputWidth,
                  int outputDepth,
                  int outputHeight,
                  int outputWidth,
                  bool align_corners);

#endif
