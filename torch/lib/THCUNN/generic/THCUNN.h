#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCUNN.h"
#else

TH_API void THNN_(Abs_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output);

TH_API void THNN_(Abs_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput);

TH_API void THNN_(AbsCriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *output,
                  bool sizeAverage);

TH_API void THNN_(AbsCriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *gradInput,
                  bool sizeAverage);

TH_API void THNN_(BatchNormalization_updateOutput)(
                  THCState *state,
                  THCTensor *input_,
                  THCTensor *output_,
                  THCTensor *weight_,        // [OPTIONAL]
                  THCTensor *bias_,          // [OPTIONAL]
                  THCTensor *runningMean_,
                  THCTensor *runningVar_,
                  THCTensor *saveMean_,
                  THCTensor *saveStd_,
                  bool train,
                  double momentum,
                  double eps);

TH_API void THNN_(BatchNormalization_backward)(
                  THCState *state,
                  THCTensor *input_,
                  THCTensor *gradOutput_,
                  THCTensor *gradInput_,        // [OPTIONAL]
                  THCTensor *gradWeight_,       // [OPTIONAL]
                  THCTensor *gradBias_,         // [OPTIONAL]
                  THCTensor *weight_,           // [OPTIONAL]
                  THCTensor *runningMean_,
                  THCTensor *runningVar_,
                  THCTensor *saveMean_,
                  THCTensor *saveStd_,
                  bool train,
                  double scale,
                  double eps);

TH_API void THNN_(BCECriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *output,
                  bool sizeAverage,
                  THCTensor *weights);        // [OPTIONAL]

TH_API void THNN_(BCECriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *gradInput,
                  bool sizeAverage,
                  THCTensor *weights);        // [OPTIONAL]

TH_API void THNN_(ClassNLLCriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCIndexTensor *target,
                  THCTensor *output,
                  bool sizeAverage,
                  THCTensor *weights,       // [OPTIONAL]
                  THCTensor *total_weight,
                  long ignore_index);

TH_API void THNN_(ClassNLLCriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCIndexTensor *target,
                  THCTensor *gradInput,
                  bool sizeAverage,
                  THCTensor *weights,       // [OPTIONAL]
                  THCTensor *total_weight,
                  long ignore_index);

TH_API void THNN_(DistKLDivCriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *output,
                  bool sizeAverage);

TH_API void THNN_(DistKLDivCriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *gradInput,
                  bool sizeAverage);

TH_API void THNN_(ELU_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  accreal alpha,
                  bool inplace);

TH_API void THNN_(ELU_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *output,
                  accreal alpha,
                  bool inplace);

TH_API void THNN_(HardTanh_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  accreal min_val,
                  accreal max_val,
                  bool inplace);

TH_API void THNN_(HardTanh_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  accreal min_val,
                  accreal max_val,
                  bool inplace);

TH_API void THNN_(GatedLinear_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int dim);

TH_API void THNN_(GatedLinear_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int dim);

TH_API void THNN_(LeakyReLU_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  accreal negval,
                  bool inplace);

TH_API void THNN_(LeakyReLU_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  accreal negval,
                  bool inplace);

TH_API void THNN_(GRUFused_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *hidden,
                  THCTensor *bias1, // [OPTIONAL]
                  THCTensor *bias2, // [OPTIONAL]
                  THCTensor *hx,
                  THCTensor *hy,
                  THCTensor *storage);

TH_API void THNN_(GRUFused_updateGradInput)(
                  THCState *state,
                  THCTensor *gradInInput,
                  THCTensor *gradInHidden,
                  THCTensor *gradOutput,
                  THCTensor *gradInputHx,
                  THCTensor *storage);

TH_API void THNN_(LSTMFused_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *hidden,
                  THCTensor *bias1, // [OPTIONAL]
                  THCTensor *bias2, // [OPTIONAL]
                  THCTensor *cx,
                  THCTensor *hy,
                  THCTensor *cy);

TH_API void THNN_(LSTMFused_updateGradInput)(
                  THCState *state,
                  THCTensor *storage,
                  THCTensor *gradInGates,
                  THCTensor *prevC,
                  THCTensor *cy,
                  THCTensor *gradOutput,
                  THCTensor *gradOutputCell,
                  THCTensor *gradInputCx);

TH_API void THNN_(LogSigmoid_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *buffer);

TH_API void THNN_(LogSigmoid_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *buffer);

TH_API void THNN_(LogSoftMax_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output);

TH_API void THNN_(LogSoftMax_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *output);

TH_API void THNN_(LookupTable_accGradParameters)(
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

TH_API void THNN_(LookupTable_renorm)(
                  THCState *state,
                  THCIndexTensor *idx,
                  THCTensor *weight,
                  accreal maxNorm,
                  accreal normType);

TH_API void THNN_(LookupTableSum_updateOutput)(
           THCState *state,
           THCIndexTensor *input,
           THCIndexTensor *offsets,
           THCTensor *weight,
           THCTensor *output,
           THCIndexTensor *offset2bag);

TH_API void THNN_(LookupTableSum_accGradParameters)(
           THCState *state,
           THCIndexTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradWeight,
           THCIndexTensor *offset2bag,
           THCIndexTensor *count,
           THCIndexTensor *sortedIndices,
           THCIndexTensor *origIndices,
           bool scaleGradByFreq,
           accreal scale_);

TH_API void THNN_(L1Cost_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output);

TH_API void THNN_(L1Cost_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,       // [OPTIONAL]
                  THCTensor *gradInput);

TH_API void THNN_(MarginCriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *output,
                  bool sizeAverage,
                  accreal margin);

TH_API void THNN_(MarginCriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *gradInput,
                  bool sizeAverage,
                  accreal margin);

TH_API void THNN_(MSECriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *output,
                  bool sizeAverage);

TH_API void THNN_(MSECriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *gradInput,
                  bool sizeAverage);

TH_API void THNN_(MultiLabelMarginCriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCIndexTensor *target,
                  THCTensor *output,
                  THCTensor *istarget,
                  bool sizeaverage);

TH_API void THNN_(MultiLabelMarginCriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCIndexTensor *target,
                  THCTensor *gradInput,
                  THCTensor *istarget,
                  bool sizeaverage);

TH_API void THNN_(MultiMarginCriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCIndexTensor *target,
                  THCTensor *output,
                  bool sizeAverage,
                  int p,
                  THCTensor *weights,           // [OPTIONAL]
                  accreal margin);

TH_API void THNN_(MultiMarginCriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCIndexTensor *target,
                  THCTensor *gradInput,
                  bool sizeAverage,
                  int p,
                  THCTensor *weights,           // [OPTIONAL]
                  accreal margin);

TH_API void THNN_(PReLU_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  long nOutputPlane);

TH_API void THNN_(PReLU_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *weight,
                  long nOutputPlane);

TH_API void THNN_(PReLU_accGradParameters)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *weight,
                  THCTensor *gradWeight,
                  THCTensor *gradWeightBuf,
                  THCTensor *gradWeightBuf2,
                  long nOutputPlane,
                  accreal scale);

TH_API void THNN_(SmoothL1Criterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *output,
                  bool sizeAverage);

TH_API void THNN_(SmoothL1Criterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *gradInput,
                  bool sizeAverage);

TH_API void THNN_(SparseLinear_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias);

TH_API void THNN_(SparseLinear_accGradParameters)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,
                  THCTensor *weight,
                  THCTensor *bias,
                  accreal weightDecay,
                  accreal scale);

TH_API void THNN_(SparseLinear_legacyUpdateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias);

TH_API void THNN_(SparseLinear_legacyAccGradParameters)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,
                  THCTensor *weight,
                  THCTensor *bias,
                  accreal weightDecay,
                  accreal scale);

TH_API void THNN_(SparseLinear_zeroGradParameters)(
                  THCState *state,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,
                  THCTensor *lastInput);

TH_API void THNN_(SparseLinear_updateParameters)(
                  THCState *state,
                  THCTensor *weight,
                  THCTensor *bias,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,
                  THCTensor *lastInput,
                  accreal learningRate);

TH_API void THNN_(IndexLinear_updateOutput)(
                  THCState *state,
                  THCIndexTensor *keys,
                  long keysOffset,
                  THCTensor *values,
                  THCIndexTensor *sizes,
                  THCIndexTensor *cumSumSizes,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias,
                  THCTensor *normalizedValues,
                  int   train);

TH_API void THNN_(IndexLinear_accGradParameters)(
                  THCState *state,
                  THCIndexTensor *keys,
                  long keysOffset,
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

TH_API void THNN_(IndexLinear_accUpdateGradParameters)(
                  THCState *state,
                  THCIndexTensor *keys,
                  long keysOffset,
                  THCTensor *values,
                  THCIndexTensor *sizes,
                  THCIndexTensor *cumSumSizes,
                  THCTensor *gradOutput,
                  THCTensor *weight,
                  THCTensor *bias,
                  accreal weightDecay,
                  accreal scale);

TH_API void THNN_(IndexLinear_updateParameters)(
                  THCState *state,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,
                  THCTensor *weight,
                  THCTensor *bias,
                  THCIndexTensor *runningKeys,
                  THCIndexTensor *cumSumSizes,
                  long keysOffset,
                  accreal weightDecay,
                  accreal learningRate);

TH_API void THNN_(SpatialAdaptiveMaxPooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCIndexTensor *indices,
                  int nOutputCols,
                  int nOutputRows);

TH_API void THNN_(SpatialAdaptiveMaxPooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCIndexTensor *indices);

TH_API void THNN_(SpatialAdaptiveAveragePooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int nOutputCols,
                  int nOutputRows);

TH_API void THNN_(SpatialAdaptiveAveragePooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput);

TH_API void THNN_(SpatialAveragePooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  bool ceil_mode,
                  bool count_include_pad);

TH_API void THNN_(SpatialAveragePooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  bool ceil_mode,
                  bool count_include_pad);

TH_API void THNN_(SpatialClassNLLCriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCIndexTensor *target,
                  THCTensor *output,
                  bool sizeAverage,
                  THCTensor *weights,       // [OPTIONAL]
                  THCTensor *total_weight);

TH_API void THNN_(SpatialClassNLLCriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCIndexTensor *target,
                  THCTensor *gradInput,
                  bool sizeAverage,
                  THCTensor *weights,       // [OPTIONAL]
                  THCTensor *total_weight);

TH_API void THNN_(SpatialConvolutionLocal_updateOutput)(
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
                  long inputWidth, long inputHeight,
                  long outputWidth, long outputHeight);

TH_API void THNN_(SpatialConvolutionLocal_updateGradInput)(
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
                  long inputWidth, long inputHeight,
                  long outputWidth, long outputHeight);

TH_API void THNN_(SpatialConvolutionLocal_accGradParameters)(
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
                  long inputWidth, long inputHeight,
                  long outputWidth, long outputHeight,
                  accreal scale);

TH_API void THNN_(SpatialConvolutionMM_updateOutput)(
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

TH_API void THNN_(SpatialConvolutionMM_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *weight,
                  THCTensor *gradColumns,
                  THCTensor *ones,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH);

TH_API void THNN_(SpatialConvolutionMM_accGradParameters)(
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

TH_API void THNN_(SpatialDepthWiseConvolution_updateOutput)(
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

TH_API void THNN_(SpatialDepthWiseConvolution_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *weight,
                  THCTensor *gradColumns,
                  THCTensor *ones,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH);

TH_API void THNN_(SpatialDepthWiseConvolution_accGradParameters)(
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


TH_API void THNN_(SpatialCrossMapLRN_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *scale,
                  int size,
                  accreal alpha,
                  accreal beta,
                  accreal k);

TH_API void THNN_(SpatialCrossMapLRN_updateGradInput)(
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

TH_API void THNN_(SpatialDilatedConvolution_updateOutput)(
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

TH_API void THNN_(SpatialDilatedConvolution_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *weight,
                  THCTensor *gradColumns,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int dilationW, int dilationH);

TH_API void THNN_(SpatialDilatedConvolution_accGradParameters)(
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

TH_API void THNN_(SpatialDilatedMaxPooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCIndexTensor *indices,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int dilationW, int dilationH,
                  bool ceil_mode);

TH_API void THNN_(SpatialDilatedMaxPooling_updateGradInput)(
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

TH_API void THNN_(SpatialFractionalMaxPooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int outputW, int outputH,
                  int poolSizeW, int poolSizeH,
                  THCIndexTensor *indices,
                  THCTensor *randomSamples);

TH_API void THNN_(SpatialFractionalMaxPooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int outputW, int outputH,
                  int poolSizeW, int poolSizeH,
                  THCIndexTensor *indices);

TH_API void THNN_(SpatialFullConvolution_updateOutput)(
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

TH_API void THNN_(SpatialFullConvolution_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *weight,
                  THCTensor *gradColumns,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int adjW, int adjH);

TH_API void THNN_(SpatialFullConvolution_accGradParameters)(
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

TH_API void THNN_(SpatialMaxPooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCIndexTensor *indices,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  bool ceil_mode);

TH_API void THNN_(SpatialMaxPooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCIndexTensor *indices,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  bool ceil_mode);

TH_API void THNN_(SpatialMaxUnpooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCIndexTensor *indices,
                  int owidth, int oheight);

TH_API void THNN_(SpatialMaxUnpooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCIndexTensor *indices,
                  int owidth, int oheight);

TH_API void THNN_(SpatialReflectionPadding_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int padL, int padR,
                  int padT, int padB);

TH_API void THNN_(SpatialReflectionPadding_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int padL, int padR,
                  int padT, int padB);

TH_API void THNN_(SpatialReplicationPadding_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int padL, int padR,
                  int padT, int padB);

TH_API void THNN_(SpatialReplicationPadding_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int padL, int padR,
                  int padT, int padB);

TH_API void THNN_(SpatialSubSampling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias,
                  int kW, int kH,
                  int dW, int dH);

TH_API void THNN_(SpatialSubSampling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *weight,
                  int kW, int kH,
                  int dW, int dH);

TH_API void THNN_(SpatialSubSampling_accGradParameters)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,
                  int kW, int kH,
                  int dW, int dH,
                  accreal scale);

TH_API void THNN_(SpatialUpSamplingBilinear_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int outputHeight,
                  int outputWidth);

TH_API void THNN_(SpatialUpSamplingBilinear_updateGradInput)(
                  THCState *state,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int nbatch,
                  int nchannels,
                  int inputHeight,
                  int inputWidth,
                  int outputHeight,
                  int outputWidth);

TH_API void THNN_(SpatialUpSamplingNearest_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int scale_factor);

TH_API void THNN_(SpatialUpSamplingNearest_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int scale_factor);

TH_API void THNN_(RReLU_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *noise,
                  double lower,
                  double upper,
                  bool train,
                  bool inplace,
                  void *generator);

TH_API void THNN_(RReLU_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *noise,
                  double lower,
                  double upper,
                  bool train,
                  bool inplace);

TH_API void THNN_(Sigmoid_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output);

TH_API void THNN_(Sigmoid_updateGradInput)(
                  THCState *state,
                  THCTensor *input,          // [OPTIONAL]
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *output);

TH_API void THNN_(SoftMarginCriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *output,
                  bool sizeAverage);

TH_API void THNN_(SoftMarginCriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *gradInput,
                  bool sizeAverage);

TH_API void THNN_(SoftMax_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output);

TH_API void THNN_(SoftMax_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *output);

TH_API void THNN_(SoftPlus_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  accreal beta,
                  accreal threshold);

TH_API void THNN_(SoftPlus_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *output,
                  accreal beta,
                  accreal threshold);

TH_API void THNN_(SoftShrink_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  accreal lambda);

TH_API void THNN_(SoftShrink_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  accreal lambda);

TH_API void THNN_(Square_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output);

TH_API void THNN_(Square_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput);

TH_API void THNN_(Sqrt_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  accreal eps);

TH_API void THNN_(Sqrt_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *output);

TH_API void THNN_(Tanh_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output);

TH_API void THNN_(Tanh_updateGradInput)(
                  THCState *state,
                  THCTensor *input,          // [OPTIONAL]
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *output);

TH_API void THNN_(TemporalConvolution_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias,
                  int kW, int dW,
                  int inputFrameSize,
                  int outputFrameSize);

TH_API void THNN_(TemporalConvolution_updateGradInput)(
                  THCState* state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *weight,
                  int kW, int dW);

TH_API void THNN_(TemporalConvolution_accGradParameters)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradWeight,
                  THCTensor *gradBias,
                  int kW, int dW,
                  accreal scale);

TH_API void THNN_(TemporalMaxPooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCIndexTensor *indices,
                  int kW, int dW);

TH_API void THNN_(TemporalMaxPooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCIndexTensor *indices,
                  int kW, int dW);

TH_API void THNN_(TemporalRowConvolution_updateOutput)(
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

TH_API void THNN_(TemporalRowConvolution_updateGradInput)(
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

TH_API void THNN_(TemporalRowConvolution_accGradParameters)(
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

TH_API void THNN_(Threshold_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  accreal threshold,
                  accreal val,
                  bool inplace);

TH_API void THNN_(Threshold_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  accreal threshold,
                  accreal val,
                  bool inplace);

TH_API void THNN_(VolumetricAveragePooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int kT, int kW, int kH,
                  int dT, int dW, int dH);

TH_API void THNN_(VolumetricAveragePooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int kT, int kW, int kH,
                  int dT, int dW, int dH);

TH_API void THNN_(VolumetricConvolution_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias,         // [OPTIONAL]
                  THCTensor *finput,
                  THCTensor *fgradInput,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH);

TH_API void THNN_(VolumetricConvolution_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *weight,
                  THCTensor *finput,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH);

TH_API void THNN_(VolumetricConvolution_accGradParameters)(
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

TH_API void THNN_(VolumetricDilatedConvolution_updateOutput)(
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

TH_API void THNN_(VolumetricDilatedConvolution_updateGradInput)(
                  THCState *state,
                  THCTensor  *input,
                  THCTensor  *gradOutput,
                  THCTensor  *gradInput,
                  THCTensor  *weight,
                  THCTensor  *gradColumns,
                  int kT, int kW, int kH,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  int dilationT, int dilationW, int dilationH);

TH_API void THNN_(VolumetricDilatedConvolution_accGradParameters)(
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

TH_API void THNN_(VolumetricDilatedMaxPooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCIndexTensor *indices,
                  int kT, int kW, int kH,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  int dilationT, int dilationW, int dilationH,
                  bool ceilMode);

TH_API void THNN_(VolumetricDilatedMaxPooling_updateGradInput)(
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

TH_API void THNN_(VolumetricFractionalMaxPooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int outputT, int outputW, int outputH,
                  int poolSizeT, int poolSizeW, int poolSizeH,
                  THCIndexTensor *indices,
                  THCTensor *randomSamples);

TH_API void THNN_(VolumetricFractionalMaxPooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int outputT, int outputW, int outputH,
                  int poolSizeT, int poolSizeW, int poolSizeH,
                  THCIndexTensor *indices);

TH_API void THNN_(VolumetricFullConvolution_updateOutput)(
                  THCState *state,
                  THCTensor  *input,
                  THCTensor  *output,
                  THCTensor  *weight,
                  THCTensor  *bias,        // [OPTIONAL]
                  THCTensor  *finput,
                  THCTensor  *fgradInput,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  int adjT, int adjW, int adjH);

TH_API void THNN_(VolumetricFullConvolution_updateGradInput)(
                  THCState *state,
                  THCTensor  *input,
                  THCTensor  *gradOutput,
                  THCTensor  *gradInput,
                  THCTensor  *weight,
                  THCTensor  *finput,
                  THCTensor  *fgradInput,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  int adjT, int adjW, int adjH);

TH_API void THNN_(VolumetricFullConvolution_accGradParameters)(
                  THCState *state,
                  THCTensor  *input,
                  THCTensor  *gradOutput,
                  THCTensor  *gradWeight,
                  THCTensor  *gradBias,    // [OPTIONAL]
                  THCTensor  *finput,
                  THCTensor  *fgradInput,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  int adjT, int adjW, int adjH,
                  accreal scale);

TH_API void THNN_(VolumetricMaxPooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCIndexTensor *indices,
                  int kT, int kW, int kH,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  bool ceilMode);

TH_API void THNN_(VolumetricMaxPooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCIndexTensor *indices,
                  int kT, int kW, int kH,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH,
                  bool ceilMode);

TH_API void THNN_(VolumetricMaxUnpooling_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCIndexTensor *indices,
                  int outputTime, int outputWidth, int outputHeight,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH);

TH_API void THNN_(VolumetricMaxUnpooling_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCIndexTensor *indices,
                  int outputTime, int outputWidth, int outputHeight,
                  int dT, int dW, int dH,
                  int padT, int padW, int padH);

TH_API void THNN_(VolumetricReplicationPadding_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int pleft, int pright,
                  int ptop, int pbottom,
                  int pfront, int pback);

TH_API void THNN_(VolumetricReplicationPadding_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int pleft, int pright,
                  int ptop, int pbottom,
                  int pfront, int pback);

TH_API void THNN_(VolumetricUpSamplingNearest_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  int scale_factor);

TH_API void THNN_(VolumetricUpSamplingNearest_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int scale_factor);

TH_API void THNN_(VolumetricUpSamplingTrilinear_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  int outputDepth,
                  int outputHeight,
                  int outputWidth);

TH_API void THNN_(VolumetricUpSamplingTrilinear_updateGradInput)(
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

#endif
