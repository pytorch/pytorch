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

TH_API void THNN_(ELU_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  real alpha,
                  bool inplace);

TH_API void THNN_(ELU_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *output,
                  real alpha,
                  bool inplace);

TH_API void THNN_(HardTanh_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  real min_val,
                  real max_val,
                  bool inplace);

TH_API void THNN_(HardTanh_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  real min_val,
                  real max_val,
                  bool inplace);

TH_API void THNN_(LeakyReLU_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  real negval,
                  bool inplace);

TH_API void THNN_(LeakyReLU_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  real negval,
                  bool inplace);

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

TH_API void THNN_(MarginCriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *output,
                  bool sizeAverage,
                  real margin);

TH_API void THNN_(MarginCriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *gradInput,
                  bool sizeAverage,
                  real margin);

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

TH_API void THNN_(MultiMarginCriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCIndexTensor *target,
                  THCTensor *output,
                  bool sizeAverage,
                  int p,
                  THCTensor *weights,       // [OPTIONAL]
                  real margin);

TH_API void THNN_(MultiMarginCriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCIndexTensor *target,
                  THCTensor *gradInput,
                  bool sizeAverage,
                  int p,
                  THCTensor *weights,       // [OPTIONAL]
                  real margin);

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
                  real scale);

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
                  real scale);

TH_API void THNN_(SpatialConvolutionMM_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias,
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
                  THCTensor *gradBias,
                  THCTensor *columns,
                  THCTensor *ones,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  real scale);

TH_API void THNN_(SpatialCrossMapLRN_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *scale,
                  int size,
                  real alpha,
                  real beta,
                  real k);

TH_API void THNN_(SpatialCrossMapLRN_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *scale,
                  THCTensor *output,
                  int size,
                  real alpha,
                  real beta,
                  real k);

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
                  real scale);

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
                  real scale);

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
                  float scale);

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
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *output);

TH_API void THNN_(SoftMarginCriterion_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *output,
                  int sizeAverage);

TH_API void THNN_(SoftMarginCriterion_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *target,
                  THCTensor *gradInput,
                  int sizeAverage);

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
                  real beta,
                  real threshold);

TH_API void THNN_(SoftPlus_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *output,
                  real beta,
                  real threshold);

TH_API void THNN_(SoftShrink_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  real lambda);

TH_API void THNN_(SoftShrink_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  real lambda);

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
                  real eps);

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
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *output);

TH_API void THNN_(Threshold_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  real threshold,
                  real val,
                  bool inplace);

TH_API void THNN_(Threshold_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  real threshold,
                  real val,
                  bool inplace);

#endif
