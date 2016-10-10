#include <THC/THC.h>
#include <THC/THCApply.cuh>

#define THIndexTensor THCudaLongTensor
#define THIndexTensor_(NAME) THCudaLongTensor_ ## NAME

#define THNN_(NAME) TH_CONCAT_3(THNN_, CReal, NAME)

TH_API void THNN_CudaAbsCriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *output,
          bool sizeAverage);
TH_API void THNN_CudaAbsCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *gradInput,
          bool sizeAverage);

TH_API void THNN_CudaBCECriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *output,
          bool sizeAverage,
          THCudaTensor *weights);      // [OPTIONAL]
TH_API void THNN_CudaBCECriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *gradInput,
          bool sizeAverage,
          THCudaTensor *weights);      // [OPTIONAL]

TH_API void THNN_CudaClassNLLCriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THIndexTensor *target,
          THCudaTensor *output,
          bool sizeAverage,
          THCudaTensor *weights,       // [OPTIONAL]
          THCudaTensor *total_weight);
TH_API void THNN_CudaClassNLLCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THIndexTensor *target,
          THCudaTensor *gradInput,
          bool sizeAverage,
          THCudaTensor *weights,       // [OPTIONAL]
          THCudaTensor *total_weight);

TH_API void THNN_CudaSpatialClassNLLCriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THIndexTensor *target,
          THCudaTensor *output,
          bool sizeAverage,
          THCudaTensor *weights,       // [OPTIONAL]
          THCudaTensor *total_weight);
TH_API void THNN_CudaSpatialClassNLLCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THIndexTensor *target,
          THCudaTensor *gradInput,
          bool sizeAverage,
          THCudaTensor *weights,       // [OPTIONAL]
          THCudaTensor *total_weight);

TH_API void THNN_CudaDistKLDivCriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *output,
          bool sizeAverage);
TH_API void THNN_CudaDistKLDivCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *gradInput,
          bool sizeAverage);

TH_API void THNN_CudaL1Cost_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output);
TH_API void THNN_CudaL1Cost_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,    // [OPTIONAL]
          THCudaTensor *gradInput);

TH_API void THNN_CudaLookupTable_accGradParameters(
          THCState *state,
          THIndexTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradWeight,
          THIndexTensor *count,
          THIndexTensor *sorted,        // [OPTIONAL]
          THIndexTensor *indices,       // [OPTIONAL]
          bool scaleGradByFreq,
          int paddingValue,
          float scale);

TH_API void THNN_CudaLookupTable_renorm(
          THCState *state,
          THIndexTensor *idx,
          THCudaTensor *weight,
          float maxNorm,
          float normType);

TH_API void THNN_CudaMarginCriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *output,
          bool sizeAverage,
          float margin);
TH_API void THNN_CudaMarginCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *gradInput,
          bool sizeAverage,
          float margin);

TH_API void THNN_CudaSoftMarginCriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *output,
          int sizeAverage);

TH_API void THNN_CudaSoftMarginCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *gradInput,
          int sizeAverage);

TH_API void THNN_CudaMSECriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *output,
          bool sizeAverage);
TH_API void THNN_CudaMSECriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *gradInput,
          bool sizeAverage);

TH_API void THNN_CudaMultiMarginCriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *output,
          bool sizeAverage,
          int p,
          THCudaTensor *weights,       // [OPTIONAL]
          float margin);
TH_API void THNN_CudaMultiMarginCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *gradInput,
          bool sizeAverage,
          int p,
          THCudaTensor *weights,       // [OPTIONAL]
          float margin);

TH_API void THNN_CudaMultiLabelMarginCriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *output,
          THCudaTensor *istarget,
          bool sizeAverage);
TH_API void THNN_CudaMultiLabelMarginCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *gradInput,
          THCudaTensor *istarget,
          bool sizeAverage);

TH_API void THNN_CudaPReLU_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *weight,
          long nOutputPlane);
TH_API void THNN_CudaPReLU_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *weight,
          long nOutputPlane);
TH_API void THNN_CudaPReLU_accGradParameters(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *weight,
          THCudaTensor *gradWeight,
          THCudaTensor *gradWeightBuf,
          THCudaTensor *gradWeightBuf2,
          long nOutputPlane,
          float scale);

TH_API void THNN_CudaRReLU_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *noise,
          double lower,
          double upper,
          bool train,
          bool inplace,
          void *generator);
TH_API void THNN_CudaRReLU_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *noise,
          double lower,
          double upper,
          bool train,
          bool inplace);

TH_API void THNN_CudaSmoothL1Criterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *output,
          bool sizeAverage);
TH_API void THNN_CudaSmoothL1Criterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *gradInput,
          bool sizeAverage);

TH_API void THNN_CudaTemporalConvolution_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *weight,
          THCudaTensor *bias,
          int kW, int dW,
          int inputFrameSize,
          int outputFrameSize);

TH_API void THNN_CudaTemporalConvolution_updateGradInput(
          THCState* state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *weight,
          int kW, int dW);

TH_API void THNN_CudaTemporalConvolution_accGradParameters(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradWeight,
          THCudaTensor *gradBias,
          int kW, int dW,
          float scale);

TH_API void THNN_CudaTemporalMaxPooling_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *indices,
          int kW, int dW);

TH_API void THNN_CudaTemporalMaxPooling_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *indices,
          int kW, int dW);

TH_API void THNN_CudaSparseLinear_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *weight,
          THCudaTensor *bias);
TH_API void THNN_CudaSparseLinear_accGradParameters(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradWeight,
          THCudaTensor *gradBias,
          THCudaTensor *weight,
          THCudaTensor *bias,
          double weightDecay,
          double scale);
TH_API void THNN_CudaSparseLinear_legacyUpdateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *weight,
          THCudaTensor *bias);
TH_API void THNN_CudaSparseLinear_legacyAccGradParameters(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradWeight,
          THCudaTensor *gradBias,
          THCudaTensor *weight,
          THCudaTensor *bias,
          double weightDecay,
          double scale);
TH_API void THNN_CudaSparseLinear_zeroGradParameters(
          THCState *state,
          THCudaTensor *gradWeight,
          THCudaTensor *gradBias,
          THCudaTensor *lastInput);
TH_API void THNN_CudaSparseLinear_updateParameters(
          THCState *state,
          THCudaTensor *weight,
          THCudaTensor *bias,
          THCudaTensor *gradWeight,
          THCudaTensor *gradBias,
          THCudaTensor *lastInput,
          double learningRate);

TH_API void THNN_CudaBatchNormalization_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *weight,        // [OPTIONAL]
          THCudaTensor *bias,          // [OPTIONAL]
          THCudaTensor *runningMean,
          THCudaTensor *runningVar,
          THCudaTensor *saveMean,
          THCudaTensor *saveStd,
          bool train,
          double momentum,
          double eps);
TH_API void THNN_CudaBatchNormalization_backward(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,     // [OPTIONAL]
          THCudaTensor *gradWeight,    // [OPTIONAL]
          THCudaTensor *gradBias,      // [OPTIONAL]
          THCudaTensor *weight,        // [OPTIONAL]
          THCudaTensor *running_mean,
          THCudaTensor *running_var,
          THCudaTensor *save_mean,
          THCudaTensor *save_std,
          bool train,
          float scale,
          double eps);

TH_API void THNN_CudaSpatialConvolutionMM_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *weight,
          THCudaTensor *bias,          // [OPTIONAL]
          THCudaTensor *columns,
          THCudaTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH);
TH_API void THNN_CudaSpatialConvolutionMM_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *weight,
          THCudaTensor *columns,
          THCudaTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH);
TH_API void THNN_CudaSpatialConvolutionMM_accGradParameters(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradWeight,
          THCudaTensor *gradBias,      // [OPTIONAL]
          THCudaTensor *columns,
          THCudaTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          float scale);

TH_API void THNN_CudaSpatialConvolutionLocal_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *weight,
          THCudaTensor *bias,
          THCudaTensor *finput,
          THCudaTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          long inputWidth, long inputHeight,
          long outputWidth, long outputHeight);
TH_API void THNN_CudaSpatialConvolutionLocal_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *weight,
          THCudaTensor *finput,
          THCudaTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          long inputWidth, long inputHeight,
          long outputWidth, long outputHeight);
TH_API void THNN_CudaSpatialConvolutionLocal_accGradParameters(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradWeight,
          THCudaTensor *gradBias,
          THCudaTensor *finput,
          THCudaTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          long inputWidth, long inputHeight,
          long outputWidth, long outputHeight,
          float scale);

TH_API void THNN_CudaSpatialFullConvolution_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *weight,
          THCudaTensor *bias,          // [OPTIONAL]
          THCudaTensor *columns,
          THCudaTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int adjW, int adjH);
TH_API void THNN_CudaSpatialFullConvolution_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *weight,
          THCudaTensor *gradColumns,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int adjW, int adjH);
TH_API void THNN_CudaSpatialFullConvolution_accGradParameters(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradWeight,
          THCudaTensor *gradBias,      // [OPTIONAL]
          THCudaTensor *columns,
          THCudaTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int adjW, int adjH,
          float scale);

TH_API void THNN_CudaSpatialDilatedConvolution_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *weight,
          THCudaTensor *bias,          // [OPTIONAL]
          THCudaTensor *columns,
          THCudaTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int dilationW, int dilationH);

TH_API void THNN_CudaSpatialDilatedConvolution_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *weight,
          THCudaTensor *gradColumns,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int dilationW, int dilationH );

TH_API void THNN_CudaSpatialDilatedConvolution_accGradParameters(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradWeight,
          THCudaTensor *gradBias,      // [OPTIONAL]
          THCudaTensor *columns,
          THCudaTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int dilationW, int dilationH,
          float scale);

TH_API void THNN_CudaSpatialCrossMapLRN_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *scale,
          int size,
          float alpha,
          float beta,
          float k);
TH_API void THNN_CudaSpatialCrossMapLRN_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *scale,
          THCudaTensor *output,
          int size,
          float alpha,
          float beta,
          float k);

TH_API void THNN_CudaSpatialAdaptiveMaxPooling_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *indices,
          int nOutputCols,
          int nOutputRows);
TH_API void THNN_CudaSpatialAdaptiveMaxPooling_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *indices);

TH_API void THNN_CudaSpatialAveragePooling_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          bool ceil_mode,
          bool count_include_pad);
TH_API void THNN_CudaSpatialAveragePooling_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          bool ceil_mode,
          bool count_include_pad);

TH_API void THNN_CudaSpatialMaxPooling_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *indices,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          bool ceil_mode);
TH_API void THNN_CudaSpatialMaxPooling_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *indices,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          bool ceil_mode);

TH_API void THNN_CudaSpatialDilatedMaxPooling_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *indices,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int dilationW, int dilationH,
          bool ceil_mode);
TH_API void THNN_CudaSpatialDilatedMaxPooling_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *indices,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int dilationW, int dilationH,
          bool ceil_mode);

TH_API void THNN_CudaSpatialMaxUnpooling_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *indices,
          int owidth, int oheight);
TH_API void THNN_CudaSpatialMaxUnpooling_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *indices,
          int owidth, int oheight);

TH_API void THNN_CudaSpatialFractionalMaxPooling_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          int outputW, int outputH,
          int poolSizeW, int poolSizeH,
          THCudaTensor *indices,
          THCudaTensor *randomSamples);
TH_API void THNN_CudaSpatialFractionalMaxPooling_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          int outputW, int outputH,
          int poolSizeW, int poolSizeH,
          THCudaTensor *indices);

TH_API void THNN_CudaSpatialSubSampling_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *weight,
          THCudaTensor *bias,
          int kW, int kH,
          int dW, int dH);
TH_API void THNN_CudaSpatialSubSampling_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *weight,
          int kW, int kH,
          int dW, int dH);
TH_API void THNN_CudaSpatialSubSampling_accGradParameters(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradWeight,
          THCudaTensor *gradBias,
          int kW, int kH,
          int dW, int dH,
          float scale);

TH_API void THNN_CudaSpatialUpSamplingNearest_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          int scale_factor);
TH_API void THNN_CudaSpatialUpSamplingNearest_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          int scale_factor);

TH_API void THNN_CudaSpatialUpSamplingBilinear_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
	  int outputHeight,
          int outputWidth);
TH_API void THNN_CudaSpatialUpSamplingBilinear_updateGradInput(
          THCState *state,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          int nbatch,
          int nchannels,
          int inputHeight,
          int inputWidth,
          int outputHeight,
          int outputWidth);

TH_API void THNN_CudaVolumetricAveragePooling_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          int kT, int kW, int kH,
          int dT, int dW, int dH);
TH_API void THNN_CudaVolumetricAveragePooling_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          int kT, int kW, int kH,
          int dT, int dW, int dH);

TH_API void THNN_CudaVolumetricConvolution_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *weight,
          THCudaTensor *bias,
          THCudaTensor *finput,
          THCudaTensor *fgradInput,
          int dT, int dW, int dH,
          int padT, int padW, int padH);
TH_API void THNN_CudaVolumetricConvolution_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *weight,
          THCudaTensor *finput,
          int dT, int dW, int dH,
          int padT, int padW, int padH);
TH_API void THNN_CudaVolumetricConvolution_accGradParameters(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradWeight,
          THCudaTensor *gradBias,
          THCudaTensor *finput,
          THCudaTensor *fgradInput,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          float scale);

TH_API void THNN_CudaVolumetricFullConvolution_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *weight,
          THCudaTensor *bias,
          THCudaTensor *finput,
          THCudaTensor *fgradInput,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int adjT, int adjW, int adjH);
TH_API void THNN_CudaVolumetricFullConvolution_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *weight,
          THCudaTensor *finput,
          THCudaTensor *fgradInput,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int adjT, int adjW, int adjH);
TH_API void THNN_CudaVolumetricFullConvolution_accGradParameters(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradWeight,
          THCudaTensor *gradBias,
          THCudaTensor *finput,
          THCudaTensor *fgradInput,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int adjT, int adjW, int adjH,
          float scale);

TH_API void THNN_CudaVolumetricDilatedConvolution_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *weight,
          THCudaTensor *bias,
          THCudaTensor *columns,
          THCudaTensor *ones,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int dilationT, int dilationW, int dilationH);

TH_API void THNN_CudaVolumetricDilatedConvolution_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *weight,
          THCudaTensor *gradColumns,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int dilationT, int dilationW, int dilationH);

TH_API void THNN_CudaVolumetricDilatedConvolution_accGradParameters(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradWeight,
          THCudaTensor *gradBias,
          THCudaTensor *columns,
          THCudaTensor *ones,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int dilationT, int dilationW, int dilationH,
          float scale);

TH_API void THNN_CudaVolumetricMaxPooling_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *indices,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          bool ceilMode);
TH_API void THNN_CudaVolumetricMaxPooling_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *indices,
          int dT, int dW, int dH,
          int padT, int padW, int padH);

TH_API void THNN_CudaVolumetricDilatedMaxPooling_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *indices,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int dilationT, int dilationW, int dilationH,
          bool ceilMode);
TH_API void THNN_CudaVolumetricDilatedMaxPooling_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *indices,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int dilationT, int dilationW, int dilationH);

TH_API void THNN_CudaVolumetricMaxUnpooling_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *indices,
          int outputTime, int outputWidth, int outputHeight,
          int dT, int dW, int dH,
          int padT, int padW, int padH);
TH_API void THNN_CudaVolumetricMaxUnpooling_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *indices,
          int outputTime, int outputWidth, int outputHeight,
          int dT, int dW, int dH,
          int padT, int padW, int padH);

TH_API void THNN_CudaSpatialReflectionPadding_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          int padL, int padR,
          int padT, int padB);
TH_API void THNN_CudaSpatialReflectionPadding_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          int padL, int padR,
          int padT, int padB);

TH_API void THNN_CudaSpatialReplicationPadding_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          int padL, int padR,
          int padT, int padB);
TH_API void THNN_CudaSpatialReplicationPadding_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          int padL, int padR,
          int padT, int padB);

TH_API void THNN_CudaVolumetricReplicationPadding_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          int pleft, int pright,
          int ptop, int pbottom,
          int pfront, int pback);
TH_API void THNN_CudaVolumetricReplicationPadding_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          int pleft, int pright,
          int ptop, int pbottom,
          int pfront, int pback);

#include "generic/THCUNN.h"
#include "THCGenerateFloatTypes.h"
