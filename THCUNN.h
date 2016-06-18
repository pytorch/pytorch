#include <THC/THC.h>
#include <THC/THCApply.cuh>

#define THIndexTensor THCudaTensor
#define THIndexTensor_(NAME) THCudaTensor_ ## NAME

#define THIntegerTensor THCudaTensor
#define THIntegerTensor_(NAME) THCudaTensor_ ## NAME

TH_API void THNN_CudaAbs_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output);
TH_API void THNN_CudaAbs_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput);

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

TH_API void THNN_CudaClassNLLCriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *output,
          bool sizeAverage,
          THCudaTensor *weights,
          THCudaTensor *total_weight);
TH_API void THNN_CudaClassNLLCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *gradInput,
          bool sizeAverage,
          THCudaTensor *weights,
          THCudaTensor *total_weight);

TH_API void THNN_CudaSpatialClassNLLCriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *output,
          bool sizeAverage,
          THCudaTensor *weights,
          THCudaTensor *total_weight);
TH_API void THNN_CudaSpatialClassNLLCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *gradInput,
          bool sizeAverage,
          THCudaTensor *weights,
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

TH_API void THNN_CudaELU_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          float alpha,
          bool inplace);
TH_API void THNN_CudaELU_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *output,
          float alpha,
          bool inplace);

TH_API void THNN_CudaHardTanh_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          float min_val,
          float max_val,
          bool inplace);
TH_API void THNN_CudaHardTanh_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          float min_val,
          float max_val,
          bool inplace);

TH_API void THNN_CudaL1Cost_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output);
TH_API void THNN_CudaL1Cost_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput);

TH_API void THNN_CudaLeakyReLU_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          double negval, bool inplace);
TH_API void THNN_CudaLeakyReLU_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          double negval,
          bool inplace);

TH_API void THNN_CudaLogSigmoid_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *buffer);
TH_API void THNN_CudaLogSigmoid_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *buffer);

TH_API void THNN_CudaLogSoftMax_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output);
TH_API void THNN_CudaLogSoftMax_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *output);

TH_API void THNN_CudaLookupTable_accGradParameters(
          THCState *state,
          THIndexTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradWeight,
          THIntegerTensor *count,
          THCudaTensor *sorted,
          THCudaTensor *indices,
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
          THCudaTensor *weights,
          float margin);
TH_API void THNN_CudaMultiMarginCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *gradInput,
          bool sizeAverage,
          int p,
          THCudaTensor *weights,
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

TH_API void THNN_CudaSigmoid_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output);
TH_API void THNN_CudaSigmoid_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *output);

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

TH_API void THNN_CudaSoftMax_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output);
TH_API void THNN_CudaSoftMax_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *output);

TH_API void THNN_CudaSoftPlus_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          float beta,
          float threshold);
TH_API void THNN_CudaSoftPlus_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *output,
          float beta,
          float threshold);

TH_API void THNN_CudaSoftShrink_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          double lambda);
TH_API void THNN_CudaSoftShrink_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          double lambda);

TH_API void THNN_CudaSqrt_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          float eps);
TH_API void THNN_CudaSqrt_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *output);

TH_API void THNN_CudaSquare_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output);
TH_API void THNN_CudaSquare_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput);

TH_API void THNN_CudaTanh_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output);
TH_API void THNN_CudaTanh_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *output);

TH_API void THNN_CudaThreshold_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          double threshold,
          double val,
          bool inplace);
TH_API void THNN_CudaThreshold_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          double threshold,
          bool inplace);

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
          THCudaTensor *weight,
          THCudaTensor *bias,
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
          THCudaTensor *gradInput,
          THCudaTensor *gradWeight,
          THCudaTensor *gradBias,
          THCudaTensor *weight,
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
          THCudaTensor *bias,
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
          THCudaTensor *gradBias,
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
          THCudaTensor *bias,
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
          THCudaTensor *gradBias,
          THCudaTensor *columns,
          THCudaTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int adjW, int adjH,
          float scale);
TH_API void THNN_CudaSpatialDilatedConvolution_updateOutput(THCState *state,
            THCudaTensor *input, THCudaTensor *output, THCudaTensor *weight,
            THCudaTensor *bias, THCudaTensor *columns,
            THCudaTensor *ones, int kW, int kH, int dW, int dH,
            int padW, int padH, int dilationW, int dilationH);
TH_API void THNN_CudaSpatialDilatedConvolution_updateGradInput(THCState *state,
               THCudaTensor *input, THCudaTensor *gradOutput,
               THCudaTensor *gradInput, THCudaTensor *weight,
               THCudaTensor *gradColumns,
               int kW, int kH, int dW, int dH, int padW, int padH,
               int dilationW, int dilationH );
TH_API void THNN_CudaSpatialDilatedConvolution_accGradParameters(THCState *state,
                     THCudaTensor *input, THCudaTensor *gradOutput,
                     THCudaTensor *gradWeight, THCudaTensor *gradBias,
                     THCudaTensor *columns, THCudaTensor *ones,
                     int kW, int kH, int dW, int dH,
                     int padW, int padH, int dilationW, int dilationH, float scale);

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

TH_API void THNN_CudaSpatialReflectionPadding_updateOutput(THCState *state,
                                                    THCudaTensor *input,
                                                    THCudaTensor *output,
                                                    int padL, int padR,
                                                    int padT, int padB
                                                   );
TH_API void THNN_CudaSpatialReflectionPadding_updateGradInput(THCState *state,
                                                       THCudaTensor *input,
                                                       THCudaTensor *gradOutput,
                                                       THCudaTensor *gradInput,
                                                       int padL, int padR,
                                                       int padT, int padB);

TH_API void THNN_CudaSpatialReplicationPadding_updateOutput(THCState *state,
                                                    THCudaTensor *input,
                                                    THCudaTensor *output,
                                                    int padL, int padR,
                                                    int padT, int padB
                                                   );
TH_API void THNN_CudaSpatialReplicationPadding_updateGradInput(THCState *state,
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
