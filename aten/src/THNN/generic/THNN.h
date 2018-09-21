#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THNN.h"
#else

#include <ATen/core/Reduction.h>

TH_API void THNN_(BCECriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          int64_t reduction,
          THTensor *weights);          // [OPTIONAL]
TH_API void THNN_(BCECriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradOutput,
          THTensor *gradInput,
          int64_t reduction,
          THTensor *weights);          // [OPTIONAL]

TH_API void THNN_(GatedLinear_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *output,            // [OUT] output tensor, half size of input along dimension dim
          int dim);                    // dimension for halving operation
TH_API void THNN_(GatedLinear_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *gradOutput,        // gradient w.r.t module's output
          THTensor *gradInput,         // [OUT] gradient w.r.t input
          int dim);                    // dimension for halving operation

TH_API void THNN_(Im2Col_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int64_t kH, int64_t kW,
          int64_t dH, int64_t dW,
          int64_t padH, int64_t padW,
          int64_t sH, int64_t sW);

TH_API void THNN_(Im2Col_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          int64_t inputHeight, int64_t inputWidth,
          int64_t kH, int64_t kW,
          int64_t dH, int64_t dW,
          int64_t padH, int64_t padW,
          int64_t sH, int64_t sW);

TH_API void THNN_(Col2Im_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int64_t outputHeight, int64_t outputWidth,
          int64_t kH, int64_t kW,
          int64_t dH, int64_t dW,
          int64_t padH, int64_t padW,
          int64_t sH, int64_t sW);

TH_API void THNN_(Col2Im_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          int64_t kH, int64_t kW,
          int64_t dH, int64_t dW,
          int64_t padH, int64_t padW,
          int64_t sH, int64_t sW);

TH_API void THNN_(MSECriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          int64_t reduction);
TH_API void THNN_(MSECriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradOutput,
          THTensor *gradInput,
          int64_t reduction);

TH_API void THNN_(IndexLinear_updateOutput)(
          THNNState *state,
          THIndexTensor *keys,
          int64_t keysOffset,
          THTensor *values,
          THIndexTensor *sizes,
          THIndexTensor *cumSumSizes,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *normalizedValues,
          int train);
TH_API void THNN_(IndexLinear_accGradParameters)(
          THNNState *state,
          THIndexTensor *keys,
          int64_t keysOffset,
          THTensor *values,
          THIndexTensor *sizes,
          THIndexTensor *cumSumSizes,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *weight,
          THTensor *bias,
          THTensor* valuesBuffer,
          accreal weightDecay,
          accreal scale);
TH_API void THNN_(IndexLinear_accUpdateGradParameters)(
          THNNState *state,
          THIndexTensor *keys,
          int64_t keysOffset,
          THTensor *values,
          THIndexTensor *sizes,
          THIndexTensor *cumSumSizes,
          THTensor *gradOutput,
          THTensor *weight,
          THTensor *bias,
          accreal weightDecay,
          accreal scale);
TH_API void THNN_(IndexLinear_updateParameters)(
          THNNState *state,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *weight,
          THTensor *bias,
          THIndexTensor *runningKeys,
          THIndexTensor *cumSumSizes,
          int64_t keysOffset,
          accreal weightDecay,
          accreal learningRate);

TH_API void THNN_(SparseLinear_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias);
TH_API void THNN_(SparseLinear_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *weight,
          THTensor *bias,
          accreal weightDecay,
          accreal scale);
TH_API void THNN_(SparseLinear_zeroGradParameters)(
          THNNState *state,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *lastInput);
TH_API void THNN_(SparseLinear_updateParameters)(
          THNNState *state,
          THTensor *weight,
          THTensor *bias,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *lastInput,
          accreal learningRate);
TH_API void THNN_(SparseLinear_legacyUpdateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias);
TH_API void THNN_(SparseLinear_legacyAccGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *weight,
          THTensor *bias,
          accreal weightDecay,
          accreal scale);
TH_API void THNN_(SparseLinear_legacyZeroGradParameters)(
          THNNState *state,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *lastInput);
TH_API void THNN_(SparseLinear_legacyUpdateParameters)(
          THNNState *state,
          THTensor *weight,
          THTensor *bias,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *lastInput,
          accreal learningRate);

TH_API void THNN_(TemporalRowConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          THTensor *fgradInput,
          int kW,
          int dW,
          int padW,
          bool featFirst);
TH_API void THNN_(TemporalRowConvolution_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          THTensor *fgradInput,
          int kW,
          int dW,
          int padW,
          bool featFirst);
TH_API void THNN_(TemporalRowConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          THTensor *fgradInput,
          int kW,
          int dW,
          int padW,
          bool featFirst,
          accreal scale);

TH_API void THNN_(TemporalUpSamplingNearest_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int osizeW);
TH_API void THNN_(TemporalUpSamplingNearest_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          int isizeB,
          int isizeC,
          int isizeW,
          int osizeW);

TH_API void THNN_(TemporalUpSamplingLinear_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int osizeW,
          bool align_corners);
TH_API void THNN_(TemporalUpSamplingLinear_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          int isizeB,
          int isizeC,
          int isizeW,
          int osizeW,
          bool align_corners);

TH_API void THNN_(SpatialConvolutionMM_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,         // [OPTIONAL]
          THTensor *finput,
          THTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH);
TH_API void THNN_(SpatialConvolutionMM_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          THTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH);
TH_API void THNN_(SpatialConvolutionMM_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,     // [OPTIONAL]
          THTensor *finput,
          THTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          accreal scale);

TH_API void THNN_(SpatialAdaptiveAveragePooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int osizeW, int osizeH);
TH_API void THNN_(SpatialAdaptiveAveragePooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput);

TH_API void THNN_(SpatialFractionalMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int outputW, int outputH,
          int kW, int kH,
          THIndexTensor *indices,
          THTensor *randomSamples);
TH_API void THNN_(SpatialFractionalMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int outputW, int outputH,
          int kW, int kH,
          THIndexTensor *indices);

TH_API void THNN_(SpatialFullDilatedConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,         // [OPTIONAL]
          THTensor *columns,
          THTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int dilationW, int dilationH,
          int adjW, int adjH);

TH_API void THNN_(SpatialFullDilatedConvolution_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *columns,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int dilationW, int dilationH,
          int adjW, int adjH);

TH_API void THNN_(SpatialFullDilatedConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,     // [OPTIONAL]
          THTensor *columns,
          THTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int dilationW, int dilationH,
          int adjW, int adjH,
          accreal scale);

TH_API void THNN_(SpatialDilatedMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int dilationW, int dilationH,
          bool ceil_mode);
TH_API void THNN_(SpatialDilatedMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int dilationW, int dilationH,
          bool ceil_mode);

TH_API void THNN_(SpatialUpSamplingBilinear_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int osizeH,
          int osizeW,
          bool align_corners);
TH_API void THNN_(SpatialUpSamplingBilinear_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          int isizeB,
          int isizeC,
          int isizeH,
          int isizeW,
          int osizeH,
          int osizeW,
          bool align_corners);

TH_API void THNN_(unfolded_acc)(
          THTensor *finput,
          THTensor *input,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int nInputPlane,
          int inputWidth, int inputHeight,
          int osizeW, int outputHeight);
TH_API void THNN_(unfolded_copy)(
          THTensor *finput,
          THTensor *input,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int nInputPlane,
          int inputWidth, int inputHeight,
          int outputWidth, int outputHeight);

TH_API void THNN_(VolumetricAveragePooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          bool ceil_mode, bool count_include_pad);
TH_API void THNN_(VolumetricAveragePooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          bool ceil_mode, bool count_include_pad);

TH_API void THNN_(VolumetricConvolutionMM_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,           // [OPTIONAL]
          THTensor *finput,
          THTensor *fgradInput,     // HACK to make signature line up with backward
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int pT, int pW, int pH);
TH_API void THNN_(VolumetricConvolutionMM_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          THTensor *fgradInput,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int pT, int pW, int pH);
TH_API void THNN_(VolumetricConvolutionMM_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,       // [OPTIONAL]
          THTensor *finput,
          THTensor *fgradInput,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int pT, int pW, int pH,
          accreal scale);

TH_API void THNN_(VolumetricFractionalMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int outputT, int outputW, int outputH,
          int poolSizeT, int poolSizeW, int poolSizeH,
          THIndexTensor *indices,
          THTensor *randomSamples);
TH_API void THNN_(VolumetricFractionalMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int outputT, int outputW, int outputH,
          int poolSizeT, int poolSizeW, int poolSizeH,
          THIndexTensor *indices);

TH_API void THNN_(VolumetricDilatedConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,           // [OPTIONAL]
          THTensor *columns,
          THTensor *ones,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int dilationT, int dilationW, int dilationH);

TH_API void THNN_(VolumetricDilatedConvolution_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *columns,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int dilationT, int dilationW, int dilationH);

TH_API void THNN_(VolumetricDilatedConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,       // [OPTIONAL]
          THTensor *columns,
          THTensor *ones,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int dilationT, int dilationW, int dilationH,
          accreal scale);

TH_API void THNN_(VolumetricFullDilatedConvolution_updateOutput)(
          THNNState *state,         // library state
          THTensor *input,          // 4D or 5D (batch) tensor
          THTensor *output,         // [OUT] volumetric convolution output
          THTensor *weight,         // weight tensor (nInputPlane x nOutputPlane x kT x kH x kW)
          THTensor *bias,           // [OPTIONAL] gradBias tensor (nOutputPlane)
          THTensor *finput,         // [OUT] internal columns buffer
          THTensor *fgradInput,     // [OUT] internal ones buffer
          int kT, int kW, int kH,   // kernel size
          int dT, int dW, int dH,   // stride of the convolution
          int pT, int pW, int pH,   // padding
          int dilationT, int dilationW, int dilationH,
          int aT, int aW, int aH);  // extra output adjustment
TH_API void THNN_(VolumetricFullDilatedConvolution_updateGradInput)(
          THNNState *state,         // library state
          THTensor *input,          // 4D or 5D (batch) tensor
          THTensor *gradOutput,     // gradient w.r.t. output
          THTensor *gradInput,      // [OUT] gradient w.r.t. input
          THTensor *weight,         // weight tensor (nInputPlane x nOutputPlane x kT x kH x kW)
          THTensor *finput,         // internal columns buffer
          THTensor *fgradInput,     // internal ones buffer
          int kT, int kW, int kH,   // kernel size
          int dT, int dW, int dH,   // stride
          int pT, int pW, int pH,   // padding
          int dilationT, int dilationW, int dilationH,
          int aT, int aW, int aH);  // extra output adjustment

TH_API void THNN_(VolumetricFullDilatedConvolution_accGradParameters)(
          THNNState *state,         // library state
          THTensor *input,          // 4D or 5D (batch) tensor
          THTensor *gradOutput,     // gradient w.r.t. output
          THTensor *gradWeight,     // gradWeight tensor (nInputPlane x nOutputPlane x kT x kH x kW)
          THTensor *gradBias,       // [OPTIONAL] gradBias tensor (nOutputPlane)
          THTensor *finput,         // internal columns buffer
          THTensor *fgradInput,     // internal ones buffer
          int kT, int kW, int kH,   // kernel size
          int dT, int dW, int dH,   // stride
          int pT, int pW, int pH,   // padding
          int dilationT, int dilationW, int dilationH,
          int aT, int aW, int aH,   // extra output adjustment
          accreal scale);           // scaling factor

TH_API void THNN_(VolumetricDilatedMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int pT, int pW, int pH,
          int dilationT, int dilationW, int dilationH,
          bool ceilMode);
TH_API void THNN_(VolumetricDilatedMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int pT, int pW, int pH,
          int dilationT, int dilationW, int dilationH,
          bool ceilMode);

TH_API void THNN_(VolumetricAdaptiveMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int osizeT, int osizeW, int osizeH);
TH_API void THNN_(VolumetricAdaptiveMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices);

TH_API void THNN_(FeatureLPPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal power,
          int width,
          int stride,
          bool batchMode);

TH_API void THNN_(FeatureLPPooling_updateGradInput)(
          THNNState *state,
          THTensor* gradOutput,
          THTensor* input,
          THTensor* output,
          THTensor* gradInput,
          accreal power,
          int width,
          int stride,
          bool batchMode);

TH_API void THNN_(VolumetricUpSamplingNearest_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int osizeT,
          int osizeH,
          int osizeW);

TH_API void THNN_(VolumetricUpSamplingNearest_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          int isizeB,
          int isizeC,
          int isizeT,
          int isizeH,
          int isizeW,
          int osizeT,
          int osizeH,
          int osizeW);

TH_API void THNN_(VolumetricUpSamplingTrilinear_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int osizeT,
          int osizeH,
          int osizeW,
          bool align_corners);

TH_API void THNN_(VolumetricUpSamplingTrilinear_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          int isizeB,
          int isizeC,
          int isizeT,
          int isizeH,
          int isizeW,
          int osizeT,
          int osizeH,
          int osizeW,
          bool align_corners);

TH_API void THNN_(TemporalReflectionPadding_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int pad_left, int pad_right);

TH_API void THNN_(TemporalReflectionPadding_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int pad_left, int pad_right);

TH_API void THNN_(TemporalReplicationPadding_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int pad_left, int pad_right);

TH_API void THNN_(TemporalReplicationPadding_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int pad_left, int pad_right);

#endif
