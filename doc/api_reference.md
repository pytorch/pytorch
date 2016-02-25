# API docs

This document only describes a THNN API. For a thorough review of all modules present here please refer to [nn's docs](http://github.com/torch/nn/tree/master/doc).

### Note on function names

Please remember, that because C doesn't support function overloading, functions taking different tensor types have different names. So e.g. for an Abs module, there are actually two updateOutput functions:

* `void THNN_FloatAbs_updateOutput(...)`
* `void THNN_DoubleAbs_updateOutput(...)`

In these docs such function will be referred to as `void THNN_Abs_updateOutput(...)`, and it's up to developer to add a type prefix. `real` is an alias for that type.

### Argument types

Some arguments have additional tags placed in square brackets:
* **[OUT]** - This is the output argument. It will be reshaped if needed.
* **[OPTIONAL]** - This argument is optional and can be safely set to NULL
* **[BUFFER]** - A buffer. `updateGradInput` and `accGradParameters` should get the same buffers that were used in `updateOutput` call.
* **[MODIFIED]** - Some functions accept an `inplace` flag. If set to true, this argument might be modified (in addition to the output).

## Module list

These are all modules implemented in THNN:

* [Abs](#abs)
* [AbsCriterion](#abscriterion)
* [ClassNLLCriterion](#classnllcriterion)
* [DistKLDivCriterion](#distkldivcriterion)
* [ELU](#elu)
* [HardShrink](#hardshrink)
* [HardTanh](#hardtanh)
* [L1Cost](#l1cost)
* [LeakyReLU](#leakyrelu)
* [LogSigmoid](#logsigmoid)
* [LogSoftMax](#logsoftmax)
* [LookupTable](#lookuptable)
* [MSECriterion](#msecriterion)
* [MarginCriterion](#margincriterion)
* [MultiLabelMarginCriterion](#multilabelmargincriterion)
* [MultiMarginCriterion](#multimargincriterion)
* [PReLU](#prelu)
* [RReLU](#rrelu)
* [Sigmoid](#sigmoid)
* [SmoothL1Criterion](#smoothl1criterion)
* [SoftMax](#softmax)
* [SoftPlus](#softplus)
* [SoftShrink](#softshrink)
* [SparseLinear](#sparselinear)
* [SpatialAdaptiveMaxPooling](#spatialadaptivemaxpooling)
* [SpatialAveragePooling](#spatialaveragepooling)
* [SpatialBatchNormalization](#spatialbatchnormalization)
* [SpatialConvolutionLocal](#spatialconvolutionlocal)
* [SpatialConvolutionMM](#spatialconvolutionmm)
* [SpatialConvolutionMap](#spatialconvolutionmap)
* [SpatialFractionalMaxPooling](#spatialfractionalmaxpooling)
* [SpatialFullConvolution](#spatialfullconvolution)
* [SpatialFullConvolutionMap](#spatialfullconvolutionmap)
* [SpatialMaxPooling](#spatialmaxpooling)
* [SpatialMaxUnpooling](#spatialmaxunpooling)
* [SpatialSubSampling](#spatialsubsampling)
* [SpatialUpSamplingNearest](#spatialupsamplingnearest)
* [Sqrt](#sqrt)
* [Square](#square)
* [Tanh](#tanh)
* [Threshold](#threshold)
* [VolumetricAveragePooling](#volumetricaveragepooling)
* [VolumetricConvolution](#volumetricconvolution)
* [VolumetricConvolutionMM](#volumetricconvolutionmm)
* [VolumetricFullConvolution](#volumetricfullconvolution)
* [VolumetricMaxPooling](#volumetricmaxpooling)
* [VolumetricMaxUnpooling](#volumetricmaxunpooling)

## Abs
```C
void THNN_Abs_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *output` - **[OUT]** Abs output
<br/>
```C
void THNN_Abs_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *gradOutput` - gradient w.r.t. output
<br/>
`THTensor *gradInput` - **[OUT]** gradient w.r.t. input
<br/>
## AbsCriterion
```C
void THNN_AbsCriterion_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *target` - tensor with target values
<br/>
`THTensor *output` - **[OUT]** a one-element tensor with loss
<br/>
`bool sizeAverage` - if true, the loss will be divided by batch size
<br/>
```C
void THNN_AbsCriterion_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *target` - tensor with target values
<br/>
`THTensor *gradInput` - **[OUT]** gradient w.r.t. input
<br/>
`bool sizeAverage` - if true, the gradient will be normalized by batch size
<br/>
## ClassNLLCriterion
```C
void THNN_ClassNLLCriterion_updateOutput(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *output,
          bool sizeAverage,
          THTensor *weights,
          THTensor *total_weight);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input tensor (1D/2D)
<br/>
`THIndexTensor *target` - tensor containing indexes of target classes
<br/>
`THTensor *output` - **[OUT]** a one-element tensor with loss
<br/>
`bool sizeAverage` - if true, the loss will be normalized by batch size and class weights
<br/>
`THTensor *weights` - **[OPTIONAL]** class weights
<br/>
`THTensor *total_weight` - **[BUFFER]**
<br/>
```C
void THNN_ClassNLLCriterion_updateGradInput(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *gradInput,
          bool sizeAverage,
          THTensor *weights,
          THTensor *total_weight);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input tensor (1D/2D)
<br/>
`THIndexTensor *target` - tensor containing indexes of target classes
<br/>
`THTensor *gradInput` - **[OUT]** gradient w.r.t. input
<br/>
`bool sizeAverage` - if true, the loss will be normalized by batch size and class weights
<br/>
`THTensor *weights` - **[OPTIONAL]** class weights
<br/>
`THTensor *total_weight` - **[BUFFER]**
<br/>
## DistKLDivCriterion
```C
void THNN_DistKLDivCriterion_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *target` - target tensor
<br/>
`THTensor *output` - **[OUT]** a one-element tensor containing the loss
<br/>
`bool sizeAverage` - if true, the loss will be normalized **by total number of elements**
<br/>
```C
void THNN_DistKLDivCriterion_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *target` - target tensor
<br/>
`THTensor *gradInput` - **[OUT]** gradient w.r.t. input
<br/>
`bool sizeAverage` - if true, the loss will be normalized **by total number of elements**
<br/>
## ELU
```C
void THNN_ELU_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real alpha);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *output` - **[OUT]** ELU output
<br/>
`real alpha` - an ELU parameter (as in paper)
<br/>
```C
void THNN_ELU_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output,
          real alpha);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *gradOutput` - gradient w.r.t. output
<br/>
`THTensor *gradInput` - **[OUT]** gradient w.r.t. input
<br/>
`THTensor *output` - output from a forward pass
<br/>
`real alpha` - an ELU parameter (as in paper)
<br/>
## HardShrink
```C
void THNN_HardShrink_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real lambda);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *output` - **[OUT]** output tensor
<br/>
`real lambda` - HardShrink parameter
<br/>
```C
void THNN_HardShrink_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real lambda);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *gradOutput` - gradient w.r.t. module's output
<br/>
`THTensor *gradInput` - **[OUT]** gradient w.r.t. input
<br/>
`real lambda` - HardShrink parameter
<br/>
## HardTanh
```C
void THNN_HardTanh_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real min_val,
          real max_val);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *output` - **[OUT]** output tensor
<br/>
`real min_val` - lower threshold
<br/>
`real max_val` - upper threshold
<br/>
```C
void THNN_HardTanh_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real min_val,
          real max_val);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *gradOutput` - gradient w.r.t. module's output
<br/>
`THTensor *gradInput` - **[OUT]** gradient w.r.t. the input
<br/>
`real min_val` - lower threshold
<br/>
`real max_val` - upper threshold
<br/>
## L1Cost
```C
void THNN_L1Cost_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *output` - **[OUT]** output tensor
<br/>
```C
void THNN_L1Cost_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *gradOutput` - gradient w.r.t module's output
<br/>
`THTensor *gradInput` - **[OUT]** gradient w.r.t the input
<br/>
## LeakyReLU
```C
void THNN_LeakyReLU_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real negval,
          bool inplace);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - **[MODIFIED]** input tensor
<br/>
`THTensor *output` - **[OUT]** output tensor
<br/>
`real negval` - negative part slope
<br/>
`bool inplace` - if true, modifies the input tensor and sets the output tensor on it (no additional memory is allocated)
<br/>
```C
void THNN_LeakyReLU_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real negval,
          bool inplace);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *gradOutput` - **[MODIFIED]** gradient w.r.t. module's output
<br/>
`THTensor *gradInput` - **[OUT]** gradient w.r.t. the input
<br/>
`real negval` - negative part slope
<br/>
`bool inplace` - if true, modifies gradOutput and sets gradInput onto it (no additional memory is allocated)
<br/>
## LogSigmoid
```C
void THNN_LogSigmoid_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *buffer);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *output` - output tensor
<br/>
`THTensor *buffer` - **[BUFFER]**
<br/>
```C
void THNN_LogSigmoid_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *buffer);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input
<br/>
`THTensor *gradOutput` - gradient w.r.t. module's output
<br/>
`THTensor *gradInput` - **[OUT]** gradient w.r.t. input
<br/>
`THTensor *buffer` - **[BUFFER]**
<br/>
## LogSoftMax
```C
void THNN_LogSoftMax_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *output` - **[OUT]** output tensor
<br/>
```C
void THNN_LogSoftMax_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *gradOutput` - gradient w.r.t. module's output
<br/>
`THTensor *gradInput` - **[OUT]** gradient w.r.t. input
<br/>
`THTensor *output` - module's output
<br/>
## LookupTable
```C
void THNN_LookupTable_accGradParameters(
          THNNState *state,
          THIndexTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THIntegerTensor *count,
          THTensor *sorted,
          THTensor *indices,
          bool scaleGradByFreq,
          int paddingValue,
          real scale);
```
## MSECriterion
```C
void THNN_MSECriterion_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage);
```
```C
void THNN_MSECriterion_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage);
```
## MarginCriterion
```C
void THNN_MarginCriterion_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage,
          real margin);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *target` - target tensor (should contain only 1s and -1s)
<br/>
`THTensor *output` - **[OUT]** a one-element tensor containing the loss
<br/>
`bool sizeAverage` - if true, the loss is normalized by **total number of elements**
<br/>
`real margin` - a margin that is required for the loss to be 0
<br/>
```C
void THNN_MarginCriterion_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage,
          real margin);
```
`THNNState *state` - library's state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *target` - target tensor (should contin only 1s and -1s)
<br/>
`THTensor *gradInput` - **[OUT]** gradient w.r.t. module's input
<br/>
`bool sizeAverage` - if true, the gradient is normalized by **total number of elements**
<br/>
`real margin` - a margin that is required for the loss to be 0
<br/>
## MultiLabelMarginCriterion
```C
void THNN_MultiLabelMarginCriterion_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage);
```
```C
void THNN_MultiLabelMarginCriterion_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage);
```
## MultiMarginCriterion
```C
void THNN_MultiMarginCriterion_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage,
          int p,
          THTensor* weights);
```
```C
void THNN_MultiMarginCriterion_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage,
          int p,
          THTensor *weights);
```
## PReLU
```C
void THNN_PReLU_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THIndex_t nOutputPlane);
```
```C
void THNN_PReLU_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THIndex_t nOutputPlane);
```
```C
void THNN_PReLU_accGradParameters(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *gradWeight,
          THTensor *gradWeightBuf,
          THTensor *gradWeightBuf2,
          THIndex_t nOutputPlane,
          real scale);
```
## RReLU
```C
void THNN_RReLU_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *noise,
          real lower,
          real upper,
          bool train,
          bool inplace,
          THGenerator *generator);
```
```C
void THNN_RReLU_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *noise,
          real lower,
          real upper,
          bool train,
          bool inplace);
```
## Sigmoid
```C
void THNN_Sigmoid_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output);
```
```C
void THNN_Sigmoid_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output);
```
## SmoothL1Criterion
```C
void THNN_SmoothL1Criterion_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage);
```
```C
void THNN_SmoothL1Criterion_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage);
```
## SoftMax
```C
void THNN_SoftMax_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output);
```
```C
void THNN_SoftMax_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output);
```
## SoftPlus
```C
void THNN_SoftPlus_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real beta,
          real threshold);
```
```C
void THNN_SoftPlus_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output,
          real beta,
          real threshold);
```
## SoftShrink
```C
void THNN_SoftShrink_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real lambda);
```
```C
void THNN_SoftShrink_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real lambda);
```
## SparseLinear
```C
void THNN_SparseLinear_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *shardBuffer);
```
```C
void THNN_SparseLinear_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight);
```
```C
void THNN_SparseLinear_accGradParameters(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *weight,
          THTensor *bias,
          real weightDecay,
          real scale);
```
```C
void THNN_SparseLinear_zeroGradParameters(
          THNNState *state,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *lastInput);
```
```C
void THNN_SparseLinear_updateParameters(
          THNNState *state,
          THTensor *weight,
          THTensor *bias,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *lastInput,
          real learningRate);
```
## SpatialAdaptiveMaxPooling
```C
void THNN_SpatialAdaptiveMaxPooling_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *indices,
          int owidth, int oheight);
```
```C
void THNN_SpatialAdaptiveMaxPooling_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *indices);
```
## SpatialAveragePooling
```C
void THNN_SpatialAveragePooling_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          bool ceil_mode,
          bool count_include_pad);
```
```C
void THNN_SpatialAveragePooling_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          bool ceil_mode,
          bool count_include_pad);
```
## SpatialBatchNormalization
```C
void THNN_SpatialBatchNormalization_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *running_mean,
          THTensor *running_var,
          THTensor *save_mean,
          THTensor *save_std,
          bool train,
          double momentum,
          double eps);
```
```C
void THNN_SpatialBatchNormalization_backward(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *weight,
          THTensor *save_mean,
          THTensor *save_std,
          double scale);
```
## SpatialConvolutionLocal
```C
void THNN_SpatialConvolutionLocal_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          THTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          long inputWidth, long inputHeight,
          long outputWidth, long outputHeight);
```
```C
void THNN_SpatialConvolutionLocal_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          THTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          long inputWidth, long inputHeight,
          long outputWidth, long outputHeight);
```
```C
void THNN_SpatialConvolutionLocal_accGradParameters(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          THTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          long inputWidth, long inputHeight,
          long outputWidth, long outputHeight,
          real scale);
```
## SpatialConvolutionMM
```C
void THNN_SpatialConvolutionMM_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          THTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH);
```
```C
void THNN_SpatialConvolutionMM_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          THTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH);
```
```C
void THNN_SpatialConvolutionMM_accGradParameters(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          THTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          real scale);
```
## SpatialConvolutionMap
```C
void THNN_SpatialConvolutionMap_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *connTable,
          int nInputPlane,
          int nOutputPlane,
          int dW, int dH);
```
`THNNState *state` - library state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *output` - **[OUT]** convolution output
<br/>
`THTensor *weight` - 3D weight tensor (connTable:size(1) x kH x kW)
<br/>
`THTensor *bias` - 1D bias tensor (nOutputPlane)
<br/>
`THTensor *connTable` - connection table
<br/>
`int nInputPlane` - number of input planes
<br/>
`int nOutputPlane` - number of output planes
<br/>
`int dW, int dH` - stride
<br/>
```C
void THNN_SpatialConvolutionMap_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *bias,
          THTensor *connTable,
          int nInputPlane,
          int nOutputPlane,
          int dW, int dH);
```
`THNNState *state` - library state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *gradOutput` - gradient w.r.t. output
<br/>
`THTensor *gradInput` - **[OUT]** gradient w.r.t. input
<br/>
`THTensor *weight` - 3D weight tensor (connTable:size(1) x kH x kW)
<br/>
`THTensor *bias` - 1D bias tensor (nOutputPlane)
<br/>
`THTensor *connTable` - connection table
<br/>
`int nInputPlane` - number of input planes
<br/>
`int nOutputPlane` - number of output planes
<br/>
`int dW, int dH` - stride
<br/>
```C
void THNN_SpatialConvolutionMap_accGradParameters(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *connTable,
          int nInputPlane,
          int nOutputPlane,
          int dW, int dH,
          real scale);
```
`THNNState *state` - library state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *gradOutput` - gradient w.r.t. output
<br/>
`THTensor *gradWeight` - 3D gradWeight tensor (connTable:size(1) x kH x kW)
<br/>
`THTensor *gradBias` - 1D gradBias tensor (nOutputPlane)
<br/>
`THTensor *connTable` - connection table
<br/>
`int nInputPlane` - number of input planes
<br/>
`int nOutputPlane` - number of output planes
<br/>
`int dW, int dH` - stride
<br/>
`real scale` - scaling factor
<br/>
## SpatialFractionalMaxPooling
```C
void THNN_SpatialFractionalMaxPooling_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int outputW, int outputH,
          int poolSizeW, int poolSizeH,
          THTensor *indices,
          THTensor *randomSamples);
```
```C
void THNN_SpatialFractionalMaxPooling_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int outputW, int outputH,
          int poolSizeW, int poolSizeH,
          THTensor *indices);
```
## SpatialFullConvolution
```C
void THNN_SpatialFullConvolution_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *columns,
          THTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int adjW, int adjH);
```
```C
void THNN_SpatialFullConvolution_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *gradColumns,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int adjW, int adjH);
```
```C
void THNN_SpatialFullConvolution_accGradParameters(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *columns,
          THTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int adjW, int adjH,
          real scale);
```
## SpatialFullConvolutionMap
```C
void THNN_SpatialFullConvolutionMap_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *connTable,
          int nInputPlane,
          int nOutputPlane,
          int dW, int dH);
```
`THNNState *state` - library state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *output` - **[OUT]** convolution output
<br/>
`THTensor *weight` - 3D weight tensor (connTable:size(1) x kH x kW)
<br/>
`THTensor *bias` - 1D bias tensor (nOutputPlane)
<br/>
`THTensor *connTable` - connection table
<br/>
`int nInputPlane` - number of input planes
<br/>
`int nOutputPlane` - number of output planes
<br/>
`int dW, int dH` - stride
<br/>
```C
void THNN_SpatialFullConvolutionMap_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *bias,
          THTensor *connTable,
          int nInputPlane,
          int nOutputPlane,
          int dW, int dH);
```
`THNNState *state` - library state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *gradOutput` - gradient w.r.t. output
<br/>
`THTensor *gradInput` - **[OUT]** gradient w.r.t. input
<br/>
`THTensor *weight` - 3D weight tensor (connTable:size(1) x kH x kW)
<br/>
`THTensor *bias` - 1D bias tensor (nOutputPlane)
<br/>
`THTensor *connTable` - connection table
<br/>
`int nInputPlane` - number of input planes
<br/>
`int nOutputPlane` - number of output planes
<br/>
`int dW, int dH` - stride
<br/>
```C
void THNN_SpatialFullConvolutionMap_accGradParameters(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *connTable,
          int nInputPlane,
          int nOutputPlane,
          int dW, int dH,
          real scale);
```
`THNNState *state` - library state
<br/>
`THTensor *input` - input tensor
<br/>
`THTensor *gradOutput` - gradient w.r.t. output
<br/>
`THTensor *gradWeight` - 3D gradWeight tensor (connTable:size(1) x kH x kW)
<br/>
`THTensor *gradBias` - 1D gradBias tensor (nOutputPlane)
<br/>
`THTensor *connTable` - connection table
<br/>
`int nInputPlane` - number of input planes
<br/>
`int nOutputPlane` - number of output planes
<br/>
`int dW, int dH` - stride
<br/>
`real scale` - scaling factor
<br/>
## SpatialMaxPooling
```C
void THNN_SpatialMaxPooling_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *indices,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          bool ceil_mode);
```
```C
void THNN_SpatialMaxPooling_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *indices,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          bool ceil_mode);
```
## SpatialMaxUnpooling
```C
void THNN_SpatialMaxUnpooling_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *indices,
          int owidth, int oheight);
```
```C
void THNN_SpatialMaxUnpooling_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *indices,
          int owidth, int oheight);
```
## SpatialSubSampling
```C
void THNN_SpatialSubSampling_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          int kW, int kH,
          int dW, int dH);
```
```C
void THNN_SpatialSubSampling_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          int kW, int kH,
          int dW, int dH);
```
```C
void THNN_SpatialSubSampling_accGradParameters(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          int kW, int kH,
          int dW, int dH,
          real scale);
```
## SpatialUpSamplingNearest
```C
void THNN_SpatialUpSamplingNearest_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int scale_factor);
```
```C
void THNN_SpatialUpSamplingNearest_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int scale_factor);
```
## Sqrt
```C
void THNN_Sqrt_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real eps);
```
```C
void THNN_Sqrt_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output);
```
## Square
```C
void THNN_Square_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output);
```
```C
void THNN_Square_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput);
```
## Tanh
```C
void THNN_Tanh_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output);
```
```C
void THNN_Tanh_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output);
```
## Threshold
```C
void THNN_Threshold_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real threshold,
          real val,
          bool inplace);
```
```C
void THNN_Threshold_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          real threshold,
          bool inplace);
```
## VolumetricAveragePooling
```C
void THNN_VolumetricAveragePooling_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int kT, int kW, int kH,
          int dT, int dW, int dH);
```
```C
void THNN_VolumetricAveragePooling_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int kT, int kW, int kH,
          int dT, int dW, int dH);
```
## VolumetricConvolution
```C
void THNN_VolumetricConvolution_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          THTensor *fgradInput,
          int dT, int dW, int dH,
          int pT, int pW, int pH);
```
```C
void THNN_VolumetricConvolution_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          int dT, int dW, int dH,
          int pT, int pW, int pH);
```
```C
void THNN_VolumetricConvolution_accGradParameters(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          THTensor *fgradInput,
          int dT, int dW, int dH,
          int pT, int pW, int pH,
          real scale);
```
## VolumetricConvolutionMM
```C
void THNN_VolumetricConvolutionMM_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int pT, int pW, int pH);
```
```C
void THNN_VolumetricConvolutionMM_updateGradInput(
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
```
```C
void THNN_VolumetricConvolutionMM_accGradParameters(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          real scale);
```
## VolumetricFullConvolution
```C
void THNN_VolumetricFullConvolution_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          THTensor *fgradInput,
          int dT, int dW, int dH,
          int pT, int pW, int pH);
```
```C
void THNN_VolumetricFullConvolution_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          THTensor *fgradInput,
          int dT, int dW, int dH,
          int pT, int pW, int pH);
```
```C
void THNN_VolumetricFullConvolution_accGradParameters(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          THTensor *fgradInput,
          int dT, int dW, int dH,
          int pT, int pW, int pH,
          real scale);
```
## VolumetricMaxPooling
```C
void THNN_VolumetricMaxPooling_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *indices,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int pT, int pW, int pH,
          bool ceilMode);
```
```C
void THNN_VolumetricMaxPooling_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *indices,
          int dT, int dW, int dH,
          int pT, int pW, int pH);
```
## VolumetricMaxUnpooling
```C
void THNN_VolumetricMaxUnpooling_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *indices,
          int oT, int oW, int oH,
          int dT, int dW, int dH,
          int pT, int pW, int pH);
```
```C
void THNN_VolumetricMaxUnpooling_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *indices,
          int oT, int oW, int oH,
          int dT, int dW, int dH,
          int pT, int pW, int pH);
```
