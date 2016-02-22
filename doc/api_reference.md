# API docs

This document only describes a THNN API. For a thorough review of all modules present here please refer to [nn's docs](http://github.com/torch/nn/tree/master/doc).

### Note on function names

Please remember, that because C doesn't support function overloading, functions taking different tensor types have different names. So e.g. for an Abs module, there are actually two updateOutput functions:

* `void THNN_FloatAbs_updateOutput(...)`
* `void THNN_DoubleAbs_updateOutput(...)`

In these docs such function will be referred to as `void THNN_Abs_updateOutput(...)`, and it's up to developer to add a type prefix. `real` is an alias for that type.

## Module list

These are all modules implemented in THNN:

* Nonlinear functions
  * [Abs](#abs)
  * [ELU](#elu)
  * HardShrink
  * HardTanh
  * LeakyReLU
  * LogSigmoid
  * LogSoftMax
  * PReLU
  * RReLU
  * Sigmoid
  * SoftMax
  * SoftPlus
  * SoftShrink
  * Sqrt
  * Square
  * Tanh
  * Threshold
* Criterions
  * AbsCriterion
  * ClassNLLCriterion
  * DistKLDivCriterion
  * L1Cost
  * MSECriterion
  * MarginCriterion
  * MultiLabelMarginCriterion
  * MultiMarginCriterion
  * SmoothL1Criterion
* Modules
  * LookupTable
  * SparseLinear
* Spatial modules
  * SpatialAdaptiveMaxPooling
  * SpatialAdaptiveMaxPooling
  * SpatialAveragePooling
  * SpatialConvolutionMM
* Volumetric modules
  * VolumetricAveragePooling
  * VolumetricConvoluion
  * VolumetricConvoluionMM
  * VolumetricFullConvolution
  * VolumetricMaxPooling
  * VolumetricMaxUnpooling

## Abs

```C
void THNN_Abs_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output);
```

`state` - library's state
<br/>
`input` - input tensor
<br/>
`output` - **[OUT]** Abs output

```C
void THNN_Abs_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput);
```

`state` - library's state
<br/>
`input` - input tensor
<br/>
`gradOutput` - gradient w.r.t. output
<br/>
`gradInput` - **[OUT]** gradient w.r.t. input

## ELU

For reference see [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](http://arxiv.org/abs/1511.07289).

```C
void THNN_ELU_updateOutput(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real alpha);
```

`state` - library state
<br/>
`input` - input tensor
<br/>
`output` - **[OUT]** ELU output
<br/>
`alpha` - an ELU parameter

```C
void THNN_ELU_updateGradInput(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output,
          real alpha);
```

`state` - library state
<br/>
`input` - input tensor
<br/>
`gradOutput` - gradient w.r.t. output
<br/>
`gradInput` - **[OUT]** gradient w.r.t. input
<br/>
`output` - module output for given input
<br/>
`alpha` - an ELU parameter
