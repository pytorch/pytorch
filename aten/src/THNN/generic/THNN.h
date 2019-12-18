#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/THNN.h"
#else

#include <ATen/core/Reduction.h>
#include <ATen/core/Generator.h>
#include <ATen/core/DistributionsHelper.h>

#if !defined(TH_REAL_IS_LONG)

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

TH_API void THNN_(LogSigmoid_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *output,            // output tensor
          THTensor *buffer);           // [BUFFER]
TH_API void THNN_(LogSigmoid_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input
          THTensor *gradOutput,        // gradient w.r.t. module's output
          THTensor *gradInput,         // [OUT] gradient w.r.t. input
          THTensor *buffer);           // [BUFFER]

TH_API void THNN_(RReLU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *noise,
          accreal lower,
          accreal upper,
          bool train,
          bool inplace,
          at::Generator *generator);
TH_API void THNN_(RReLU_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *noise,
          accreal lower,
          accreal upper,
          bool train,
          bool inplace);

TH_API void THNN_(SoftPlus_updateOutput)(
          THNNState *state,
          THTensor *input, THTensor *output,
          accreal beta,
          accreal threshold);
TH_API void THNN_(SoftPlus_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output,
          accreal beta,
          accreal threshold);

#endif
#endif
