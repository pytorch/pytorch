#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/THNN.h"
#else

#include <ATen/core/Reduction.h>
#include <ATen/core/Generator.h>
#include <ATen/core/DistributionsHelper.h>

#if !defined(TH_REAL_IS_LONG)

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

#endif
#endif
