#include <THC/THC.h>
#include "THCApply.cuh"

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
          float *output,
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
