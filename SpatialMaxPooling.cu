#include "THCUNN.h"
#include "common.h"

void THNN_CudaSpatialMaxPooling_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output, THCudaTensor *indices, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode)
{
  THNN_CudaSpatialDilatedMaxPooling_updateOutput(
    state, input, output, indices, 
    kW, kH, dW, dH, padW, padH, 1, 1, ceil_mode);

}

void THNN_CudaSpatialMaxPooling_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput, THCudaTensor *gradInput, THCudaTensor *indices, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode)
{
  THNN_CudaSpatialDilatedMaxPooling_updateGradInput(
    state, input, gradOutput, gradInput, indices,
    kW, kH, dW, dH, padW, padH, 1, 1, ceil_mode);

}
