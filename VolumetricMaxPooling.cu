#include "THCUNN.h"
#include "common.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"

#include <cfloat>

void THNN_CudaVolumetricMaxPooling_updateOutput(
  THCState *state, THCudaTensor *input, THCudaTensor *output, THCIndexTensor *indices,
  int kT, int kW, int kH,
  int dT, int dW, int dH,
  int padT, int padW, int padH,
  bool ceilMode)
{
  THNN_CudaVolumetricDilatedMaxPooling_updateOutput(
    state, input, output, indices,
    kT, kW, kH, dT, dW, dH, padT, padW, padH, 1, 1, 1, ceilMode);

}

void THNN_CudaVolumetricMaxPooling_updateGradInput(
  THCState *state, THCudaTensor *input, THCudaTensor *gradOutput, THCudaTensor *gradInput,
  THCIndexTensor *indices,
  int dT, int dW, int dH,
  int padT, int padW, int padH)
{
  THNN_CudaVolumetricDilatedMaxPooling_updateGradInput(
    state, input, gradOutput, gradInput, indices,
    dT, dW, dH, padT, padW, padH, 1, 1, 1);

}
