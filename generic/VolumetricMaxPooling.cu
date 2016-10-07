/*#include "THCUNN.h"
#include "common.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"

#include <cfloat>*/
#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/VolumetricMaxPooling.cu"
#else

#include "../common.h"

void THNN_(VolumetricMaxPooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *indices,
           int kT, int kW, int kH,
           int dT, int dW, int dH,
           int padT, int padW, int padH,
           bool ceilMode)
{
  THNN_(VolumetricDilatedMaxPooling_updateOutput)(
    state, input, output, indices,
    kT, kW, kH, dT, dW, dH, padT, padW, padH, 1, 1, 1, ceilMode);

}

void THNN_(VolumetricMaxPooling_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *indices,
           int dT, int dW, int dH,
           int padT, int padW, int padH)
{
  THNN_(VolumetricDilatedMaxPooling_updateGradInput)(
    state, input, gradOutput, gradInput, indices,
    dT, dW, dH, padT, padW, padH, 1, 1, 1);

}

#endif
