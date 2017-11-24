#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/VolumetricAdaptiveAveragePooling.cu"
#else

#include "../common.h"

// 5d tensor B x D x T x H x W

void THNN_(VolumetricAdaptiveAveragePooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int osizeT,
           int osizeW,
           int osizeH)
{
  THCUNN_assertSameGPU(state, 2, input, output);

  THCUNN_argCheck(state, input->nDimension == 4 || input->nDimension == 5, 2, input,
                  "4D or 5D (batch mode) tensor expected for input, but got: %s");


  real *output_data;
  real *input_data;

  int64_t sizeD, isizeT, isizeH, isizeW;
  int64_t istrideD, istrideT, istrideH, istrideW;
  int64_t totalZ;

  if (input->nDimension == 4) {
    sizeD = input->size[0];
    isizeT = input->size[1];
    isizeH = input->size[2];
    isizeW = input->size[3];

    istrideD = input->stride[0];
    istrideT = input->stride[1];
    istrideH = input->stride[2];
    istrideW = input->stride[3];

    THCTensor_(resize4d)(state, output, sizeD, osizeT, osizeH, osizeW);

    totalZ = sizeD * osizeT;
  } else {
    input = THCTensor_(newContiguous)(state, input);

    int64_t sizeB = input->size[0];
    sizeD = input->size[1];
    isizeT = input->size[2];
    isizeH = input->size[3];
    isizeW = input->size[4];

    istrideD = input->stride[1];
    istrideT = input->stride[2];
    istrideH = input->stride[3];
    istrideW = input->stride[4];

    THCTensor_(resize5d)(state, output, sizeB, sizeD, osizeT, osizeH, osizeW);

    totalZ = sizeB * sizeD * osizeT;
  }

  input_data = THCTensor_(data)(state, input);
  output_data = THCTensor_(data)(state, output);

  int64_t offsetZ = 0;
  dim3 threads(32, 8);
  // each H*W plane is processed by blocksH thread blocks
  int blocksH = max((int)(16L / totalZ), 1);
  while (totalZ > 0) {
    dim3 blocks(totalZ > 65535 ? 65535 : totalZ, blocksH);
    cunn_VolumetricAdaptiveAveragePooling_updateOutput_kernel
      <<<blocks, threads, 0, THCState_getCurrentStream(state)>>>(
        input_data, output_data, isizeT, isizeH, isizeW, osizeT, osizeH, osizeW,
        istrideD, istrideT, istrideH, istrideW, offsetZ
      );

    totalZ -= 65535;
    offsetZ += 65535;
    THCudaCheck(cudaGetLastError());
  }

  if (input->nDimension == 5) {
    // clean
    THCTensor_(free)(state, input);
  }
}

void THNN_(VolumetricAdaptiveAveragePooling_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput)
{
  THCUNN_assertSameGPU(state, 3, input, gradOutput, gradInput);

  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  THCTensor_(resizeAs)(state, gradInput, input);
  THCTensor_(zero)(state, gradInput);

  real *gradInput_data;
  real *gradOutput_data;

  int64_t sizeD, isizeT, isizeH, isizeW;
  int64_t osizeT, osizeH, osizeW;
  int64_t totalZ;

  if (input->nDimension == 4) {
    sizeD = input->size[0];
    isizeT = input->size[1];
    isizeH = input->size[2];
    isizeW = input->size[3];

    osizeT = gradOutput->size[1];
    osizeH = gradOutput->size[2];
    osizeW = gradOutput->size[3];
  } else {
    sizeD = input->size[1];
    isizeT = input->size[2];
    isizeH = input->size[3];
    isizeW = input->size[4];

    osizeT = gradOutput->size[2];
    osizeH = gradOutput->size[3];
    osizeW = gradOutput->size[4];
  }

  // somehow nonatomic is passing all test for volumetric case.
  bool atomic = false; //(isizeW%osizeW != 0) || (isizeH%osizeH != 0) || (isizeT%osizeT != 0);

  if (input->nDimension == 4) {
    totalZ = atomic ? sizeD * osizeT : sizeD * isizeT;
  } else {
    int sizeB = input->size[0];
    totalZ = atomic ? sizeB * sizeD * osizeT : sizeB * sizeD * isizeT;
  }

  gradInput_data = THCTensor_(data)(state, gradInput);
  gradOutput_data = THCTensor_(data)(state, gradOutput);

  int64_t offsetZ = 0;
  dim3 threads(32, 8);
  // each H*W plane is processed by blocksH thread blocks
  int blocksH = max((int)(16L / totalZ), 1);
  while (totalZ > 0) {
    dim3 blocks(totalZ > 65535 ? 65535 : totalZ, blocksH);

    if (atomic)
    {
      cunn_atomic_VolumetricAdaptiveAveragePooling_updateGradInput_kernel
        <<<blocks, threads, 0, THCState_getCurrentStream(state)>>>(
          gradInput_data, gradOutput_data, isizeT, isizeH, isizeW,
          osizeT, osizeH, osizeW, offsetZ
        );
    } else {
        cunn_VolumetricAdaptiveAveragePooling_updateGradInput_kernel
          <<<blocks, threads, 0, THCState_getCurrentStream(state)>>>(
            gradInput_data, gradOutput_data, isizeT, isizeH, isizeW,
            osizeT, osizeH, osizeW, offsetZ
          );
    }

    totalZ -= 65535;
    offsetZ += 65535;
    THCudaCheck(cudaGetLastError());
  }
  // clean
  THCTensor_(free)(state, gradOutput);

}

#endif
