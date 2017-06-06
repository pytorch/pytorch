#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/VolumetricSubSampling.cu"
#else

#include "../common.h"

static inline void THNN_(VolumetricSubSampling_shapeCheck)(
                         THCState *state,
                         THCTensor *input,
                         THCTensor *gradOutput,
                         THCTensor *weight,
                         int kW, int kH, int kT) {
  THCUNN_argCheck(state, input->nDimension == 4 || input->nDimension == 5, 2, input,
                  "4D or 5D input tensor expected but got: %s");

  int nInputPlane = THCTensor_(size)(state, weight, 0);

  int dimc = 3;
  int dimr = 2;
  int dimt = 1;
  int dimp = 0;

  if (input->nDimension == 5) {
    dimc++;
    dimr++;
    dimt++;
    dimp++;
  }

  long nInputCols = input->size[dimc];
  long nInputRows = input->size[dimr];
  long nInputFrames = input->size[dimt];
  THArgCheck(input->size[dimp] == nInputPlane, 2, "invalid number of input planes");
  THArgCheck(nInputCols >= kW && nInputRows >= kH && nInputFrames >= kT, 2, "input image smaller than kernel size");
}

void THNN_(VolumetricSubSampling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *weight,
           THCTensor *bias,
           int kW, int kH, int kT,
           int dW, int dH, int dT)
{
  real *weight_data = THCTensor_(data)(state, weight);
  real *bias_data = THCTensor_(data)(state, bias);
  real *output_data;
  real *input_data;

  int nInputPlane = THCTensor_(size)(state, weight, 0);

  THCUNN_assertSameGPU(state, 4, input, output, weight, bias);
  THNN_(VolumetricSubSampling_shapeCheck)(state, input, NULL, weight, kW, kH, kT);

  if (input->nDimension == 4) {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nInputFrames = input->size[1];
    long nOutputCols = (nInputCols - kW) / dW + 1;
    long nOutputRows = (nInputRows - kH) / dH + 1;
    long nOutputFrames = (nInputFrames - kT) / dT + 1;

    input = THCTensor_(newContiguous)(state, input);
    input_data = THCTensor_(data)(state, input);

    THCTensor_(resize4d)(state, output, nInputPlane, nOutputFrames, nOutputRows, nOutputCols);
    output_data = THCTensor_(data)(state, output);

    // cuda blocks & threads:
    int zblocks = (int)(16L / nInputPlane);
    zblocks = zblocks < 1 ? 1 : zblocks;
    dim3 blocks(nInputPlane,zblocks);
    dim3 threads(32,8);

    // run subsample kernel
    vsubsample<real, accreal> <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
      input_data, output_data, weight_data, bias_data,
      nInputPlane, nInputFrames, nInputRows, nInputCols, kT, kH, kW, dT, dH, dW);
    THCudaCheck(cudaGetLastError());
  } else {
    long nInputCols = input->size[4];
    long nInputRows = input->size[3];
    long nInputFrames = input->size[2];
    long nbatch = input->size[0];
    long nOutputCols = (nInputCols - kW) / dW + 1;
    long nOutputRows = (nInputRows - kH) / dH + 1;
    long nOutputFrames = (nInputFrames - kT) / dT + 1;

    input = THCTensor_(newContiguous)(state, input);
    input_data = THCTensor_(data)(state, input);

    THCTensor_(resize5d)(state, output, nbatch, nInputPlane, nOutputFrames, nOutputRows, nOutputCols);
    output_data = THCTensor_(data)(state, output);

    // cuda blocks & threads:
    int zblocks = (int)(16L / nInputPlane);
    zblocks = zblocks < 1 ? 1 : zblocks;
    dim3 blocks(nInputPlane*nbatch,zblocks);
    dim3 threads(32,8);

    // run subsample kernel
    vsubsample<real, accreal> <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
      input_data, output_data, weight_data, bias_data,
      nInputPlane, nInputFrames, nInputRows, nInputCols, kT, kH, kW, dT, dH, dW);
    THCudaCheck(cudaGetLastError());
  }

  // clean
  THCTensor_(free)(state, input);

}

void THNN_(VolumetricSubSampling_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *weight,
           int kW, int kH, int kT,
           int dW, int dH, int dT)
{
  THCUNN_assertSameGPU(state, 4, input, gradOutput, weight, gradInput);
  THNN_(VolumetricSubSampling_shapeCheck)(state, input, gradOutput, weight, kW, kH, kT);

  int nInputPlane = THCTensor_(size)(state, weight, 0);

  if (input->nDimension == 4) {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nInputFrames = input->size[1];

    real *weight_data = THCTensor_(data)(state, weight);
    gradOutput = THCTensor_(newContiguous)(state, gradOutput);
    real *gradOutput_data = THCTensor_(data)(state, gradOutput);
    real *gradInput_data;

    THCTensor_(resizeAs)(state, gradInput, input);
    THCTensor_(zero)(state, gradInput);
    gradInput_data = THCTensor_(data)(state, gradInput);

    // cuda blocks & threads:
    int zblocks = (int)(16L / nInputPlane);
    zblocks = zblocks < 1 ? 1 : zblocks;
    dim3 blocks(nInputPlane,zblocks);
    dim3 threads(32,8);

    // run updateGradInput kernel
    if (kT <= dT && kH <= dH && kW <= dW) {
      vsubgradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
        gradInput_data, gradOutput_data, weight_data,
        nInputPlane, nInputFrames, nInputRows, nInputCols, kT, kH, kW, dT, dH, dW);
    } else {
      vsubgradinputAtomic <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
        gradInput_data, gradOutput_data, weight_data,
        nInputPlane, nInputFrames, nInputRows, nInputCols, kT, kH, kW, dT, dH, dW);
    }
    THCudaCheck(cudaGetLastError());
  } else {
    long nInputCols = input->size[4];
    long nInputRows = input->size[3];
    long nInputFrames = input->size[2];
    long nbatch = input->size[0];

    real *weight_data = THCTensor_(data)(state, weight);
    gradOutput = THCTensor_(newContiguous)(state, gradOutput);
    real *gradOutput_data = THCTensor_(data)(state, gradOutput);
    real *gradInput_data;

    THCTensor_(resizeAs)(state, gradInput, input);
    THCTensor_(zero)(state, gradInput);
    gradInput_data = THCTensor_(data)(state, gradInput);

    // cuda blocks & threads:
    int zblocks = (int)(16L / nInputPlane);
    zblocks = zblocks < 1 ? 1 : zblocks;
    dim3 blocks(nInputPlane*nbatch,zblocks);
    dim3 threads(32,8);

    // run updateGradInput kernel
    if (kT <= dT && kH <= dH && kW <= dW) {
      vsubgradinput <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
        gradInput_data, gradOutput_data, weight_data,
        nInputPlane, nInputFrames, nInputRows, nInputCols, kT, kH, kW, dT, dH, dW);
    } else {
      vsubgradinputAtomic <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
        gradInput_data, gradOutput_data, weight_data,
        nInputPlane, nInputFrames, nInputRows, nInputCols, kT, kH, kW, dT, dH, dW);
    }
    THCudaCheck(cudaGetLastError());
  }
  THCTensor_(free)(state, gradOutput);
}

void THNN_(VolumetricSubSampling_accGradParameters)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradWeight,
           THCTensor *gradBias,
           int kW, int kH, int kT,
           int dW, int dH, int dT,
           accreal scale)
{
  THCUNN_assertSameGPU(state, 4, input, gradOutput, gradWeight, gradBias);
  THNN_(VolumetricSubSampling_shapeCheck)(state, input, gradOutput, gradWeight, kW, kH, kT);

  int nInputPlane = THCTensor_(size)(state, gradWeight, 0);

  if (input->nDimension == 4) {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nInputFrames = input->size[1];

    real *gradWeight_data = THCTensor_(data)(state, gradWeight);
    real *gradBias_data = THCTensor_(data)(state, gradBias);
    gradOutput = THCTensor_(newContiguous)(state, gradOutput);
    real *gradOutput_data = THCTensor_(data)(state, gradOutput);
    real *input_data;

    input = THCTensor_(newContiguous)(state, input);
    input_data = THCTensor_(data)(state, input);

    // cuda blocks & threads:
    dim3 blocks(nInputPlane);
    dim3 threads(32,8);

    // run gradweight kernel
    vsubgradweight<real, accreal> <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
      input_data, gradOutput_data, gradWeight_data, gradBias_data,
      nInputPlane, nInputFrames, nInputRows, nInputCols, kT, kH, kW, dT, dH, dW, scale);
    THCudaCheck(cudaGetLastError());
  } else {
    long nInputCols = input->size[4];
    long nInputRows = input->size[3];
    long nInputFrames = input->size[2];
    long nbatch = input->size[0];

    real *gradWeight_data = THCTensor_(data)(state, gradWeight);
    real *gradBias_data = THCTensor_(data)(state, gradBias);
    gradOutput = THCTensor_(newContiguous)(state, gradOutput);
    real *gradOutput_data = THCTensor_(data)(state, gradOutput);
    real *input_data;

    input = THCTensor_(newContiguous)(state, input);
    input_data = THCTensor_(data)(state, input);

    // cuda blocks & threads:
    dim3 blocks(nInputPlane);
    dim3 threads(32,8);

    // run gradweight kernel
    long sl;
    for (sl=0; sl<nbatch; sl++) {
      vsubgradweight<real, accreal> <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (
        input_data + sl*input->stride[0],
        gradOutput_data + sl*gradOutput->stride[0],
        gradWeight_data, gradBias_data,
        nInputPlane, nInputFrames, nInputRows, nInputCols, kT, kH, kW, dT, dH, dW, scale);
    }
    THCudaCheck(cudaGetLastError());
  }

  // clean
  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);

}

#endif
