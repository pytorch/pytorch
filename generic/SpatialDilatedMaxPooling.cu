#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialDilatedMaxPooling.cu"
#else

#include "../common.h"

void THNN_(SpatialDilatedMaxPooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *indices,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           int dilationW, int dilationH,
           bool ceil_mode)
{

  THCUNN_assertSameGPU_generic(state, 3, input, output, indices);
  THArgCheck(input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

  long nInputCols, nInputRows, nInputPlane, batchSize;
  long nOutputCols, nOutputRows;

  if (input->nDimension == 3) {
    nInputCols = input->size[2];
    nInputRows = input->size[1];
    nInputPlane = input->size[0];
    batchSize = 1;
  }
  else
  {
    nInputCols = input->size[3];
    nInputRows = input->size[2];
    nInputPlane = input->size[1];
    batchSize = input->size[0];
  }

  THArgCheck(nInputCols >= kW - padW && nInputRows >= kH - padH, 2, "input image smaller than kernel size");
  THArgCheck(kW/2 >= padW && kH/2 >= padH, 2, "pad should be smaller than half of kernel size");

  if(ceil_mode) {
    nOutputCols = ScalarConvert<real,long>::to(
      THCNumerics<real>::ceil(ScalarConvert<long,real>::to(nInputCols - (dilationW * (kW - 1) + 1) + 2*padW) / ScalarConvert<long,real>::to(dW))) + 1;
    nOutputRows = ScalarConvert<real,long>::to(
      THCNumerics<real>::ceil(ScalarConvert<long,real>::to(nInputRows - (dilationH * (kH - 1) + 1) + 2*padH) / ScalarConvert<long,real>::to(dH))) + 1;
  }
  else {
    nOutputCols = ScalarConvert<real,long>::to(
      THCNumerics<real>::floor(ScalarConvert<long,real>::to(nInputCols - (dilationW * (kW - 1) + 1) + 2*padW) / ScalarConvert<long,real>::to(dW))) + 1;
    nOutputRows = ScalarConvert<real,long>::to(
      THCNumerics<real>::floor(ScalarConvert<long,real>::to(nInputRows - (dilationH * (kH - 1) + 1) + 2*padH) / ScalarConvert<long,real>::to(dH))) + 1;
  }

if (nOutputCols < 1 || nOutputRows < 1)
    THError("Given input size: (%dx%dx%d). Calculated output size: (%dx%dx%d). Output size is too small",
            nInputPlane,nInputRows,nInputCols,nInputPlane,nOutputRows,nOutputCols);

if (padW || padH)
  {
    // ensure that the last pooling starts inside the image
    if ((nOutputRows - 1)*dH >= nInputRows + padH)
      --nOutputRows;
    if ((nOutputCols  - 1)*dW >= nInputCols  + padW)
      --nOutputCols;
  }

  input = THCTensor_(newContiguous)(state, input);
  real* input_data = THCTensor_(data)(state, input);

  THCTensor_(resize4d)(state, output, batchSize, nInputPlane, nOutputRows, nOutputCols);
  THCTensor_(resizeAs)(state, indices, output);

  real* indices_data = THCTensor_(data)(state, indices);
  real* output_data = THCTensor_(data)(state, output);

  int count = THCTensor_(nElement)(state, output);

  MaxPoolForward <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
      (count, input_data,
      batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
      kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
  THCudaCheck(cudaGetLastError());

  if(input->nDimension == 3)
    THCTensor_(resize3d)(state, output, nInputPlane, nOutputRows, nOutputCols);

  THCTensor_(free)(state, input);
}

void THNN_(SpatialDilatedMaxPooling_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *indices,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           int dilationW, int dilationH,
           bool ceil_mode)
{
  THCUNN_assertSameGPU_generic(state, 4, input, gradOutput, indices, gradInput);

  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  long nInputCols, nInputRows, nInputPlane, batchSize;
  long nOutputCols, nOutputRows;

  if (input->nDimension == 3) {
    nInputCols = input->size[2];
    nInputRows = input->size[1];
    nInputPlane = input->size[0];
    batchSize = 1;
  }
  else
  {
    nInputCols = input->size[3];
    nInputRows = input->size[2];
    nInputPlane = input->size[1];
    batchSize = input->size[0];
  }

  if(ceil_mode) {
    nOutputCols = ScalarConvert<real,long>::to(
      THCNumerics<real>::ceil(ScalarConvert<long,real>::to(nInputCols - (dilationW * (kW - 1) + 1) + 2*padW) / ScalarConvert<long,real>::to(dW))) + 1;
    nOutputRows = ScalarConvert<real,long>::to(
      THCNumerics<real>::ceil(ScalarConvert<long,real>::to(nInputRows - (dilationH * (kH - 1) + 1) + 2*padH) / ScalarConvert<long,real>::to(dH))) + 1;
  }
  else {
    nOutputCols = ScalarConvert<real,long>::to(
      THCNumerics<real>::floor(ScalarConvert<long,real>::to(nInputCols - (dilationW * (kW - 1) + 1) + 2*padW) / ScalarConvert<long,real>::to(dW))) + 1;
    nOutputRows = ScalarConvert<real,long>::to(
      THCNumerics<real>::floor(ScalarConvert<long,real>::to(nInputRows - (dilationH * (kH - 1) + 1) + 2*padH) / ScalarConvert<long,real>::to(dH))) + 1;
  }

  if (nOutputCols < 1 || nOutputRows < 1)
    THError("Given input size: (%dx%dx%d). Calculated output size: (%dx%dx%d). Output size is too small",
            nInputPlane,nInputRows,nInputCols,nInputPlane,nOutputRows,nOutputCols);

  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  THCTensor_(resizeAs)(state, gradInput, input);

  int count = THCTensor_(nElement)(state, input);

  MaxPoolBackward <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
      (count,
      THCTensor_(data)(state, gradOutput),
      THCTensor_(data)(state, indices),
      batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
      kH, kW, dH, dW, padH, padW, dilationH, dilationW,
      THCTensor_(data)(state, gradInput));
  THCudaCheck(cudaGetLastError());

  THCTensor_(free)(state, gradOutput);

  // clean
  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
}

#endif
