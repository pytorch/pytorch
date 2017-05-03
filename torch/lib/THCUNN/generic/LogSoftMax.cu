#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/LogSoftMax.cu"
#else

#include "../common.h"

void THNN_(LogSoftMax_updateOutput)(
          THCState *state,
          THCTensor *input,
          THCTensor *output)
{
  THCUNN_assertSameGPU(state, 2, input, output);

  THCTensor_(resizeAs)(state, output, input);

  bool spatial  = false;
  int batchSize = 1;
  int classSize = 0;
  int height = 0;
  int width = 0;

  int ndims = THCTensor_(nDimension)(state, input);

  if (ndims == 1)
  {
    classSize = THCTensor_(size)(state, input, 0);
    input = THCTensor_(newContiguous)(state, input);
  }
  else if (ndims == 2)
  {
    batchSize = THCTensor_(size)(state, input, 0);
    classSize = THCTensor_(size)(state, input, 1);
    input = THCTensor_(newContiguous)(state, input);
  }
  else if (ndims == 3)
  {
    spatial = true;
    classSize = THCTensor_(size)(state, input, 0);
    height = THCTensor_(size)(state, input, 1);
    width = THCTensor_(size)(state, input, 2);

    // create contiguous tensor with cuda layout from tensor with torch layout
    THCTensor *tinput = THCTensor_(new)(state);
    // C x H x W -> W x H x C
    THCTensor_(transpose)(state, tinput, input, 0, 2);
    // W x H x C -> H x W x C
    THCTensor_(transpose)(state, tinput, tinput, 0, 1);
    THCTensor *transposedInput = THCTensor_(newContiguous)(state, tinput);
    THCTensor_(free)(state, tinput);
    input = transposedInput;
  }
  else if (ndims == 4)
  {
    spatial = true;
    batchSize = THCTensor_(size)(state, input, 0);
    classSize = THCTensor_(size)(state, input, 1);
    height = THCTensor_(size)(state, input, 2);
    width = THCTensor_(size)(state, input, 3);

    // create contiguous tensor with cuda layout from tensor with torch layout
    // B x C x H x W -> B x W x H x C
    THCTensor *tinput = THCTensor_(new)(state);
    THCTensor_(transpose)(state, tinput, input, 1, 3);
    // B x W x H x C -> B x H x W x C
    THCTensor_(transpose)(state, tinput, tinput, 1, 2);
    THCTensor *transposedInput = THCTensor_(newContiguous)(state, tinput);
    THCTensor_(free)(state, tinput);
    input = transposedInput;
  }
  else
  {
    THError("1D, 2D, 3D or 4D Tensor expected");
  }

  if (!spatial)
  {
    dim3 grid(batchSize);
    dim3 block(1024);

    cunn_LogSoftMax_updateOutput_kernel<2, real, accreal>
      <<<grid, block, block.x * sizeof(accreal), THCState_getCurrentStream(state)>>>(
        THCTensor_(data)(state, output),
        THCTensor_(data)(state, input),
        classSize
    );
  }
  else
  {
    dim3 grid(batchSize);
    dim3 block(1024);

    cunn_SpatialLogSoftMax_updateOutput_kernel<real, accreal>
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
        THCTensor_(data)(state, output),
        THCTensor_(data)(state, input),
        classSize, height, width
    );
  }

  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess)
  {
    THError(cudaGetErrorString(errcode));
  }

  THCTensor_(free)(state, input);
}

void THNN_(LogSoftMax_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *output)
{
  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);

  THCTensor_(resizeAs)(state, gradInput, output);

  bool spatial  = false;
  int batchSize = 1;
  int classSize = 0;
  int height = 0;
  int width = 0;

  int ndims = THCTensor_(nDimension)(state, input);

  if (ndims == 1)
  {
    classSize = THCTensor_(size)(state, gradInput, 0);
    output = THCTensor_(newContiguous)(state, output);
    gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  }
  else if (ndims == 2)
  {
    batchSize = THCTensor_(size)(state, gradInput, 0);
    classSize = THCTensor_(size)(state, gradInput, 1);
    output = THCTensor_(newContiguous)(state, output);
    gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  }
  else if (ndims == 3)
  {
    spatial = true;
    classSize = THCTensor_(size)(state, input, 0);
    height = THCTensor_(size)(state, input, 1);
    width = THCTensor_(size)(state, input, 2);

    // create contiguous tensor with cuda layout from tensor with torch layout
    // C x H x W -> W x H x C
    THCTensor_(transpose)(state, output, output, 0, 2);
    // W x H x C -> H x W x C
    THCTensor_(transpose)(state, output, output, 0, 1);
    THCTensor *transposedOutput = THCTensor_(newContiguous)(state, output);
    THCTensor_(transpose)(state, output, output, 0, 1);
    THCTensor_(transpose)(state, output, output, 0, 2);
    output = transposedOutput;

    // create contiguous tensor with cuda layout from tensor with torch layout
    // C x H x W -> W x H x C
    THCTensor_(transpose)(state, gradOutput, gradOutput, 0, 2);
    // W x H x C -> H x W x C
    THCTensor_(transpose)(state, gradOutput, gradOutput, 0, 1);
    THCTensor *transposedGradOutput = THCTensor_(newContiguous)(state, gradOutput);
    THCTensor_(transpose)(state, gradOutput, gradOutput, 0, 1);
    THCTensor_(transpose)(state, gradOutput, gradOutput, 0, 2);
    gradOutput = transposedGradOutput;
  }
  else if (ndims == 4)
  {
    spatial = true;
    batchSize = THCTensor_(size)(state, gradInput, 0);
    classSize = THCTensor_(size)(state, input, 1);
    height = THCTensor_(size)(state, input, 2);
    width = THCTensor_(size)(state, input, 3);

    // create contiguous tensor with cuda layout from tensor with torch layout
    // B x C x H x W -> B x W x H x C
    THCTensor_(transpose)(state, output, output, 1, 3);
    // B x W x H x C -> B x H x W x C
    THCTensor_(transpose)(state, output, output, 1, 2);
    THCTensor *transposedOutput = THCTensor_(newContiguous)(state, output);
    THCTensor_(transpose)(state, output, output, 1, 2);
    THCTensor_(transpose)(state, output, output, 1, 3);
    output = transposedOutput;

    // create contiguous tensor with cuda layout from tensor with torch layout
    // B x C x H x W -> B x W x H x C
    THCTensor_(transpose)(state, gradOutput, gradOutput, 1, 3);
    // B x W x H x C -> B x H x W x C
    THCTensor_(transpose)(state, gradOutput, gradOutput, 1, 2);
    THCTensor *transposedGradOutput = THCTensor_(newContiguous)(state, gradOutput);
    THCTensor_(transpose)(state, gradOutput, gradOutput, 1, 2);
    THCTensor_(transpose)(state, gradOutput, gradOutput, 1, 3);
    gradOutput = transposedGradOutput;
  }
  else
  {
    THError("1D, 2D, 3D or 4D Tensor expected");
  }

  if (!spatial)
  {
    dim3 grid(batchSize);
    dim3 block(1024);

    cunn_LogSoftMax_updateGradInput_kernel<2, real, accreal>
      <<<grid, block, block.x * sizeof(accreal), THCState_getCurrentStream(state)>>>(
        THCTensor_(data)(state, gradInput),
        THCTensor_(data)(state, output),
        THCTensor_(data)(state, gradOutput),
        classSize
    );
  }
  else
  {
    dim3 grid(batchSize);
    dim3 block(1024);

    cunn_SpatialLogSoftMax_updateGradInput_kernel<real, accreal>
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
        THCTensor_(data)(state, gradInput),
        THCTensor_(data)(state, output),
        THCTensor_(data)(state, gradOutput),
        classSize, height, width
    );
  }

  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess)
  {
    THError(cudaGetErrorString(errcode));
  }

  THCTensor_(free)(state, gradOutput);
  THCTensor_(free)(state, output);
}

#endif
