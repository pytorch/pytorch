#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SoftMax.cu"
#else

#include "../common.h"

void THNN_(SoftMax_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int _dim)
{
  THCUNN_assertSameGPU(state, 2, input, output);

  input = THCTensor_(newContiguous)(state, input);
  THCTensor_(resizeAs)(state, output, input);
  int64_t batchSize, dim, stride0, stride1 = 1;
  int64_t blocksY = 1, blocksZ = 1;

  if (input->nDimension == 1)
  {
    batchSize = 1;
    dim = input->size[0];
    stride0 = 1;
    THArgCheck(_dim == 0, 4, "dim has to be 0 for 1D input");
  }
  else if (input->nDimension == 2)
  {
    batchSize = input->size[0];
    dim = input->size[1];
    stride0 = 1;
    THArgCheck(_dim == 1, 4, "dim has to be 1 for 2D input");
  }
  else if (input->nDimension == 3)
  {
    batchSize = 1;
    dim = input->size[0];
    blocksY = input->size[1];
    blocksZ = input->size[2];
    stride0 = blocksY * blocksZ;
    stride1 = blocksZ;
    THArgCheck(_dim == 0, 4, "dim has to be 0 for 3D input");
  }
  else if (input->nDimension == 4)
  {
    batchSize = input->size[0];
    dim = input->size[1];
    blocksY = input->size[2];
    blocksZ = input->size[3];
    stride0 = blocksY * blocksZ;
    stride1 = blocksZ;
    THArgCheck(_dim == 1, 4, "dim has to be 1 for 4D input");
  }
  else
  {
    THError("1D, 2D, 3D or 4D tensor expected");
  }

  // when possible use only 2d grid of thread blocks to stay compatible with compute capability 2.X devices.
  if (blocksY * blocksZ < 65536)
  {
    blocksY *= blocksZ;
    blocksZ = 1;
    if (input->nDimension == 3 || input->nDimension == 4) {
      stride0 = blocksY * blocksZ;
      stride1 = blocksZ;
    }
  }

  dim3 blocks(batchSize, blocksY, blocksZ);
  dim3 threads(SOFTMAX_THREADS);
  cunn_SoftMax_updateOutput_kernel<real, accreal><<<blocks, threads, 0, THCState_getCurrentStream(state)>>>(
    THCTensor_(data)(state, output),
    THCTensor_(data)(state, input),
    batchSize, dim, stride0, stride1
  );
  THCudaCheck(cudaGetLastError());

  THCTensor_(free)(state, input);
}

void THNN_(SoftMax_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *output,
           int _dim)
{
  THCUNN_check_nElement(state, output, gradOutput);
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);

  output = THCTensor_(newContiguous)(state, output);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  THCTensor_(resizeAs)(state, gradInput, output);
  int64_t batchSize, dim, stride0, stride1 = 1;
  int64_t blocksY = 1, blocksZ = 1;

  if (gradInput->nDimension == 1)
  {
    batchSize = 1;
    dim = gradInput->size[0];
    stride0 = 1;
    THArgCheck(_dim == 0, 6, "dim has to be 0 for 1D input");
  }
  else if (gradInput->nDimension == 2)
  {
    batchSize = gradInput->size[0];
    dim = gradInput->size[1];
    stride0 = 1;
    THArgCheck(_dim == 1, 6, "dim has to be 0 for 2D input");
  }
  else if (gradInput->nDimension == 3)
  {
    batchSize = 1;
    dim = gradInput->size[0];
    blocksY = gradInput->size[1];
    blocksZ = gradInput->size[2];
    stride0 = blocksY * blocksZ;
    stride1 = blocksZ;
    THArgCheck(_dim == 0, 6, "dim has to be 0 for 3D input");
  }
  else if (gradInput->nDimension == 4)
  {
    batchSize = gradInput->size[0];
    dim = gradInput->size[1];
    blocksY = gradInput->size[2];
    blocksZ = gradInput->size[3];
    stride0 = blocksY * blocksZ;
    stride1 = blocksZ;
    THArgCheck(_dim == 1, 6, "dim has to be 0 for 4D input");
  }
  else
  {
    THError("1D, 2D, 3D or 4D tensor expected");
  }

  // when possible use only 2d grid of thread blocks to stay compatible with compute capability 2.X devices.
  if (blocksY * blocksZ < 65536)
  {
    blocksY *= blocksZ;
    blocksZ = 1;
    if (output->nDimension == 3 || output->nDimension == 4) {
      stride0 = blocksY * blocksZ;
      stride1 = blocksZ;
    }
  }

  dim3 blocks(batchSize, blocksY, blocksZ);
  dim3 threads(SOFTMAX_THREADS);
  cunn_SoftMax_updateGradInput_kernel<real, accreal><<<blocks, threads, 0, THCState_getCurrentStream(state)>>>(
    THCTensor_(data)(state, gradInput),
    THCTensor_(data)(state, output),
    THCTensor_(data)(state, gradOutput),
    batchSize, dim, stride0, stride1
  );
  THCudaCheck(cudaGetLastError());

  THCTensor_(free)(state, gradOutput);
  THCTensor_(free)(state, output);
}

#endif
