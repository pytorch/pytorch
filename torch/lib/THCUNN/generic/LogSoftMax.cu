#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/LogSoftMax.cu"
#else

#include "../common.h"

void THNN_(LogSoftMax_updateOutput)(
          THCState *state,
          THCTensor *input,
          THCTensor *output,
          int dim)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THArgCheck(dim >= 0 && dim < input->nDimension, 4,
	     "dim out of range (got %d, but input has %d dims)", dim, input->nDimension);
	THArgCheck(TensorUtils<THCTensor>::canUse32BitIndexMath(state, input), 4,
	     "input tensor is too large (unsupported size. file a feature request)");

  THCTensor_(resizeAs)(state, output, input);

  uint64_t outer_size = 1;
  uint64_t dim_size = input->size[dim];
  uint64_t inner_size = 1;
  for (uint64_t i = 0; i < dim; ++i)
    outer_size *= input->size[i];
  for (uint64_t i = dim + 1; i < input->nDimension; ++i)
    inner_size *= input->size[i];

  // This kernel spawns a block of 1024 threads per each element in the batch.
  // XXX: it assumes that inner_size == 1
  input = THCTensor_(newContiguous)(state, input);
  if (inner_size == 1 && dim_size >= 64) {
    dim3 grid(outer_size);
    dim3 block(1024);

    cunn_LogSoftMax_updateOutput_kernel<2, real, accreal>
      <<<grid, block, block.x * sizeof(accreal), THCState_getCurrentStream(state)>>>(
        THCTensor_(data)(state, output),
        THCTensor_(data)(state, input),
        dim_size
    );
  // This kernel runs in a 2D grid, where each application along y dimension has a fixed
  // outer_size, and runs in parallel over inner_size. Dimension x is parallel over outer_size.
  // Reductions over dim are done in a single-threaded manner.
  } else {
    dim3 grid, block;
    uint32_t block_size = 1024;
    while (block_size > inner_size) block_size >>= 1; // block_size = floor(log2(inner_size))
    int max_active_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                                                  &cunn_SpatialLogSoftMax_updateOutput_kernel<real, accreal>,
                                                  block_size, 0);
    max_active_blocks *= THCState_getCurrentDeviceProperties(state)->multiProcessorCount;
    LogSoftMax_getSpatialGridSize(block_size, max_active_blocks, outer_size, dim_size, inner_size, grid, block);

    cunn_SpatialLogSoftMax_updateOutput_kernel<real, accreal>
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
        THCTensor_(data)(state, output),
        THCTensor_(data)(state, input),
        outer_size, dim_size, inner_size
    );
  }
  THCudaCheck(cudaGetLastError());

  THCTensor_(free)(state, input);
}

void THNN_(LogSoftMax_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *output,
           int dim)
{
  THArgCheck(dim >= 0 && dim < output->nDimension, 6,
	     "dim out of range (got %d, but input has %d dims)", dim, output->nDimension);
	THArgCheck(TensorUtils<THCTensor>::canUse32BitIndexMath(state, output), 6,
	     "input tensor is too large (unsupported size. file a feature request)");
  THCUNN_check_nElement(state, output, gradOutput);
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);

  THCTensor_(resizeAs)(state, gradInput, output);

  uint64_t outer_size = 1;
  uint64_t dim_size = output->size[dim];
  uint64_t inner_size = 1;
  for (uint64_t i = 0; i < dim; ++i)
    outer_size *= output->size[i];
  for (uint64_t i = dim + 1; i < output->nDimension; ++i)
    inner_size *= output->size[i];

  output = THCTensor_(newContiguous)(state, output);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  // See descriptions of kernels above.
  if (inner_size == 1 && dim_size >= 64) {
    dim3 grid(outer_size);
    dim3 block(1024);

    cunn_LogSoftMax_updateGradInput_kernel<2, real, accreal>
      <<<grid, block, block.x * sizeof(accreal), THCState_getCurrentStream(state)>>>(
        THCTensor_(data)(state, gradInput),
        THCTensor_(data)(state, output),
        THCTensor_(data)(state, gradOutput),
        dim_size
    );
  } else {
    dim3 grid, block;
    uint32_t block_size = 1024;
    while (block_size > inner_size) block_size >>= 1; // block_size = floor(log2(inner_size))
    int max_active_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                                                  &cunn_SpatialLogSoftMax_updateGradInput_kernel<real, accreal>,
                                                  block_size, 0);
    max_active_blocks *= THCState_getCurrentDeviceProperties(state)->multiProcessorCount;
    LogSoftMax_getSpatialGridSize(block_size, max_active_blocks, outer_size, dim_size, inner_size, grid, block);

    cunn_SpatialLogSoftMax_updateGradInput_kernel<real, accreal>
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
        THCTensor_(data)(state, gradInput),
        THCTensor_(data)(state, output),
        THCTensor_(data)(state, gradOutput),
        outer_size, dim_size, inner_size
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
