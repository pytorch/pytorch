#include "THCUNN.h"
#include "common.h"

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>


/*
 * Description:
 */

__device__ int translate_idx(int ii, int d1, int d2, int d3, int scale_factor)
{
  int x, y, z, w;
  w = ii % d3;
  ii = ii/d3;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  w = w/scale_factor;
  z = z/scale_factor;
  d2 /= scale_factor;
  d3 /= scale_factor;
  return (((x*d1+y)*d2)+z)*d3+w;

}
__device__ int translate_idx_inv(int ii, int d1, int d2, int d3, int scale_factor, int off_x, int off_y)
{
  int x, y, z, w;
  w = ii % d3;
  ii = ii/d3;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  w = w*scale_factor+off_x;
  z = z*scale_factor+off_y;
  d2 *= scale_factor;
  d3 *= scale_factor;
  return (((x*d1+y)*d2)+z)*d3+w;

}

__global__ void upscale(float *input, float *output, long no_elements,
                        int scale_factor, int d1, int d2, int d3)
{
  // output offset:
  long ii = threadIdx.x + blockDim.x * blockIdx.x;
  ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
  if (ii >= no_elements) return;
  int ipidx = translate_idx(ii, d1, d2, d3, scale_factor);
  output[ii]=input[ipidx];
}


void THNN_CudaSpatialUpSamplingNearest_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output, int scale_factor)
{
  THCudaTensor_zero(state, output);

  THCUNN_assertSameGPU(state, 2, input, output);

  input = THCudaTensor_newContiguous(state, input);
  // This is for allocating output Tensor
  long no_elements = 1;
  for(int i = 0; i < input->nDimension; i++){
    no_elements *= input->size[i];
  }
  no_elements *= scale_factor * scale_factor;

  int d1;
  int d2;
  int d3;

  if (input->nDimension == 3) {
    d1 = output->size[0];
    d2 = output->size[1];
    d3 = output->size[2];
  } else {
    d1 = output->size[1];
    d2 = output->size[2];
    d3 = output->size[3];
  }

  float *input_data = THCudaTensor_data(state, input);
  float *output_data = THCudaTensor_data(state, output);

  // cuda blocks & threads:
  long nthreads = 256;
  // Max number of blocks: http://en.wikipedia.org/wiki/CUDA
  // 65535 for SM 2.x, 2^32 -1 for >= 3.0
  // TODO: When we move to SM 3.5 we should update this
  long n_xblocks = min(max((int)ceil((float)no_elements / nthreads), 1), 65535);
  long n_yblocks = (long)ceil((float)no_elements / (float)(n_xblocks * nthreads));
  if (n_yblocks > 65535) {
    THError("Input size is too large!  aborting");
  }
  dim3 blocks(n_xblocks, n_yblocks);
  dim3 threads(nthreads);

  // kernel:
  upscale<<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (input_data, output_data, no_elements, scale_factor, d1, d2, d3);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialUpSamplingNearest.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }

  // final cut:
  THCudaTensor_free(state, input);
}

/*
 * Description:
 */
__global__ void downscale(float *gradInput_data, float *gradOutput_data, long no_elements,
                              int scale_factor, int d1, int d2, int d3)
{
  // output offset:
  long ii = threadIdx.x + blockDim.x * blockIdx.x;
  ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
  if (ii >= no_elements) return;
  for (int i=0; i < scale_factor; i++){
    for(int j=0; j < scale_factor; j++){
      int ipidx = translate_idx_inv(ii, d1, d2, d3, scale_factor, i, j);
      gradInput_data[ii] += gradOutput_data[ipidx];
    }
  }
}


void THNN_CudaSpatialUpSamplingNearest_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput, THCudaTensor *gradInput, int scale_factor)
{
  THCUNN_assertSameGPU(state, 2, gradOutput, gradInput);

  THCudaTensor_zero(state, gradInput);

  float *gradInput_data = THCudaTensor_data(state, gradInput);
  float *gradOutput_data = THCudaTensor_data(state, gradOutput);

  long no_elements = 1;
  for(int i = 0; i < gradInput->nDimension; i++){
    no_elements *= gradInput->size[i];
  }

  int d1;
  int d2;
  int d3;

  if (gradInput->nDimension == 3) {
    d1 = gradInput->size[0];
    d2 = gradInput->size[1];
    d3 = gradInput->size[2];
  } else {
    d1 = gradInput->size[1];
    d2 = gradInput->size[2];
    d3 = gradInput->size[3];
  }

  // cuda blocks & threads:
  long nthreads = 256;
  // Max number of blocks: http://en.wikipedia.org/wiki/CUDA
  // 65535 for SM 2.x, 2^32 -1 for >= 3.0
  // TODO: When we move to SM 3.5 we should update this
  long n_xblocks = min(max((int)ceil((float)no_elements / nthreads), 1), 65535);
  long n_yblocks = (long)ceil((float)no_elements / (float)(n_xblocks * nthreads));
  if (n_yblocks > 65535) {
    THError("Input size is too large!  aborting");
  }
  dim3 blocks(n_xblocks, n_yblocks);
  dim3 threads(nthreads);

  // kernel:
  downscale<<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data, no_elements,
    scale_factor, d1, d2, d3);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialUpSamplingNearest.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
}
