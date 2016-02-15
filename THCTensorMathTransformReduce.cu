#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCBlas.h"
#include "THCTensorCopy.h"
#include "THCTensorRandom.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"

#include <thrust/functional.h>

/* A set of reduction kernels that take in binary ops on thrust pairs (of value, index).
   These are useful when you not only have to do a reduction, but you might have
   to preserve the location of contention (for example min/max operations).
   The structure of the kernels follows the structure of the reduction kernels.
*/
template<class BinaryFunction>
__global__ void THCudaTensor_kernel_transformReduceOuterDimIndex(float *tgt1, float *tgt2,
                                                             float *src_,
                                                             unsigned num_orows,
                                                             unsigned num_irows,
                                                             unsigned row_size,
                                                             thrust::pair<float,float> init,
                                                             BinaryFunction binary_op)
{
  for (unsigned orow = blockIdx.x; orow < num_orows; orow += gridDim.x) {
    for (unsigned irow = blockIdx.y * blockDim.x + threadIdx.x; irow < num_irows; irow += gridDim.y * blockDim.x) {
      float *src = src_ + orow * row_size * num_irows + irow;
      thrust::pair<float,float> acc = init;

      for (unsigned col = 0; col < row_size; ++col) {
        acc = binary_op(thrust::make_pair(*src, col+1), acc); // i+1 for 1-indexing
        src += num_irows;
      }
      tgt1[orow * num_irows + irow] = acc.first;
      tgt2[orow * num_irows + irow] = acc.second;
    }
  }
}

template<class BinaryFunction>
__host__ void THCudaTensor_transformReduceOuterDimIndex(THCState *state, THCudaTensor *tgt1, THCudaTensor *tgt2,
                                                   THCudaTensor *src,
                                                   long rdim, thrust::pair<float,float> init,
                                                   BinaryFunction binary_op)
{
  unsigned ndim = THCudaTensor_nDimension(state, src);
  unsigned num_orows = 1;
  for (unsigned dim = 0; dim < rdim; dim++) {
    num_orows *= THCudaTensor_size(state, src, dim);
  }
  unsigned row_size = THCudaTensor_size(state, src, rdim);
  unsigned num_irows = 1;
  for (unsigned dim = rdim + 1; dim < ndim; dim++) {
    num_irows *= THCudaTensor_size(state, src, dim);
  }

  dim3 threads(min(512, num_irows));
  unsigned maxGridDim = 1024;
  dim3 grid(min(maxGridDim, num_orows), min(maxGridDim, THCCeilDiv(num_irows, threads.x)));

  THCudaTensor_kernel_transformReduceOuterDimIndex<<<grid, threads, 0, THCState_getCurrentStream(state)>>>(
    THCudaTensor_data(state, tgt1), THCudaTensor_data(state, tgt2),
    THCudaTensor_data(state, src), num_orows, num_irows, row_size, init, binary_op);
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}

/* Reduce the innermost dimension of a tensor (on thrust::pair functors which are (value, index))
 *
 * For an n-d tensor (n <= 4) where the reduction is along the innermost dimension:
 *
 * - block.x is the innermost dimension, i.e. dimension 0;
 * - block.y and grid.y make up dimension 1; and
 * - grid.x and grid z are the remaining two outer dimensions (if any)
 *
 * Reduction along other dimensions is handled in a separate kernel.
 */
template<class BinaryFunction>
__global__ void THCudaTensor_kernel_transformReduceInnermostDimIndex(
  float *tgt1, float* tgt2, float *src_,
  unsigned num_rows, unsigned row_size,
  thrust::pair<float,float> init, BinaryFunction binary_op)
{
  __shared__ float sbuf[32][16];
  __shared__ float ibuf[32][16];

  for (unsigned block_row = blockIdx.x * blockDim.y; block_row < num_rows; block_row += blockDim.y * gridDim.x) {
    unsigned row = block_row + threadIdx.y;
    thrust::pair<float,float> acc = init;
    if (row < num_rows) {
      float *src = src_ + row * row_size;
      // Sequential reduction within a thread.
      for (unsigned col = threadIdx.x; col < row_size; col += blockDim.x) {
        acc = binary_op(thrust::make_pair(src[col], col+1), acc);
      }
    }

    sbuf[threadIdx.y][threadIdx.x] = acc.first;
    ibuf[threadIdx.y][threadIdx.x] = acc.second;

    // Reduce intermediate values to single value.
    float* sline = &sbuf[threadIdx.y][0];
    float* iline = &ibuf[threadIdx.y][0];
    for (unsigned s = 8; s > 0; s >>= 1) {
      if (row < num_rows && threadIdx.x < s) {
        thrust::pair<float,float> arg1 = thrust::make_pair<float,float>(sline[threadIdx.x], iline[threadIdx.x]);
        thrust::pair<float,float> arg2 = thrust::make_pair<float,float>(sline[threadIdx.x + s], iline[threadIdx.x + s]);
        thrust::pair<float,float> res = binary_op(arg1, arg2);
        sline[threadIdx.x] = res.first;
        iline[threadIdx.x] = res.second;
      }
      __syncthreads();
    }

    if (row < num_rows && threadIdx.x == 0) {
      tgt1[row] = sline[0];
      tgt2[row] = iline[0];
    }
    __syncthreads();
  }
}

template<class BinaryFunction>
__host__ void THCudaTensor_transformReduceInnermostDimIndex(
  THCState *state, THCudaTensor *tgt1, THCudaTensor *tgt2, THCudaTensor *src,
  thrust::pair<float,float> init, BinaryFunction binary_op)
{
  unsigned ndim = THCudaTensor_nDimension(state, src);
  unsigned num_rows = 1;
  for (unsigned dim = 0; dim < ndim - 1; dim++) {
    num_rows *= THCudaTensor_size(state, src, dim);
  }
  unsigned row_size = THCudaTensor_size(state, src, ndim - 1);

  dim3 threads(16, 32);
  dim3 grid(min(1024, THCCeilDiv(num_rows, threads.y)));

  THCudaTensor_kernel_transformReduceInnermostDimIndex<<<grid, threads, 0, THCState_getCurrentStream(state)>>>(
    THCudaTensor_data(state, tgt1), THCudaTensor_data(state, tgt2),
    THCudaTensor_data(state, src), num_rows, row_size, init, binary_op);
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}

template<class BinaryFunction>
void THCudaTensor_reduceDimIndex(THCState *state, THCudaTensor *tgt1_, THCudaTensor *tgt2_, THCudaTensor *src,
                             long dimension, thrust::pair<float,float> init,
                                     BinaryFunction binary_op)
{
  THArgCheck(dimension >= 0 && dimension < THCudaTensor_nDimension(state, src), 3, "dimension out of range");

  THLongStorage *dim = THCudaTensor_newSizeOf(state, src);
  THLongStorage_set(dim, dimension, 1);
  THCudaTensor_resize(state, tgt1_, dim, NULL);
  THCudaTensor_resize(state, tgt2_, dim, NULL);
  THLongStorage_free(dim);

  THCudaTensor *tgt1 = THCudaTensor_newContiguous(state, tgt1_);
  THCudaTensor *tgt2 = THCudaTensor_newContiguous(state, tgt2_);
  src = THCudaTensor_newContiguous(state, src);

  if(dimension == THCudaTensor_nDimension(state, src)-1) {
    THCudaTensor_transformReduceInnermostDimIndex(state, tgt1, tgt2, src, init, binary_op);
  } else {
    THCudaTensor_transformReduceOuterDimIndex(state, tgt1, tgt2, src, dimension, init, binary_op);
  }

  THCudaTensor_free(state, src);
  THCudaTensor_freeCopyTo(state, tgt1, tgt1_);
  THCudaTensor_freeCopyTo(state, tgt2, tgt2_);
}

struct maxvalue_functor
{
  __host__ __device__ thrust::pair<float,float> operator()(const thrust::pair<float,float> &a,
                                                            const thrust::pair<float,float> &b)
  {
    if (a.first > b.first) return a;
    else return b;
  }
};

void THCudaTensor_max(THCState *state, THCudaTensor *values, THCudaTensor *indices, THCudaTensor *src, long dimension)
{
  THAssert(THCudaTensor_checkGPU(state, 3, values, indices, src));
  const float minfloat32 = -3.402823466e+38f;
  thrust::pair<float,float> init = thrust::make_pair<float,float>(minfloat32, 1);
  return THCudaTensor_reduceDimIndex(state, values, indices, src, dimension, init,
                                 maxvalue_functor());
}

struct minvalue_functor
{
  __host__ __device__ thrust::pair<float,float> operator()(const thrust::pair<float,float> &a,
                                                            const thrust::pair<float,float> &b)
  {
    if (a.first < b.first) return a;
    else return b;
  }
};

void THCudaTensor_min(THCState *state, THCudaTensor *values, THCudaTensor *indices, THCudaTensor *src, long dimension)
{
  THAssert(THCudaTensor_checkGPU(state, 3, values, indices, src));
  const float maxfloat32 = 3.402823466e+38f;
  thrust::pair<float,float> init = thrust::make_pair<float,float>(maxfloat32, 1);
  return THCudaTensor_reduceDimIndex(state, values, indices, src, dimension, init,
                                     minvalue_functor());
}
