#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCBlas.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"

#include <thrust/functional.h>

/* Perform an inclusive scan along an outer dimension of a tensor.
 *
 * - num_orows is the size of the flattened outer dimensions;
 * - num_irows is the size of the flattened inner dimensions;
 * - row_size is the size of the dimension along which to compute the variance;
 *
 * The dimensions to the outside and inside of the specified dimension are considered as flattened.
 * Thread blocks with the same blockIdx.y process an "outer row" (i.e. an element of the flattened
 * outer dimensions, which contains several "inner rows").
 * Each thread processes a single inner row at a time.
 */
template<class BinaryOp>
__global__ void THCudaTensor_kernel_scanOuterDim(float *tgt_, float *src_,
                                                 unsigned num_orows, unsigned num_irows, unsigned row_size,
                                                 float init, BinaryOp binary_op)
{
  for (unsigned orow = blockIdx.x; orow < num_orows; orow += gridDim.x) {
    for (unsigned irow = blockIdx.y * blockDim.x + threadIdx.x; irow < num_irows; irow += gridDim.y * blockDim.x) {
      float *src = src_ + orow * row_size * num_irows + irow;
      float *tgt = tgt_ + orow * row_size * num_irows + irow;
      float acc = init;

      for (unsigned col = 0; col < row_size; ++col) {
        acc = binary_op(acc, *src);
        *tgt = acc;

        src += num_irows;
        tgt += num_irows;
      }
    }
  }
}

template<class BinaryOp>
__host__ void THCudaTensor_scanOuterDim(THCState *state, THCudaTensor *tgt, THCudaTensor *src, long dimension,
                                        float init, BinaryOp binary_op)
{
  unsigned ndim = THCudaTensor_nDimension(state, src);
  // Treat all outer dimensions (i.e. dim < dimension) as one.
  unsigned num_orows = 1;
  for (unsigned dim = 0; dim < dimension; dim++) {
    num_orows *= THCudaTensor_size(state, src, dim);
  }
  unsigned row_size = THCudaTensor_size(state, src, dimension);
  // Treat all inner dimensions (i.e. dim > dimension) as one.
  unsigned num_irows = 1;
  for (unsigned dim = dimension + 1; dim < ndim; dim++) {
    num_irows *= THCudaTensor_size(state, src, dim);
  }

  dim3 threads(min(512, num_irows));
  unsigned maxGridDim = 1024;
  dim3 grid(min(maxGridDim, num_orows), min(maxGridDim, THCCeilDiv(num_irows, threads.x)));

  THCudaTensor_kernel_scanOuterDim<<<grid, threads, 0, THCState_getCurrentStream(state)>>>(
      THCudaTensor_data(state, tgt), THCudaTensor_data(state, src), num_orows, num_irows, row_size, init, binary_op);
  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}


/* Perform an inclusive scan along the innermost dimension of a tensor.
 *
 * - num_rows is the size of the flattened outer dimensions;
 * - row_size is the size of the innermost dimension;
 *
 * The outer dimensions of the tensor are considered as a single dimension, i.e. the tensor is
 * considered as having 'num_rows' rows of size 'row_size'.
 * Each thread block processes one or more sets of contiguous rows (processing multiple rows
 * per thread block is quicker than processing a single row, especially for short rows).
 */
template<int num_threads_x, int num_threads_y, class BinaryFunction>
__global__ void THCudaTensor_kernel_scanInnermostDim(float *tgt_, float *src_,
                                                     unsigned num_rows, unsigned row_size,
                                                     float init, BinaryFunction binary_op)
{
  __shared__ float sbuf[num_threads_y][2 * num_threads_x];

  float* row_buf = sbuf[threadIdx.y];

  for (unsigned block_row = blockIdx.x * blockDim.y;
       block_row < num_rows;
       block_row += blockDim.y * gridDim.x) {
    unsigned row = block_row + threadIdx.y;
    float block_total = init;

    float *row_src = src_ + row * row_size;
    float *row_tgt = tgt_ + row * row_size;

    // Perform scan on one block at a time, keeping track of the total value of
    // all blocks processed so far.
    for (unsigned block_col = 0; block_col < row_size; block_col += 2 * num_threads_x) {
      // Load data into shared memory (two values per thread).
      unsigned col1 = block_col + threadIdx.x;
      unsigned col2 = block_col + num_threads_x + threadIdx.x;
      if (row < num_rows) {
        if (col1 < row_size) {
          row_buf[threadIdx.x] = row_src[col1];
        } else {
          row_buf[threadIdx.x] = init;
        }

        if (col2 < row_size) {
          row_buf[num_threads_x + threadIdx.x] = row_src[col2];
        } else {
          row_buf[num_threads_x + threadIdx.x] = init;
        }

        // Add the total value of all previous blocks to the first value of this block.
        if (threadIdx.x == 0) {
          row_buf[0] = binary_op(row_buf[0], block_total);
        }
      }
      __syncthreads();

      // Parallel reduction (up-sweep).
      for (unsigned s = num_threads_x, d = 1; s >= 1; s >>= 1, d <<= 1) {
        if (row < num_rows && threadIdx.x < s) {
          unsigned offset = (2 * threadIdx.x + 1) * d - 1;
          row_buf[offset + d] = binary_op(row_buf[offset], row_buf[offset + d]);
        }
        __syncthreads();
      }

      // Down-sweep.
      for (unsigned s = 2, d = num_threads_x / 2; d >= 1; s <<= 1, d >>= 1) {
        if (row < num_rows && threadIdx.x < s - 1) {
          unsigned offset = 2 * (threadIdx.x + 1) * d - 1;
          row_buf[offset + d] = binary_op(row_buf[offset], row_buf[offset + d]);
        }
        __syncthreads();
      }

      // Write back to output.
      if (row < num_rows) {
        if (col1 < row_size) row_tgt[col1] = row_buf[threadIdx.x];
        if (col2 < row_size) row_tgt[col2] = row_buf[num_threads_x + threadIdx.x];
      }
      block_total = row_buf[2 * num_threads_x - 1];
      __syncthreads();
    }
  }
}

template<class BinaryFunction>
__host__ void THCudaTensor_scanInnermostDim(THCState *state, THCudaTensor *tgt, THCudaTensor *src, float init, BinaryFunction binary_op)
{
  unsigned ndim = THCudaTensor_nDimension(state, src);
  // Treat all outer dimensions as a single dimension.
  unsigned num_rows = 1;
  for (unsigned dim = 0; dim < ndim - 1; dim++) {
    num_rows *= THCudaTensor_size(state, src, dim);
  }
  unsigned row_size = THCudaTensor_size(state, src, ndim - 1);

  dim3 threads(16, 32);
  dim3 grid(min(1024, THCCeilDiv(num_rows, threads.y)));

  THCudaTensor_kernel_scanInnermostDim<16, 32><<<grid, threads, 0, THCState_getCurrentStream(state)>>>(
      THCudaTensor_data(state, tgt), THCudaTensor_data(state, src), num_rows, row_size, init, binary_op);
  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}

template<class BinaryFunction>
void THCudaTensor_scanDim(THCState *state, THCudaTensor *self_, THCudaTensor *src, long dimension, float init, BinaryFunction binary_op)
{
  THCudaTensor_resizeAs(state, self_, src);

  THCudaTensor *self = THCudaTensor_newContiguous(state, self_);
  src = THCudaTensor_newContiguous(state, src);

  if (dimension == THCudaTensor_nDimension(state, src) - 1) {
    THCudaTensor_scanInnermostDim(state, self, src, init, binary_op);
  } else {
    THCudaTensor_scanOuterDim(state, self, src, dimension, init, binary_op);
  }

  THCudaTensor_free(state, src);
  THCudaTensor_freeCopyTo(state, self, self_);
}

void THCudaTensor_cumsum(THCState *state, THCudaTensor *self, THCudaTensor *src, long dimension)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self, src));
  return THCudaTensor_scanDim(state, self, src, dimension, 0.0f, thrust::plus<float>());
}

void THCudaTensor_cumprod(THCState *state, THCudaTensor *self, THCudaTensor *src, long dimension)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self, src));
  return THCudaTensor_scanDim(state, self, src, dimension, 1.0f, thrust::multiplies<float>());
}
