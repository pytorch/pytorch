#include "THCTensorConv.h"
#include "THCTensorMath.h"
#include "THCTensorCopy.h"
#include "THCGeneral.h"
#include <stdio.h>

/*
 * Description:
 *   This code provides convolutions and xcorrelations that are API compatible with
 *   the ones in THLabConv.
 *
 * History:
 *   Sept 11, 2011, 11:59PM  -  Clement Farabet  -  Optimized RevConv by a good x2
 *   July 22, 2011, 8:38PM   -  Clement Farabet  -  All Valid/Full/XCORR/CONV implemented
 *   July 22, 2011, 4:00PM   -  Clement Farabet  -  Rewrote for loop to insure memory coalescing
 *   July 21, 2011, 11:21PM  -  Clement Farabet  -  Creation, based conv2d routine
 */

#define CUDA_SHARED_MEM_SIZE (12*1024-32) // this is given by nVidia: max shared mem per block

/*
 * Description:
 *   base conv2D routine: 3D input, 3D output, 4D kernel
 *
 *   - all chunks of data should be contiguous
 *   - the swapkernel flag can be used to generate a conv2 instead of xcorr2
 *   - the templated kernel size is useful to generate code that's 2x faster
 *     but can be set to 0 to allow arbitrary kernel sizes
 */
template <bool swapkernel, int T_kernel_h, int T_kernel_w>
  __global__ void conv2generic(float *input, float *kernel, float *output,
                               int input_n, int input_h, int input_w,
                               int kernel_n, int kernel_h, int kernel_w,
                               int stride_h, int stride_w)
{
  // output dimensions
  int output_h = (input_h - kernel_h) / stride_h + 1;
  int output_w = (input_w - kernel_w) / stride_w + 1;

  // xcorr or conv
  int koffset = swapkernel ? kernel_w*kernel_h-1 : 0;

  // nb outputs
  int output_n = kernel_n / input_n;

  // generate offsets according to block/thread ids
  int xx_start = threadIdx.x;
  int xx_end = output_w;
  int xx_step = blockDim.x;

  int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
  int yy_end = output_h;
  int yy_step = blockDim.y*gridDim.y;

  int oo_start = blockIdx.x;
  int oo_end = oo_start+1;

  int ii_start = (blockIdx.x / output_n) * input_n;
  int ii_end = ii_start + input_n;

  // nb threads, unique thread id
  int tid = blockDim.x*blockDim.y*threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  int nthreads = blockDim.x * blockDim.y * blockDim.z;

  // iterators
  int oo, ii, xx, yy, kx, ky, kk;

  // do the kernels fit in shared mem ?
  if (input_n*kernel_w*kernel_h <= CUDA_SHARED_MEM_SIZE) {

    // put the kernel in shared memory
    __shared__ float shared_kernel[CUDA_SHARED_MEM_SIZE];

    // first thread of each block does the copy
    for (kk = tid; kk < kernel_w*kernel_h*input_n; kk += nthreads) {
      shared_kernel[kk] = kernel[input_n*kernel_w*kernel_h*(oo_start % output_n) + kk];
    }
    __syncthreads();

    // templated kernel size
    if ((T_kernel_w > 0) && (T_kernel_h > 0)) {
      // unrolled convolution loop
      for(oo = oo_start; oo < oo_end; oo++) {
        for(ii = ii_start; ii < ii_end; ii++) {
          for(yy = yy_start; yy < yy_end; yy+=yy_step) {
            for(xx = xx_start; xx < xx_end; xx+=xx_step) {
              // Dot product in two dimensions... (between input image and the mask)
              float *input_p = input + ii*input_h*input_w + yy*stride_h*input_w + xx*stride_w;
              float *output_p = output + oo*output_h*output_w + yy*output_w + xx;
              float *kernel_p = shared_kernel + (ii % input_n)*kernel_w*kernel_h + koffset;
              float sum = 0;
              if (swapkernel) {
#pragma unroll
                for(ky = 0; ky < T_kernel_h; ky++) {
#pragma unroll
                  for(kx = 0; kx < T_kernel_w; kx++) {
                    sum += input_p[kx]*(*kernel_p--);
                  }
                  input_p += input_w;
                }
              } else {
#pragma unroll
                for(ky = 0; ky < T_kernel_h; ky++) {
#pragma unroll
                  for(kx = 0; kx < T_kernel_w; kx++) {
                    sum += input_p[kx]*(*kernel_p++);
                  }
                  input_p += input_w;
                }
              }
              *output_p += sum;
            }
          }
        }
      }
    } else {
      // default convolution loop
      for(oo = oo_start; oo < oo_end; oo++) {
        for(ii = ii_start; ii < ii_end; ii++) {
          for(yy = yy_start; yy < yy_end; yy+=yy_step) {
            for(xx = xx_start; xx < xx_end; xx+=xx_step) {
              // Dot product in two dimensions... (between input image and the mask)
              float *input_p = input + ii*input_h*input_w + yy*stride_h*input_w + xx*stride_w;
              float *output_p = output + oo*output_h*output_w + yy*output_w + xx;
              float *kernel_p = shared_kernel + (ii % input_n) * kernel_w * kernel_h + koffset;
              float sum = 0;
              if (swapkernel) {
                for(ky = 0; ky < kernel_h; ky++) {
#pragma unroll 5
                  for(kx = 0; kx < kernel_w; kx++) {
                    sum += input_p[kx]*(*kernel_p--);
                  }
                  input_p += input_w;
                }
              } else {
                for(ky = 0; ky < kernel_h; ky++) {
#pragma unroll 5
                  for(kx = 0; kx < kernel_w; kx++) {
                    sum += input_p[kx]*(*kernel_p++);
                  }
                  input_p += input_w;
                }
              }
              *output_p += sum;
            }
          }
        }
      }
    }

  } else { // not enough shared mem for kernels, simply stream them

    // convolution loop
    for(oo = oo_start; oo < oo_end; oo++) {
      for(ii = ii_start; ii < ii_end; ii++) {
        for(yy = yy_start; yy < yy_end; yy+=yy_step) {
          for(xx = xx_start; xx < xx_end; xx+=xx_step) {
            // Dot product in two dimensions... (between input image and the mask)
            float *input_p = input + ii*input_h*input_w + yy*stride_h*input_w + xx*stride_w;
            float *output_p = output + oo*output_h*output_w + yy*output_w + xx;
            float *kernel_p = kernel + ((oo % output_n) * input_n + (ii % input_n))*kernel_w*kernel_h + koffset;
            float sum = 0;
            if (swapkernel) {
              for(ky = 0; ky < kernel_h; ky++) {
#pragma unroll 5
                for(kx = 0; kx < kernel_w; kx++) {
                  sum += input_p[kx]*(*kernel_p--);
                }
                input_p += input_w;
              }
            } else {
              for(ky = 0; ky < kernel_h; ky++) {
#pragma unroll 5
                for(kx = 0; kx < kernel_w; kx++) {
                  sum += input_p[kx]*(*kernel_p++);
                }
                input_p += input_w;
              }
            }
            *output_p += sum;
          }
        }
      }
    }
  }
}

/*
 * Description:
 *   base conv2D routine with reversed stride: 3D input, 4D output, 3D kernel
 *   this is useful for computing gradients with respect to kernels, where:
 *   input=input, kernel=gradOutput, output=gradWeight
 *
 *   - all chunks of data should be contiguous
 *   - the swapkernel flag can be used to generate a conv2 instead of xcorr2
 */
__global__ void conv2genericrev(float *input, float *kernel, float *output,
                                int input_n, int input_h, int input_w,
                                int kernel_n, int kernel_h, int kernel_w,
                                float alpha, int stride_h, int stride_w)
{
  // output dimensions
  int output_h = input_h - (kernel_h - 1) * stride_h;
  int output_w = input_w - (kernel_w - 1) * stride_w;

  // this thread only processes one output, defined by the block Ids
  int kk = blockIdx.x;
  int ii = blockIdx.y;

  // batch id
  int batch = threadIdx.z;

  // kernel id
  int kid = threadIdx.x;
  int nkids = blockDim.x;

  // thread ID
  int tid = kid + batch*blockDim.x;
  int nthreads = blockDim.x * blockDim.z;

  // one thread only sees one output
  output = output + (kk * input_n + ii) * output_h*output_w;

  // put the output in shared memory
  __shared__ float shared_output[CUDA_SHARED_MEM_SIZE];

  // generate tid outputs in shared memory
  float *output_s = shared_output + tid*output_w*output_h;

  // convolution loop
  int xx, yy, kx, ky;
  yy = threadIdx.y;
  float *output_p = output_s + yy * output_w;
  for(xx=0; xx<output_w; xx++) {
    // Dot product in two dimensions... (between input image and kernel)
    float *input_p = input + (ii + batch*input_n)*input_h*input_w + yy*stride_h*input_w + xx*stride_w;
    float *kernel_p = kernel + (kk + batch*kernel_n)*kernel_w*kernel_h;
    float sum = 0;
    for(ky=0; ky<kernel_h; ky++) {
      for(kx=kid; kx<kernel_w; kx+=nkids) {
        sum += input_p[kx]*kernel_p[kx];
      }
      input_p += input_w;
      kernel_p += kernel_w;
    }
    *(output_p++) = sum;
  }
  __syncthreads();

  // reduce and write back
  if (yy == 0) {
    // reduce outputs
    for (int k=1; k<nthreads; k++) {
      for (int i=tid; i<output_w*output_h; i+=nthreads) {
        shared_output[i] += shared_output[k*output_h*output_w + i];
      }
    }
    __syncthreads();

    // add existing output, and write back
    for (int i=tid; i<output_w*output_h; i+=nthreads) {
      output[i] += alpha*shared_output[i];
    }
  }
}

// A helper macro for the common pattern of checking the input
// rows/columns for a small number of values, specializing the kernel
// template paremeters if rows/columns are equal and small, and
// otherwise just passing zero to the kernel.
#define FOR_KERNEL_SPECIALIZED_DIMENSION(ROWS, COLUMNS, KERNEL) \
  if ((ROWS) == (COLUMNS)) {                                    \
    switch ((ROWS)) {                                           \
      case 3: { KERNEL(3); break; }                             \
      case 4: { KERNEL(4); break; }                             \
      case 5: { KERNEL(5); break; }                             \
      case 6: { KERNEL(6); break; }                             \
      case 7: { KERNEL(7); break; }                             \
      case 8: { KERNEL(8); break; }                             \
      case 9: { KERNEL(9); break; }                             \
      case 10: { KERNEL(10); break; }                           \
      case 11: { KERNEL(11); break; }                           \
      case 12: { KERNEL(12); break; }                           \
      case 13: { KERNEL(13); break; }                           \
      default: { KERNEL(0); break; }                            \
    }                                                           \
  } else {                                                      \
    KERNEL(0);                                                  \
  }

/*
 * API-compatible with THRealTensor_conv2Dmv
 * 3D input, 4D kernel, 3D output
 * matrix vector product like: y <- Ax + beta*y
 */
THC_API void THCudaTensor_conv2Dmv(THCState *state, THCudaTensor *output, float beta, THCudaTensor *input,
                                   THCudaTensor *kernel, int64_t srow, int64_t scol, const char *type)
{
  THCAssertSameGPU(THCudaTensor_checkGPU(state, 3, output, input, kernel));
  int64_t nInputPlane, nInputRows, nInputCols;
  int64_t nKernelRows, nKernelCols;
  int64_t nOutputPlane, nOutputRows, nOutputCols;

  THArgCheck(kernel->nDimension == 4 , 4, "kernel: 4D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");
  THArgCheck(type[0] == 'v' || type[0] == 'f', 7, "type of convolution can 'v' or 'f'");
  THArgCheck(type[1] == 'c' || type[1] == 'x', 7, "type of convolution can 'x' or 'c'");

  input = THCudaTensor_newContiguous(state, input);
  kernel = THCudaTensor_newContiguous(state, kernel);

  nInputPlane = input->size[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];

  nKernelRows  = kernel->size[2];
  nKernelCols  = kernel->size[3];
  nOutputPlane = kernel->size[0];
  THArgCheck(kernel->size[1] == nInputPlane, 2, "invalid number of input planes");

  THArgCheck( (nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *type == 'f', 2,
              "conv2Dmv : Input image is smaller than kernel");

  if (*type == 'f') {
    // output dims
    nOutputRows = (nInputRows - 1) * srow + nKernelRows;
    nOutputCols = (nInputCols - 1) * scol + nKernelCols;

    // use temp buffer
    THCudaTensor *inputP = THCudaTensor_new(state);

    // create a zero-padded input
    int64_t nInputRowsPadded = (nOutputRows - 1) * srow + nKernelRows;
    int64_t nInputColsPadded = (nOutputCols - 1) * scol + nKernelCols;
    THCudaTensor_resize3d(state, inputP, nInputPlane, nInputRowsPadded, nInputColsPadded);
    THCudaTensor_zero(state, inputP);

    THCudaTensor *centered = THCudaTensor_new(state);
    THCudaTensor_narrow(state, centered, inputP, 2, nKernelCols-1, nInputCols);
    THCudaTensor_narrow(state, centered, NULL, 1, nKernelRows-1, nInputRows);
    THCudaTensor_copy(state, centered, input);
    THCudaTensor_free(state, centered);

    // remap input to newly created tensor
    THCudaTensor_free(state, input);
    input = inputP;
    nInputRows = nInputRowsPadded;
    nInputCols = nInputColsPadded;

  } else { // 'v'
    // output dims
    nOutputRows = (nInputRows - nKernelRows) / srow + 1;
    nOutputCols = (nInputCols - nKernelCols) / scol + 1;
  }

  ptrdiff_t nelem = THCudaTensor_nElement(state, output);
  THCudaTensor_resize3d(state, output, nOutputPlane, nOutputRows, nOutputCols);

  if (beta == 0 || nelem != THCudaTensor_nElement(state, output)) {
    THCudaTensor_zero(state, output);
  } else if (beta != 1) {
    THCudaTensor_mul(state, output, output, beta);
  }

  float *input_data = THCudaTensor_data(state, input);
  float *weight_data = THCudaTensor_data(state, kernel);
  float *output_data = THCudaTensor_data(state, output);

  // cuda blocks & threads:
  int yblocks = (int)(16L / nOutputPlane);
  yblocks = yblocks < 1 ? 1 : yblocks;
  dim3 blocks(nOutputPlane,yblocks);
  dim3 threads(32,8);

  // convolution: xcorr2 or conv2
  if (type[1] == 'x') {
#define X_CONV_KERNEL(dim)                                              \
    conv2generic <false, (dim), (dim)> <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> ( \
        input_data, weight_data, output_data,                           \
        nInputPlane, nInputRows, nInputCols,                            \
        nOutputPlane*nInputPlane, nKernelRows, nKernelCols,             \
        srow, scol);

    FOR_KERNEL_SPECIALIZED_DIMENSION(nKernelRows, nKernelCols, X_CONV_KERNEL);
#undef X_CONV_KERNEL
  } else { // 'c'
#define C_CONV_KERNEL(dim)                                              \
    conv2generic <true, (dim), (dim)> <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (           \
        input_data, weight_data, output_data,                           \
        nInputPlane, nInputRows, nInputCols,                            \
        nOutputPlane*nInputPlane, nKernelRows, nKernelCols,             \
        srow, scol);

    FOR_KERNEL_SPECIALIZED_DIMENSION(nKernelRows, nKernelCols, C_CONV_KERNEL);
#undef C_CONV_KERNEL
  }

  // clean
  if (*type != 'f') THCudaTensor_free(state, input);
  THCudaTensor_free(state, kernel);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in conv2Dmv: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
}

/*
 * API-compatible with THRealTensor_conv2Dmm
 * 4D input, 4D kernel, 4D output
 * matrix vector product like: y <- Ax + beta*y
 */
THC_API void THCudaTensor_conv2Dmm(THCState *state, THCudaTensor *output, float beta, THCudaTensor *input,
                                   THCudaTensor *kernel, int64_t srow, int64_t scol, const char *type)
{
  THCAssertSameGPU(THCudaTensor_checkGPU(state, 3, output, input, kernel));
  int64_t nbatch, nInputPlane, nInputRows, nInputCols;
  int64_t nKernelRows, nKernelCols;
  int64_t nOutputPlane, nOutputRows, nOutputCols;

  THArgCheck(kernel->nDimension == 4 , 4, "kernel: 4D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");
  THArgCheck(type[0] == 'v' || type[0] == 'f', 7, "type of convolution can 'v' or 'f'");
  THArgCheck(type[1] == 'c' || type[1] == 'x', 7, "type of convolution can 'x' or 'c'");

  input = THCudaTensor_newContiguous(state, input);
  kernel = THCudaTensor_newContiguous(state, kernel);

  nbatch      = input->size[0];
  nInputPlane = input->size[1];
  nInputRows  = input->size[2];
  nInputCols  = input->size[3];

  nKernelRows  = kernel->size[2];
  nKernelCols  = kernel->size[3];
  nOutputPlane = kernel->size[0];
  THArgCheck(kernel->size[1] == nInputPlane, 2, "invalid number of input planes");

  THArgCheck( (nInputRows >= nKernelRows && nInputCols >= nKernelCols) || *type == 'f', 2,
              "conv2Dmm : Input image is smaller than kernel");

  if (*type == 'f') {
    // output dims
    nOutputRows = (nInputRows - 1) * srow + nKernelRows;
    nOutputCols = (nInputCols - 1) * scol + nKernelCols;

    // use temp buffer
    THCudaTensor *inputP = THCudaTensor_new(state);

    // create a zero-padded input
    int64_t nInputRowsPadded = (nOutputRows - 1) * srow + nKernelRows;
    int64_t nInputColsPadded = (nOutputCols - 1) * scol + nKernelCols;
    THCudaTensor_resize4d(state, inputP, nbatch, nInputPlane, nInputRowsPadded, nInputColsPadded);
    THCudaTensor_zero(state, inputP);

    THCudaTensor *centered = THCudaTensor_new(state);
    THCudaTensor_narrow(state, centered, inputP, 3, nKernelCols-1, nInputCols);
    THCudaTensor_narrow(state, centered, NULL, 2, nKernelRows-1, nInputRows);
    THCudaTensor_copy(state, centered, input);
    THCudaTensor_free(state, centered);

    // remap input to newly created tensor
    THCudaTensor_free(state, input);
    input = inputP;
    nInputRows = nInputRowsPadded;
    nInputCols = nInputColsPadded;

  } else { // 'v'
    // output dims
    nOutputRows = (nInputRows - nKernelRows) / srow + 1;
    nOutputCols = (nInputCols - nKernelCols) / scol + 1;
  }

  ptrdiff_t nelem = THCudaTensor_nElement(state, output);
  THCudaTensor_resize4d(state, output, nbatch, nOutputPlane, nOutputRows, nOutputCols);

  if (beta == 0 || nelem != THCudaTensor_nElement(state, output)) {
    THCudaTensor_zero(state, output);
  } else if (beta != 1) {
    THCudaTensor_mul(state, output, output, beta);
  }

  float *input_data = THCudaTensor_data(state, input);
  float *weight_data = THCudaTensor_data(state, kernel);
  float *output_data = THCudaTensor_data(state, output);

  // cuda blocks & threads:
  int yblocks = (int)(16L / nOutputPlane);
  yblocks = yblocks < 1 ? 1 : yblocks;
  dim3 blocks(nOutputPlane*nbatch,yblocks);
  dim3 threads(32,8);

  // convolution: xcorr2 or conv2
  if (type[1] == 'x') {
#define X_CONV_KERNEL(dim)                                              \
    conv2generic <false, (dim), (dim)> <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> ( \
      input_data, weight_data, output_data,                             \
      nInputPlane, nInputRows, nInputCols,                              \
      nOutputPlane*nInputPlane, nKernelRows, nKernelCols,               \
      srow, scol);

    FOR_KERNEL_SPECIALIZED_DIMENSION(nKernelCols, nKernelRows, X_CONV_KERNEL);
#undef X_CONV_KERNEL
  } else { // 'c'
#define C_CONV_KERNEL(dim)                                              \
    conv2generic <true, (dim), (dim)> <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> ( \
      input_data, weight_data, output_data,                             \
      nInputPlane, nInputRows, nInputCols,                              \
      nOutputPlane*nInputPlane, nKernelRows, nKernelCols,               \
      srow, scol);                                                      \

    FOR_KERNEL_SPECIALIZED_DIMENSION(nKernelCols, nKernelRows, C_CONV_KERNEL);
#undef C_CONV_KERNEL
  }

  // clean
  if (*type != 'f') THCudaTensor_free(state, input);
  THCudaTensor_free(state, kernel);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("error in conv2Dmm: %s\n", cudaGetErrorString(err));
    printf("requested grid size: %dx%dx%d, max allowed: %dx%dx%d\n",
           blocks.x, blocks.y, blocks.z,
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("requested block size: %dx%dx%d, max allowed: %dx%dx%d\n",
           threads.x, threads.y, threads.z,
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    THError("aborting");
  }
}

/*
 * API-compatible with THRealTensor_conv2DRevger
 * 3D input, 3D kernel, 4D output
 * like rank1 update
 * A <- xx' + beta*A
 * for sr,sc=1 this is equivalent to xcorr2Dger, but otherwise it is useful for
 * calculating derivatives wrt a kernel that is applied with stride sr,sc != 1
 */
THC_API void THCudaTensor_conv2DRevger(THCState *state, THCudaTensor *output, float beta, float alpha,
                                       THCudaTensor *input, THCudaTensor *kernel,
                                       int64_t srow, int64_t scol)
{
  THCAssertSameGPU(THCudaTensor_checkGPU(state, 3, output, input, kernel));
  int64_t nInputPlane, nInputRows, nInputCols;
  int64_t nKernelPlane, nKernelRows, nKernelCols;
  int64_t nOutputRows, nOutputCols;

  THArgCheck(input->nDimension == 3 , 3, "input: 3D Tensor expected");
  THArgCheck(kernel->nDimension == 3 , 4, "kernel: 3D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");

  input = THCudaTensor_newContiguous(state, input);
  kernel = THCudaTensor_newContiguous(state, kernel);

  nInputPlane = input->size[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];

  nKernelPlane = kernel->size[0];
  nKernelRows = kernel->size[1];
  nKernelCols = kernel->size[2];

  THArgCheck(nInputRows >= nKernelRows && nInputCols >= nKernelCols , 2,
             "conv2DRevger : Input image is smaller than kernel");

  nOutputRows = nInputRows - (nKernelRows - 1) * srow;
  nOutputCols = nInputCols - (nKernelCols - 1) * scol;

  ptrdiff_t nelem = THCudaTensor_nElement(state, output);
  THCudaTensor_resize4d(state, output, nKernelPlane, nInputPlane, nOutputRows, nOutputCols);

  if (nelem == 0 || beta == 0 || nelem != THCudaTensor_nElement(state, output)) {
    THCudaTensor_zero(state, output);
  } else if (beta != 1) {
    THCudaTensor_mul(state, output, output, beta);
  }

  float *input_data = THCudaTensor_data(state, input);
  float *kernel_data = THCudaTensor_data(state, kernel);
  float *output_data = THCudaTensor_data(state, output);

  // auto compute nb of blocks and threads
  dim3 blocks(nKernelPlane, nInputPlane);
  dim3 threads(128/nOutputRows, nOutputRows);

  // compute rev conv
  conv2genericrev <<<blocks, threads, 0, THCState_getCurrentStream(state)>>>(
    input_data, kernel_data, output_data,
    nInputPlane, nInputRows, nInputCols,
    nKernelPlane, nKernelRows, nKernelCols,
    alpha, srow, scol);

  // clean
  THCudaTensor_free(state, input);
  THCudaTensor_free(state, kernel);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in conv2DRevger: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
}

/*
 * API-compatible with THRealTensor_conv2DRevgerm
 * 4D input, 4D kernel, 4D output
 * conv2DRevgerm is doing the same thing as conv2DRevger, but with batch inputs
 */
THC_API void THCudaTensor_conv2DRevgerm(THCState *state, THCudaTensor *output, float beta, float alpha,
                                        THCudaTensor *input, THCudaTensor *kernel,
                                        int64_t srow, int64_t scol)
{
  int64_t nInputPlane, nInputRows, nInputCols;
  int64_t nKernelPlane, nKernelRows, nKernelCols;
  int64_t nOutputRows, nOutputCols;
  int64_t nbatch;

  THArgCheck(input->nDimension == 4 , 3, "input: 3D Tensor expected");
  THArgCheck(kernel->nDimension == 4 , 4, "kernel: 3D Tensor expected");
  THArgCheck(srow >= 1, 5, "Stride should be a positive integer");
  THArgCheck(scol >= 1, 6, "Stride should be a positive integer");

  input = THCudaTensor_newContiguous(state, input);
  kernel = THCudaTensor_newContiguous(state, kernel);

  nbatch      = input->size[0];
  nInputPlane = input->size[1];
  nInputRows  = input->size[2];
  nInputCols  = input->size[3];

  nKernelPlane = kernel->size[1];
  nKernelRows = kernel->size[2];
  nKernelCols = kernel->size[3];

  THArgCheck(nInputRows >= nKernelRows && nInputCols >= nKernelCols , 2,
             "conv2DRevger : Input image is smaller than kernel");

  nOutputRows = nInputRows - (nKernelRows - 1) * srow;
  nOutputCols = nInputCols - (nKernelCols - 1) * scol;

  ptrdiff_t nelem = THCudaTensor_nElement(state, output);
  THCudaTensor_resize4d(state, output, nKernelPlane, nInputPlane, nOutputRows, nOutputCols);

  if (nelem == 0 || beta == 0 || nelem != THCudaTensor_nElement(state, output)) {
    THCudaTensor_zero(state, output);
  } else if (beta != 1) {
    THCudaTensor_mul(state, output, output, beta);
  }

  float *input_data = THCudaTensor_data(state, input);
  float *kernel_data = THCudaTensor_data(state, kernel);
  float *output_data = THCudaTensor_data(state, output);

  // kernel is called multiple times
  // (the arbitrary split below is just here to make sure we don't go over 256 threads)
  for (int sl=0; sl<nbatch; sl+=6) {
    // auto compute nb of blocks and threads
    dim3 blocks(nKernelPlane, nInputPlane);
    int subbatch = 6;
    if (sl+subbatch > nbatch) subbatch = nbatch - sl;
    int cst = 256 / (subbatch * nOutputRows);
    dim3 threads(cst, nOutputRows, subbatch);

    // compute rev conv
    conv2genericrev <<<blocks, threads, 0, THCState_getCurrentStream(state)>>>(
      input_data + input->stride[0]*sl,
      kernel_data + kernel->stride[0]*sl,
      output_data,
      nInputPlane, nInputRows, nInputCols,
      nKernelPlane, nKernelRows, nKernelCols,
      alpha, srow, scol);
  }

  // clean
  THCudaTensor_free(state, input);
  THCudaTensor_free(state, kernel);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in conv2DRevger: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
}


///////////////////////////////////
///// ConvolutionMap
/*
 * Description:
 *   base conv2D routine: 3D input, 3D output, 4D kernel
 *
 *   - all chunks of data should be contiguous
 *   - the swapkernel flag can be used to generate a conv2 instead of xcorr2
 *   - the templated kernel size is useful to generate code that's 2x faster
 *     but can be set to 0 to allow arbitrary kernel sizes
 *   ---- the table should have the first dim with the outputs, each output
 *   ---- should have a fanin set of inputs contiguously
 */
template <bool swapkernel, int T_kernel_h, int T_kernel_w>
  __global__ void conv2mapgeneric(float *input, float *kernel, float *output,
                                  int input_n, int input_h, int input_w,
                                  int kernel_n, int kernel_h, int kernel_w,
                                  int stride_w, int stride_h,
                                  float *table, int fanin)
{
  // output dimensions
  int output_h = (input_h - kernel_h) / stride_h + 1;
  int output_w = (input_w - kernel_w) / stride_w + 1;

  // xcorr or conv
  int koffset = swapkernel ? kernel_w*kernel_h-1 : 0;

  // nb outputs
  // int output_n = kernel_n / fanin;

  // generate offsets according to block/thread ids
  int xx_start = threadIdx.x;
  int xx_end = output_w;
  int xx_step = blockDim.x;

  int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
  int yy_end = output_h;
  int yy_step = blockDim.y*gridDim.y;

  int oo_start = blockIdx.x;
  int oo_end = oo_start+1;

  int table_start = blockIdx.x * (fanin * 2);
  int table_end = table_start + (fanin * 2);

  // nb threads, unique thread id
  int tid = blockDim.x*blockDim.y*threadIdx.z
    + blockDim.x * threadIdx.y + threadIdx.x;
  int nthreads = blockDim.x * blockDim.y * blockDim.z;

  // iterators
  int oo, ii, xx, yy, kx, ky, kk;

  // do the kernels fit in shared mem ?
  if (kernel_w*kernel_h*kernel_n <= CUDA_SHARED_MEM_SIZE) {
    // put the kernel in shared memory
    __shared__ float shared_kernel[CUDA_SHARED_MEM_SIZE];

    // first thread of each block does the copy
    for (kk = tid; kk < kernel_w*kernel_h*kernel_n; kk += nthreads) {
      shared_kernel[kk] = kernel[kk];
    }
    __syncthreads();

    // templated kernel size
    if ((T_kernel_w > 0) && (T_kernel_h > 0)) {
      // unrolled convolution loop
      for(oo = oo_start; oo < oo_end; oo++) {
        for (ii = table_start; ii < table_end; ii = ii + 2) {
          for(yy = yy_start; yy < yy_end; yy+=yy_step) {
            for(xx = xx_start; xx < xx_end; xx+=xx_step) {
              // Dot product in two dimensions... (between input image and the mask)
              float *input_p = input + ((int64_t)table[ii]-1)*input_h*input_w
                + yy*stride_h*input_w + xx*stride_w;
              float *output_p = output + oo*output_h*output_w + yy*output_w + xx;
              float *kernel_p = shared_kernel
                + ((int64_t)table[ii + 1]-1) *kernel_w*kernel_h + koffset;
              float sum = 0;
              if (swapkernel) {
#pragma unroll
                for(ky = 0; ky < T_kernel_h; ky++) {
#pragma unroll
                  for(kx = 0; kx < T_kernel_w; kx++) {
                    sum += input_p[kx]*(*kernel_p--);
                  }
                  input_p += input_w;
                }
              } else {
#pragma unroll
                for(ky = 0; ky < T_kernel_h; ky++) {
#pragma unroll
                  for(kx = 0; kx < T_kernel_w; kx++) {
                    sum += input_p[kx]*(*kernel_p++);
                  }
                  input_p += input_w;
                }
              }
              *output_p += sum;
            }
          }
        }
      }
    } else {
      // default convolution loop
      for(oo = oo_start; oo < oo_end; oo++) {
        for (ii = table_start; ii < table_end; ii++) {
          for(yy = yy_start; yy < yy_end; yy+=yy_step) {
            for(xx = xx_start; xx < xx_end; xx+=xx_step) {
              // Dot product in two dims (between input image and the mask)
              float *input_p = input + ((int64_t)table[ii]-1)*input_h*input_w
                + yy*stride_h*input_w + xx*stride_w;
              float *output_p = output + oo*output_h*output_w + yy*output_w
                + xx;
              float *kernel_p = shared_kernel
                + ((int64_t)table[ii + 1]-1) *kernel_w*kernel_h + koffset;
              float sum = 0;
              if (swapkernel) {
                for(ky = 0; ky < kernel_h; ky++) {
#pragma unroll 5
                  for(kx = 0; kx < kernel_w; kx++) {
                    sum += input_p[kx]*(*kernel_p--);
                  }
                  input_p += input_w;
                }
              } else {
                for(ky = 0; ky < kernel_h; ky++) {
#pragma unroll 5
                  for(kx = 0; kx < kernel_w; kx++) {
                    sum += input_p[kx]*(*kernel_p++);
                  }
                  input_p += input_w;
                }
              }
              *output_p += sum;
            }
          }
        }
      }
    }

  } else { // not enough shared mem for kernels, simply stream them

    // convolution loop
    for(oo = oo_start; oo < oo_end; oo++) {
      for (ii = table_start; ii < table_end; ii = ii + 2) {
        for(yy = yy_start; yy < yy_end; yy+=yy_step) {
          for(xx = xx_start; xx < xx_end; xx+=xx_step) {
            // Dot product in two dimensions... (between input image and the mask)
            float *input_p = input + ((int64_t)table[ii]-1)*input_h*input_w
              + yy*stride_h*input_w + xx*stride_w;
            float *output_p = output + oo*output_h*output_w + yy*output_w + xx;
            float *kernel_p = kernel + ((int64_t)table[ii + 1]-1) *kernel_w*kernel_h + koffset;
            float sum = 0;
            if (swapkernel) {
              for(ky = 0; ky < kernel_h; ky++) {
#pragma unroll 5
                for(kx = 0; kx < kernel_w; kx++) {
                  sum += input_p[kx]*(*kernel_p--);
                }
                input_p += input_w;
              }
            } else {
              for(ky = 0; ky < kernel_h; ky++) {
#pragma unroll 5
                for(kx = 0; kx < kernel_w; kx++) {
                  sum += input_p[kx]*(*kernel_p++);
                }
                input_p += input_w;
              }
            }
            *output_p += sum;
          }
        }
      }
    }
  }
}


/*
 * API-compatible with THRealTensor_conv2Dmv
 * 3D input, 4D kernel, 3D output
 * matrix vector product like: y <- Ax + beta*y
 */
THC_API void THCudaTensor_conv2Dmap(THCState *state, THCudaTensor *output, THCudaTensor *input,
                                    THCudaTensor *kernel, int64_t stride_x, int64_t stride_y,
                                    THCudaTensor *table, int64_t fanin)
{
  THCAssertSameGPU(THCudaTensor_checkGPU(state, 4, output, input, kernel, table));
  int64_t nInputPlane, nInputRows, nInputCols;
  int64_t nKernelRows, nKernelCols;
  int64_t nOutputPlane, nOutputRows, nOutputCols;

  THArgCheck(kernel->nDimension == 3 , 4, "kernel: 3D Tensor expected");
  THArgCheck(stride_x >= 1, 5, "Stride should be a positive integer");
  THArgCheck(stride_y >= 1, 6, "Stride should be a positive integer");

  input = THCudaTensor_newContiguous(state, input);
  kernel = THCudaTensor_newContiguous(state, kernel);
  table = THCudaTensor_newContiguous(state, table);

  nInputPlane = input->size[0];
  nInputRows  = input->size[1];
  nInputCols  = input->size[2];

  nKernelRows  = kernel->size[1];
  nKernelCols  = kernel->size[2];
  nOutputPlane = kernel->size[0] / fanin;
  // THArgCheck(kernel->size[1] == nInputPlane, 2, "invalid number of input planes");

  THArgCheck( (nInputRows >= nKernelRows && nInputCols >= nKernelCols), 2,
              "conv2Dmap : Input image is smaller than kernel");

  // output dims
  nOutputRows = (nInputRows - nKernelRows) / stride_y + 1;
  nOutputCols = (nInputCols - nKernelCols) / stride_x + 1;

  // ptrdiff_t nelem = THCudaTensor_nElement(state, output);
  THCudaTensor_resize3d(state, output, nOutputPlane, nOutputRows, nOutputCols);

  float *input_data = THCudaTensor_data(state, input);
  float *kernel_data = THCudaTensor_data(state, kernel);
  float *output_data = THCudaTensor_data(state, output);
  float *table_data = THCudaTensor_data(state, table);

  // set the number of blocks and threads
  int nthreads_x = 32;
  int nthreads_y = 8;
  int block_height = (int)(16L / nOutputPlane);
  if (block_height < 1)
    block_height = 1;
  dim3 blocks(nOutputPlane,block_height);
  dim3 threads(nthreads_x,nthreads_y);

#define GENERIC_MAP_KERNEL(dim)                                         \
  conv2mapgeneric <false, (dim), (dim)> <<<blocks, threads, 0, THCState_getCurrentStream(state)>>> ( \
      input_data, kernel_data, output_data, nInputPlane, nInputRows,    \
      nInputCols, nOutputPlane*fanin, nKernelRows, nKernelCols,         \
      stride_x, stride_y, table_data, fanin);

  FOR_KERNEL_SPECIALIZED_DIMENSION(nKernelCols, nKernelRows, GENERIC_MAP_KERNEL);
#undef GENERIC_MAP_KERNEL
  // clean
  THCudaTensor_free(state, input);
  THCudaTensor_free(state, kernel);
  THCudaTensor_free(state, table);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in conv2Dmap: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
}

#undef FOR_KERNEL_SPECIALIZED_DIMENSION
