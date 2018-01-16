#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"

#define CUDA_MAX_THREADS 1024   // this is safe, in reality 256 is our limit

/*
 * Description:
 *    this function subsamples an input 3D tensor along dimensions 1 and 2
 *    3D input, 3D output, 1D weight, 1D bias
 */
 template <typename Dtype, typename Acctype>
__global__ void subsample(Dtype *input, Dtype *output, Dtype *weight, Dtype *bias,
                          int input_n, int input_h, int input_w,
                          int kH, int kW, int dH, int dW)
{
  // iterators
  int xx, yy;

  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;

  // compute offsets based on thread/block ID
  int o = blockIdx.x;
  int i = o;
  int k = blockIdx.x % input_n;

  int xx_start = threadIdx.x;
  int xx_end = output_w;
  int xx_step = blockDim.x;

  int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
  int yy_end = output_h;
  int yy_step = blockDim.y*gridDim.y;

  // select input/output plane
  output = output + o*output_w*output_h;
  input = input + i*input_w*input_h;

  // Get the good mask for (k,i) (k out, i in)
  Dtype the_weight = weight[k];

  // Initialize to the bias
  Dtype the_bias = bias[k];

  // For all output pixels...
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {
    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
      // Compute the mean of the input image...
      Dtype *ptr_input = input + yy*dH*input_w + xx*dW;
      Dtype *ptr_output = output + yy*output_w + xx;
      Acctype sum = 0;
      int kx, ky;
      for(ky = 0; ky < kH; ky++) {
        for(kx = 0; kx < kW; kx++)
          sum += ptr_input[kx];
        ptr_input += input_w; // next input line
      }
      // Update output
      *ptr_output = ScalarConvert<Acctype, Dtype>::to(the_weight*sum + the_bias);
    }
  }
}

/*
 * Description:
 *    this function computes the gradWeight from input and gradOutput
 */
 template <typename Dtype, typename Acctype>
__global__ void subgradweight(Dtype *input, Dtype *gradOutput, Dtype *gradWeight, Dtype *gradBias,
                              int input_n, int input_h, int input_w,
                              int kH, int kW, int dH, int dW,
                              float scale)
{
  // iterators
  int xx, yy;

  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;

  // compute offsets based on thread/block ID
  int o = blockIdx.x;
  int i = o;
  int k = blockIdx.x % input_n;

  int xx_start = threadIdx.x;
  int xx_end = output_w;
  int xx_step = blockDim.x;

  int yy_start = threadIdx.y;
  int yy_end = output_h;
  int yy_step = blockDim.y;

  // select input/output plane
  gradOutput = gradOutput + o*output_w*output_h;
  input = input + i*input_w*input_h;

  // thread ID
  int tid = blockDim.x*threadIdx.y + threadIdx.x;

  // create array to hold partial sums
  __shared__ Acctype sums[CUDA_MAX_THREADS];
  sums[tid] = 0;

  // compute partial sums
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {
    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
      Dtype *ptr_input = input + yy*dH*input_w + xx*dW;
      Dtype *ptr_gradOutput = gradOutput + yy*output_w + xx;
      Dtype z = *ptr_gradOutput;
      int64_t kx, ky;
      for(ky = 0; ky < kH; ky++) {
        for(kx = 0; kx < kW; kx++) {
          sums[tid] += z * ptr_input[kx];
        }
        ptr_input += input_w;
      }
    }
  }
  __syncthreads();

  // reduce: accumulate all partial sums to produce final gradWeight
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    Acctype scaledSums = Acctype(0);
    for(int i = 0; i < blockDim.x*blockDim.y; i++) {
      scaledSums += scale*sums[i];
    }
    gradWeight[k] += ScalarConvert<Acctype, Dtype>::to(scaledSums);
  }
  __syncthreads();

  // compute gradBias
  sums[tid] = 0;
  for (int i=tid; i<output_w*output_h; i+=(blockDim.x*blockDim.y)) {
    sums[tid] += gradOutput[i];
  }
  __syncthreads();

  // reduce gradBias
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    Acctype scaledSums = Acctype(0);
    for (int i=0; i<(blockDim.x*blockDim.y); i++) {
      scaledSums += scale*sums[i];
    }
    gradBias[k] += ScalarConvert<Acctype, Dtype>::to(scaledSums);
  }
}

/*
 * Description:
 *    this function computes the gradInput from weight and gradOutput
 */
 template <typename Dtype>
__global__ void subgradinput(Dtype *gradInput, Dtype *gradOutput, Dtype *weight,
                             int input_n, int input_h, int input_w,
                             int kH, int kW, int dH, int dW)
{
  // iterators
  int xx, yy;

  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;

  // compute offsets based on thread/block ID
  int o = blockIdx.x;
  int i = o;
  int k = blockIdx.x % input_n;

  int xx_start = threadIdx.x;
  int xx_end = output_w;
  int xx_step = blockDim.x;

  int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
  int yy_end = output_h;
  int yy_step = blockDim.y*gridDim.y;

  // select input/output plane
  gradOutput = gradOutput + o*output_w*output_h;
  gradInput = gradInput + i*input_w*input_h;

  // get weight
  Dtype the_weight = weight[k];

  // compute gradInput
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {
    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
      Dtype *ptr_gradInput = gradInput + yy*dH*input_w + xx*dW;
      Dtype *ptr_gradOutput = gradOutput + yy*output_w + xx;
      Dtype z = *ptr_gradOutput * the_weight;
      int kx, ky;
      for(ky = 0; ky < kH; ky++) {
        for(kx = 0; kx < kW; kx++) {
          // FIXME: should this be done at accreal precision?
          ptr_gradInput[kx] += z;
        }
        ptr_gradInput += input_w;
      }
    }
  }
}

/*
 * Description:
 *    this function computes the gradInput from weight and gradOutput
 */
 template <typename Dtype>
__global__ void subgradinputAtomic(Dtype *gradInput, Dtype *gradOutput, Dtype *weight,
                                   int input_n, int input_h, int input_w,
                                   int kH, int kW, int dH, int dW)
{
  // iterators
  int xx, yy;

  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;

  // compute offsets based on thread/block ID
  int o = blockIdx.x;
  int i = o;
  int k = blockIdx.x % input_n;

  int xx_start = threadIdx.x;
  int xx_end = output_w;
  int xx_step = blockDim.x;

  int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
  int yy_end = output_h;
  int yy_step = blockDim.y*gridDim.y;

  // select input/output plane
  gradOutput = gradOutput + o*output_w*output_h;
  gradInput = gradInput + i*input_w*input_h;

  // get weight
  Dtype the_weight = weight[k];

  // compute gradInput
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {
    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
      Dtype *ptr_gradInput = gradInput + yy*dH*input_w + xx*dW;
      Dtype *ptr_gradOutput = gradOutput + yy*output_w + xx;
      Dtype z = *ptr_gradOutput * the_weight;
      int kx, ky;
      for(ky = 0; ky < kH; ky++) {
        for(kx = 0; kx < kW; kx++) {
          // FIXME: should this be done at accreal precision?
          atomicAdd(&(ptr_gradInput[kx]), z);
        }
        ptr_gradInput += input_w;
      }
    }
  }
}


#include "generic/SpatialSubSampling.cu"
#include "THCGenerateFloatTypes.h"

#undef CUDA_MAX_THREADS
