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
__global__ void vsubsample(Dtype *input, Dtype *output, Dtype *weight, Dtype *bias,
                           int input_n, int input_t, int input_h, int input_w,
                           int kT, int kH, int kW, int dT, int dH, int dW)
{
  // iterators
  int xx, yy, zz;

  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;
  int output_t = (input_t - kT) / dT + 1;

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

  int zz_start = blockDim.z*blockIdx.z + threadIdx.z;
  int zz_end = output_t;
  int zz_step = blockDim.z*gridDim.z;

  // select input/output plane
  output = output + o*output_w*output_h*output_t;
  input = input + i*input_w*input_h*input_t;

  // Get the good mask for (k,i) (k out, i in)
  Dtype the_weight = weight[k];

  // Initialize to the bias
  Dtype the_bias = bias[k];

  // For all output pixels...
  for(zz = zz_start; zz < zz_end; zz+=zz_step) {
    for(yy = yy_start; yy < yy_end; yy+=yy_step) {
      for(xx = xx_start; xx < xx_end; xx+=xx_step) {
        // Compute the mean of the input image...
        Dtype *ptr_input = input + zz*dT*input_h*input_w + yy*dH*input_w + xx*dW;
        Dtype *ptr_output = output + zz*output_h*output_w + yy*output_w + xx;
        Acctype sum = 0;
        int kx, ky, kz;
        for(kz = 0; kz < kT; kz++) {
          for(ky = 0; ky < kH; ky++) {
            for(kx = 0; kx < kW; kx++)
              sum += ptr_input[ky*input_w+kx];
          }
          ptr_input += input_h*input_w; // next input plane
        }
        // Update output
        *ptr_output = ScalarConvert<Acctype, Dtype>::to(the_weight*sum + the_bias);
      }
    }
  }
}

/*
 * Description:
 *    this function computes the gradWeight from input and gradOutput
 */
 template <typename Dtype, typename Acctype>
__global__ void vsubgradweight(Dtype *input, Dtype *gradOutput, Dtype *gradWeight, Dtype *gradBias,
                               int input_n, int input_t, int input_h, int input_w,
                               int kT, int kH, int kW, int dT, int dH, int dW,
                               float scale)
{
  // iterators
  int xx, yy, zz;

  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;
  int output_t = (input_t - kT) / dT + 1;

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

  int zz_start = threadIdx.z;
  int zz_end = output_t;
  int zz_step = blockDim.z;

  // select input/output plane
  gradOutput = gradOutput + o*output_w*output_h*output_t;
  input = input + i*input_w*input_h*input_t;

  // thread ID
  int tid = blockDim.y*blockDim.x*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;

  // create array to hold partial sums
  __shared__ Acctype sums[CUDA_MAX_THREADS];
  sums[tid] = 0;

  // compute partial sums
  for(zz = zz_start; zz < zz_end; zz+=zz_step) {
    for(yy = yy_start; yy < yy_end; yy+=yy_step) {
      for(xx = xx_start; xx < xx_end; xx+=xx_step) {
        Dtype *ptr_input = input + zz*dT*input_h*input_w + yy*dH*input_w + xx*dW;
        Dtype *ptr_gradOutput = gradOutput + zz*output_h*output_w + yy*output_w + xx;
        Dtype z = *ptr_gradOutput;
        long kx, ky, kz;
        for(kz = 0; kz < kT; kz++) {
          for(ky = 0; ky < kH; ky++) {
            for(kx = 0; kx < kW; kx++) {
              sums[tid] += z * ptr_input[ky*input_w+kx];
            }
          }
          ptr_input += input_h*input_w;
        }
      }
    }
  }
  __syncthreads();

  // reduce: accumulate all partial sums to produce final gradWeight
  if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
    Acctype scaledSums = Acctype(0);
    for(int i = 0; i < blockDim.x*blockDim.y*blockDim.z; i++) {
      scaledSums += scale*sums[i];
    }
    gradWeight[k] += ScalarConvert<Acctype, Dtype>::to(scaledSums);
  }
  __syncthreads();

  // compute gradBias
  sums[tid] = 0;
  for (int i=tid; i<output_w*output_h*output_t; i+=(blockDim.x*blockDim.y*blockDim.z)) {
    sums[tid] += gradOutput[i];
  }
  __syncthreads();

  // reduce gradBias
  if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
    Acctype scaledSums = Acctype(0);
    for (int i=0; i<(blockDim.x*blockDim.y*blockDim.z); i++) {
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
__global__ void vsubgradinput(Dtype *gradInput, Dtype *gradOutput, Dtype *weight,
                              int input_n, int input_t, int input_h, int input_w,
                              int kT, int kH, int kW, int dT, int dH, int dW)
{
  // iterators
  int xx, yy, zz;

  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;
  int output_t = (input_t - kT) / dT + 1;

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

  int zz_start = blockDim.z*blockIdx.z + threadIdx.z;
  int zz_end = output_t;
  int zz_step = blockDim.z*gridDim.z;

  // select input/output plane
  gradOutput = gradOutput + o*output_w*output_h*output_t;
  gradInput = gradInput + i*input_w*input_h*input_t;

  // get weight
  Dtype the_weight = weight[k];

  // compute gradInput
  for(zz = zz_start; zz < zz_end; zz+=zz_step) {
    for(yy = yy_start; yy < yy_end; yy+=yy_step) {
      for(xx = xx_start; xx < xx_end; xx+=xx_step) {
        Dtype *ptr_gradInput = gradInput + zz*dT*input_h*input_w + yy*dH*input_w + xx*dW;
        Dtype *ptr_gradOutput = gradOutput + zz*output_h*output_w + yy*output_w + xx;
        Dtype z = *ptr_gradOutput * the_weight;
        int kx, ky, kz;
        for(kz = 0; kz < kT; kz++) {
          for(ky = 0; ky < kH; ky++) {
            for(kx = 0; kx < kW; kx++) {
              // FIXME: should this be done at accreal precision?
              ptr_gradInput[ky*input_w+kx] += z;
            }
          }
          ptr_gradInput += input_h*input_w;
        }
      }
    }
  }
}

/*
 * Description:
 *    this function computes the gradInput from weight and gradOutput
 */
 template <typename Dtype>
__global__ void vsubgradinputAtomic(Dtype *gradInput, Dtype *gradOutput, Dtype *weight,
                                    int input_n, int input_t, int input_h, int input_w,
                                    int kT, int kH, int kW, int dT, int dH, int dW)
{
  // iterators
  int xx, yy, zz;

  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;
  int output_t = (input_t - kT) / dT + 1;

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

  int zz_start = blockDim.z*blockIdx.z + threadIdx.z;
  int zz_end = output_t;
  int zz_step = blockDim.z*gridDim.z;

  // select input/output plane
  gradOutput = gradOutput + o*output_w*output_h*output_t;
  gradInput = gradInput + i*input_w*input_h*input_t;

  // get weight
  Dtype the_weight = weight[k];

  // compute gradInput
  for(zz = zz_start; zz < zz_end; zz+=zz_step) {
    for(yy = yy_start; yy < yy_end; yy+=yy_step) {
      for(xx = xx_start; xx < xx_end; xx+=xx_step) {
        Dtype *ptr_gradInput = gradInput + zz*dT*input_h*input_w + yy*dH*input_w + xx*dW;
        Dtype *ptr_gradOutput = gradOutput + zz*output_h*output_w + yy*output_w + xx;
        Dtype z = *ptr_gradOutput * the_weight;
        int kx, ky, kz;
        for(kz = 0; kz < kT; kz++) {
          for(ky = 0; ky < kH; ky++) {
            for(kx = 0; kx < kW; kx++) {
              // FIXME: should this be done at accreal precision?
              atomicAdd(&(ptr_gradInput[ky*input_w+kx]), z);
            }
          }
          ptr_gradInput += input_h*input_w;
        }
      }
    }
  }
}


#include "generic/VolumetricSubSampling.cu"
#include "THCGenerateFloatTypes.h"

#undef CUDA_MAX_THREADS
