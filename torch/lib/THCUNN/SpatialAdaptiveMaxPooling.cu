#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"

#define CUDA_MAX_THREADS 1024   // this is safe, in reality 256 is our limit

/*
 * Description:
 *    this function adaptively maxpools an input 4D tensor along dimensions 2 and 3
 *    4D input, 4D output, 4D argmax x and y
 */
 template <typename T>
__global__ void adaptivemaxpool(T *input, T *output, THCIndex_t *indices,
                        int input_n, int input_h, int input_w,
                        int output_h, int output_w,
                        int strideh, int stridew,
                        int strided)
{
  // iterators
  int xx, yy;

  // compute offsets based on thread/block ID
  int o = blockIdx.x;
  int i = o;
  //int k = blockIdx.x % input_n;

  int xx_start = threadIdx.x;
  int xx_end = output_w;
  const int xx_step = blockDim.x;

  int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
  int yy_end = output_h;
  const int yy_step = blockDim.y*gridDim.y;
  // select input/output plane
  output = output + o*output_w*output_h;
  input = input + i*strided;
  indices = indices + o*output_w*output_h;

  // For all output pixels...
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {

    int y_start = (int)floor(float(yy) / output_h * input_h);
    int y_end   = (int)ceil(float(yy+1) / output_h * input_h);
    int kH = y_end-y_start;

    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
      int x_start = (int)floor(float(xx) / output_w * input_w);
      int x_end   = (int)ceil(float(xx + 1) / output_w * input_w);

      int kW = x_end-x_start;

      // Compute the mean of the input image...
      T *ptr_input = input + y_start*strideh + x_start*stridew;
      T *ptr_output = output + yy*output_w + xx;
      THCIndex_t *ptr_ind = indices + yy*output_w + xx;
      int argmax = -1;
      T max = THCNumerics<T>::min();
      int kx, ky;
      for(ky = 0; ky < kH; ky++) {
        for(kx = 0; kx < kW; kx++) {
          T val = ptr_input[kx*stridew];
          if (val > max) {
            max = val;
            argmax = (ky+y_start)*input_w + kx+x_start;
          }
        }
        ptr_input += strideh; // next input line
      }
      // Update output and argmax
      *ptr_output = max;
      *ptr_ind = argmax + TH_INDEX_BASE;
    }
  }
}

/*
 * Description:
 *    this function computes the gradInput from weight and gradOutput
 */
 template <typename T>
__global__ void adaptivemaxgradinput(T *gradInput, T *gradOutput, THCIndex_t *indices,
                             int input_n, int input_h, int input_w,
                             int output_h, int output_w)
{
  // iterators
  int xx, yy;

  // compute offsets based on thread/block ID
  int o = blockIdx.x;
  int i = o;
  //int k = blockIdx.x % input_n;

  int xx_start = threadIdx.x;
  int xx_end = output_w;
  int xx_step = blockDim.x;

  int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
  int yy_end = output_h;
  int yy_step = blockDim.y*gridDim.y;

  // select input/output plane
  gradOutput = gradOutput + o*output_w*output_h;
  gradInput = gradInput + i*input_w*input_h;
  indices = indices + o*output_w*output_h;

  // compute gradInput
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {

    int y_start = (int)floor(float(yy) / output_h * input_h);

    for(xx = xx_start; xx < xx_end; xx+=xx_step) {

      int x_start = (int)floor(float(xx) / output_w * input_w);

      T *ptr_gradInput = gradInput + y_start*input_w + x_start;
      T *ptr_gradOutput = gradOutput + yy*output_w + xx;
      THCIndex_t *ptr_ind = indices + yy*output_w + xx;
      T z = *ptr_gradOutput;

      int argmax = (*ptr_ind) - TH_INDEX_BASE - x_start - y_start*input_w;

      ptr_gradInput[argmax] += z;
    }
  }
}

/*
 * Description:
 *    this function computes the gradInput from weight and gradOutput
 *    when kH != dH or kW != dW (uses atomic add)
 */
 template <typename T>
__global__ void atomicadaptivemaxgradinput(
  T *gradInput, T *gradOutput, THCIndex_t *indices,
  int input_n, int input_h, int input_w, int output_h, int output_w
)
{
  // iterators
  int xx, yy;

  // compute offsets based on thread/block ID
  int o = blockIdx.x;
  int i = o;

  int xx_start = threadIdx.x;
  int xx_end = output_w;
  int xx_step = blockDim.x;

  int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
  int yy_end = output_h;
  int yy_step = blockDim.y*gridDim.y;

  // select input/output plane
  gradOutput = gradOutput + o*output_w*output_h;
  gradInput = gradInput + i*input_w*input_h;
  indices = indices + o*output_w*output_h;

  // compute gradInput
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {

    int y_start = (int)floor(float(yy) / output_h * input_h);

    for(xx = xx_start; xx < xx_end; xx+=xx_step) {

      int x_start = (int)floor(float(xx) / output_w * input_w);

      T *ptr_gradInput = gradInput + y_start*input_w + x_start;
      T *ptr_gradOutput = gradOutput + yy*output_w + xx;
      THCIndex_t *ptr_ind = indices + yy*output_w + xx;
      T z = *ptr_gradOutput;

      int argmax = (*ptr_ind) - TH_INDEX_BASE - x_start - y_start*input_w;

      // atomic add since different threads could update same variable
      atomicAdd(&(ptr_gradInput[argmax]), z);
    }
  }
}

#include "generic/SpatialAdaptiveMaxPooling.cu"
#include "THCGenerateFloatTypes.h"

#undef CUDA_MAX_THREADS
