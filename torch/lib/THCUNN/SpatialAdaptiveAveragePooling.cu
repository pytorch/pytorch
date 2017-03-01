#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"

#define START_IND(a,b,c) (int)floor((float)(a * c) / b)
#define END_IND(a,b,c) (int)ceil((float)((a + 1) * c) / b)
// #define START_IND(a,b,c) a * c / b
// #define END_IND(a,b,c)  (a + 1) * c / b + ((a + 1) * c % b > 0)?1:0


#define CUDA_MAX_THREADS 1024   // this is safe, in reality 256 is our limit

/*
 * Description:
 *    this function adaptively average pools an input 4D tensor along dimensions 2 and 3
 *    4D input, 4D output
 */
 template <typename T>
__global__ void adaptiveaveragepool(T *input, T *output,
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

  // For all output pixels...
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {

    int y_start = START_IND(yy, output_h, input_h);
    int y_end   = END_IND(yy, output_h, input_h);
    int kH = y_end-y_start;

    for(xx = xx_start; xx < xx_end; xx+=xx_step) {

      int x_start = START_IND(xx, output_w, input_w);
      int x_end   = END_IND(xx, output_w, input_w);
      int kW = x_end-x_start;

      // Compute the average pooling
      T *ptr_input = input + y_start*strideh + x_start*stridew;
      T *ptr_output = output + yy*output_w + xx;
      T sum = ScalarConvert<int, T>::to(0);
      int kx, ky;
      for(ky = 0; ky < kH; ++ky) {
        for(kx = 0; kx < kW; ++kx) {
          T val = ptr_input[kx*stridew];
          sum += val;
        }
        ptr_input += strideh; // next input line
      }
      // Update output
      *ptr_output = sum / kH / kW;
    }
  }
}

/*
 * Description:
 *    this function computes the gradInput from gradOutput
 */
 template <typename T>
__global__ void adaptiveaveragegradinput(
  T *gradInput, T *gradOutput,
  int input_n, int input_h, int input_w, int output_h, int output_w
)
{
  // iterators
  int x, y;

  // compute offsets based on thread/block ID
  int o = blockIdx.x;
  int i = o;

  int x_start = threadIdx.x;
  int x_end = input_w;
  int x_step = blockDim.x;

  int y_start = blockDim.y*blockIdx.y + threadIdx.y;
  int y_end = input_h;
  int y_step = blockDim.y*gridDim.y;

  // select input/output plane
  gradOutput = gradOutput + o*output_w*output_h;
  gradInput = gradInput + i*input_w*input_h;

  // compute gradInput
  for(y = y_start; y < y_end; y+=y_step) {

    int yy_start = START_IND(y, input_h, output_h);
    int yy_end   = END_IND(y, input_h, output_h);
    int kH = yy_end-yy_start;

    for(x = x_start; x < x_end; x+=x_step) {

      int xx_start = START_IND(x, input_w, output_w);
      int xx_end   = END_IND(x, input_w, output_w);
      int kW = xx_end-xx_start;

      // Compute the gradients
      T *ptr_gradInput = gradInput + y*input_w + x;
      T *ptr_gradOutput = gradOutput + yy_start*output_w + xx_start;
      
      int kx, ky;
      for(ky = 0; ky < kH; ++ky) {
        int yy = yy_start + ky;
        int kkH = START_IND(yy, output_h, input_h) - END_IND(yy, output_h, input_h);
        for(kx = 0; kx < kW; ++kx) {
          int xx = xx_start + kx;
          int kkW = START_IND(xx, output_w, input_w) - END_IND(xx, output_w, input_w);
          T z = ptr_gradOutput[kx + ky*output_w] / kkW / kkH;
          *ptr_gradInput += z;
        }
      }
    }
  }
}

/*
 * Description:
 *    this function computes the gradInput from gradOutput
 *    (uses atomic add)
 */
 template <typename T>
__global__ void atomicadaptiveaveragegradinput(
  T *gradInput, T *gradOutput,
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

  // compute gradInput
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {

    int y_start = START_IND(yy, output_h, input_h);
    int y_end   = END_IND(yy, output_h, input_h);
    int kH = y_end-y_start;

    for(xx = xx_start; xx < xx_end; xx+=xx_step) {

      int x_start = START_IND(xx, output_w, input_w);
      int x_end   = END_IND(xx, output_w, input_w);
      int kW = x_end-x_start;

      // Compute the gradients
      T *ptr_gradInput = gradInput + y_start*input_w + x_start;
      T *ptr_gradOutput = gradOutput + yy*output_w + xx;
      T z = *ptr_gradOutput / kW / kH;
      int kx, ky;
      for(ky = 0; ky < kH; ++ky) {
        for(kx = 0; kx < kW; ++kx) {
          // atomic add since different threads could update same variable
          atomicAdd(&(ptr_gradInput[kx + ky*input_w]), z);
        }
      }
    }
  }
}

#include "generic/SpatialAdaptiveAveragePooling.cu"
#include "THCGenerateFloatTypes.h"

#undef CUDA_MAX_THREADS
#undef START_IND
#undef END_IND
