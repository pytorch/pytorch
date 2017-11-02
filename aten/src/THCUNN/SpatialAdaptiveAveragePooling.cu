#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"

#define START_IND(a,b,c) (int)floor((float)(a * c) / b)
#define END_IND(a,b,c) (int)ceil((float)((a + 1) * c) / b)
// #define START_IND(a,b,c) a * c / b
// #define END_IND(a,b,c)  (a + 1) * c / b + ((a + 1) * c % b > 0)?1:0


#define CUDA_MAX_THREADS 1024   // this is safe, in reality 256 is our limit

// 4d tensor B x D x H x W
// All kernels view batch dim B and feature dim D as collapsed.

/*
 * Description:
 *    this function adaptively average pools an input 4D tensor along dimensions 2 and 3
 *    4D input, 4D output
 */
 template <typename T>
__global__ void adaptiveaveragepool(T *input, T *output,
                        int isizeH, int isizeW,
                        int osizeH, int osizeW,
                        int64_t istrideD, int64_t istrideH, int64_t istrideW)
{
  // iterators on output pixels
  int oh, ow;

  // select input/output plane based on thread/block ID
  int o_plane = blockIdx.x;
  int i_plane = o_plane;

  output = output + o_plane*osizeH*osizeW;
  input = input + i_plane*istrideD;

  int ostartH = blockDim.y*blockIdx.y + threadIdx.y;
  int oendH = osizeH;
  const int ostepH = blockDim.y*gridDim.y;

  int ostartW = threadIdx.x;
  int oendW = osizeW;
  const int ostepW = blockDim.x;

  // For all output pixels...
  for(oh = ostartH; oh < oendH; oh += ostepH) {

    int istartH = START_IND(oh, osizeH, isizeH);
    int iendH   = END_IND(oh, osizeH, isizeH);
    int kH = iendH - istartH;

    for(ow = ostartW; ow < oendW; ow += ostepW) {

      int istartW = START_IND(ow, osizeW, isizeW);
      int iendW   = END_IND(ow, osizeW, isizeW);
      int kW = iendW - istartW;

      // Compute the average pooling over corresponding input pixels
      T *ptr_input = input + istartH*istrideH + istartW*istrideW;
      T *ptr_output = output + oh*osizeW + ow;
      T sum = ScalarConvert<int, T>::to(0);
      int ih, iw;
      for(ih = 0; ih < kH; ++ih) {
        for(iw = 0; iw < kW; ++iw) {
          T val = ptr_input[iw*istrideW];
          sum += val;
        }
        ptr_input += istrideH; // next input line
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
  int isizeH, int isizeW, int osizeH, int osizeW
)
{
  // iterators on input pixels
  int ih, iw;

  // select input/output plane based on thread/block ID
  int i_plane = blockIdx.x;
  int o_plane = i_plane;

  gradOutput = gradOutput + o_plane*osizeH*osizeW;
  gradInput = gradInput + i_plane*isizeH*isizeW;

  int istartH = blockDim.y*blockIdx.y + threadIdx.y;
  int iendH = isizeH;
  int istepH = blockDim.y*gridDim.y;

  int istartW = threadIdx.x;
  int iendW = isizeW;
  int istepW = blockDim.x;

  // compute gradInput
  for(ih = istartH; ih < iendH; ih += istepH) {

    int ostartH = START_IND(ih, isizeH, osizeH);
    int oendH   = END_IND(ih, isizeH, osizeH);

    for(iw = istartW; iw < iendW; iw += istepW) {

      int ostartW = START_IND(iw, isizeW, osizeW);
      int oendW   = END_IND(iw, isizeW, osizeW);

      // Compute the gradients over corresponding output pixels
      T *ptr_gradInput = gradInput + ih*isizeW + iw;

      int oh, ow;
      for(oh = ostartH; oh < oendH; ++oh) {
        int kH = START_IND(oh, osizeH, isizeH) - END_IND(oh, osizeH, isizeH);
        for(ow = ostartW; ow < oendW; ++ow) {
          int kW = START_IND(ow, osizeW, isizeW) - END_IND(ow, osizeW, isizeW);
          T grad_delta = gradOutput[ow + oh*osizeW] / kH / kW;
          *ptr_gradInput += grad_delta;
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
  int isizeH, int isizeW, int osizeH, int osizeW
)
{
  // iterators on output indices
  int oh, ow;

  // select input/output plane based on thread/block ID
  int o_plane = blockIdx.x;
  int i_plane = o_plane;

  gradOutput = gradOutput + o_plane*osizeW*osizeH;
  gradInput = gradInput + i_plane*isizeW*isizeH;

  int ostartH = blockDim.y*blockIdx.y + threadIdx.y;
  int oendH = osizeH;
  int ostepH = blockDim.y*gridDim.y;

  int ostartW = threadIdx.x;
  int oendW = osizeW;
  int ostepW = blockDim.x;

  // For all output pixels...
  for(oh = ostartH; oh < oendH; oh += ostepH) {

    int istartH = START_IND(oh, osizeH, isizeH);
    int iendH   = END_IND(oh, osizeH, isizeH);
    int kH = iendH - istartH;

    for(ow = ostartW; ow < oendW; ow += ostepW) {

      int istartW = START_IND(ow, osizeW, isizeW);
      int iendW   = END_IND(ow, osizeW, isizeW);
      int kW = iendW - istartW;

      // Compute the gradients for over corresponding input pixels
      T *ptr_gradInput = gradInput + istartH*isizeW + istartW;
      T *ptr_gradOutput = gradOutput + oh*osizeW + ow;
      T grad_delta = *ptr_gradOutput / kW / kH;

      int ih, iw;
      for(ih = 0; ih < kH; ++ih) {
        for(iw = 0; iw < kW; ++iw) {
          // atomic add since different threads could update same variable
          atomicAdd(&(ptr_gradInput[iw]), grad_delta);
        }
        ptr_gradInput += isizeW; // next input line
      }
    }
  }
}

#include "generic/SpatialAdaptiveAveragePooling.cu"
#include "THCGenerateFloatTypes.h"

#undef CUDA_MAX_THREADS
#undef START_IND
#undef END_IND
