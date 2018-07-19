#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"
#include "THCTensor.hpp"

#define CUDA_MAX_THREADS 1024   // this is safe, in reality 256 is our limit

#define START_IND(a,b,c) (int64_t)floor((float)(a * c) / b)
#define END_IND(a,b,c) (int64_t)ceil((float)((a + 1) * c) / b)
// #define START_IND(a,b,c) a * c / b
// #define END_IND(a,b,c)  (a + 1) * c / b + ((a + 1) * c % b > 0)?1:0

// 4d tensor B x D x H x W

/*
 * Description:
 *    this function adaptively maxpools an input 4D tensor along dimensions 2 and 3
 *    4D input, 4D output, 4D argmax x and y
 */
 template <typename T>
__global__ void adaptivemaxpool(T *input, T *output, THCIndex_t *indices,
                        int64_t isizeH, int64_t isizeW,
                        int64_t osizeH, int64_t osizeW,
                        int64_t istrideD, int64_t istrideH, int64_t istrideW)
{
  // iterators
  int64_t oh, ow;

  // compute offsets based on thread/block ID
  int64_t o_plane = blockIdx.x;
  int64_t i_plane = o_plane;

  int64_t ostartW = threadIdx.x;
  int64_t oendW = osizeW;
  const int64_t ostepW = blockDim.x;

  int64_t ostartH = blockDim.y*blockIdx.y + threadIdx.y;
  int64_t oendH = osizeH;
  const int64_t ostepH = blockDim.y*gridDim.y;
  // select input/output plane
  output = output + o_plane*osizeH*osizeW;
  input = input + i_plane*istrideD;
  indices = indices + o_plane*osizeH*osizeW;

  // For all output pixels...
  for(oh = ostartH; oh < oendH; oh += ostepH) {

    int64_t istartH = START_IND(oh, osizeH, isizeH);
    int64_t iendH   = END_IND(oh, osizeH, isizeH);
    int64_t kH = iendH - istartH;

    for(ow = ostartW; ow < oendW; ow += ostepW) {
      int64_t istartW = START_IND(ow, osizeW, isizeW);
      int64_t iendW   = END_IND(ow, osizeW, isizeW);

      int64_t kW = iendW - istartW;

      // Compute the mean of the input image...
      T *ptr_input = input + istartH*istrideH + istartW*istrideW;
      T *ptr_output = output + oh*osizeW + ow;
      THCIndex_t *ptr_ind = indices + oh*osizeW + ow;
      int64_t argmax = -1;
      T max = THCNumerics<T>::min();
      int64_t ih, iw;
      for(ih = 0; ih < kH; ih++) {
        for(iw = 0; iw < kW; iw++) {
          T val = ptr_input[iw*istrideW];
          if ((val > max) || THCNumerics<T>::isnan(val)) {
            max = val;
            argmax = (ih+istartH)*isizeW + iw+istartW;
          }
        }
        ptr_input += istrideH; // next input line
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
                             int64_t isizeH, int64_t isizeW,
                             int64_t osizeH, int64_t osizeW)
{
  // iterators
  int64_t oh, ow;

  // compute offsets based on thread/block ID
  int64_t o_plane = blockIdx.x;
  int64_t i_plane = o_plane;
  //int k = blockIdx.x % sizeD;

  int64_t ostartW = threadIdx.x;
  int64_t oendW = osizeW;
  int64_t ostepW = blockDim.x;

  int64_t ostartH = blockDim.y*blockIdx.y + threadIdx.y;
  int64_t oendH = osizeH;
  int64_t ostepH = blockDim.y*gridDim.y;

  // select input/output plane
  gradOutput = gradOutput + o_plane*osizeH*osizeW;
  gradInput = gradInput + i_plane*isizeH*isizeW;
  indices = indices + o_plane*osizeH*osizeW;

  // compute gradInput
  for(oh = ostartH; oh < oendH; oh += ostepH) {

    for(ow = ostartW; ow < oendW; ow += ostepW) {

      T *ptr_gradOutput = gradOutput + oh*osizeW + ow;
      THCIndex_t *ptr_ind = indices + oh*osizeW + ow;
      T z = *ptr_gradOutput;

      int64_t argmax = (*ptr_ind) - TH_INDEX_BASE;

      gradInput[argmax] += z;
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
  int64_t isizeH, int64_t isizeW, int64_t osizeH, int64_t osizeW
)
{
  // iterators
  int64_t oh, ow;

  // compute offsets based on thread/block ID
  int64_t o_plane = blockIdx.x;
  int64_t i_plane = o_plane;

  int64_t ostartW = threadIdx.x;
  int64_t oendW = osizeW;
  int64_t ostepW = blockDim.x;

  int64_t ostartH = blockDim.y*blockIdx.y + threadIdx.y;
  int64_t oendH = osizeH;
  int64_t ostepH = blockDim.y*gridDim.y;

  // select input/output plane
  gradOutput = gradOutput + o_plane*osizeH*osizeW;
  gradInput = gradInput + i_plane*isizeH*isizeW;
  indices = indices + o_plane*osizeH*osizeW;

  // compute gradInput
  for(oh = ostartH; oh < oendH; oh += ostepH) {

    for(ow = ostartW; ow < oendW; ow += ostepW) {

      T *ptr_gradOutput = gradOutput + oh*osizeW + ow;
      THCIndex_t *ptr_ind = indices + oh*osizeW + ow;
      T z = *ptr_gradOutput;

      int64_t argmax = (*ptr_ind) - TH_INDEX_BASE;

      // atomic add since different threads could update same variable
      atomicAdd(&(gradInput[argmax]), z);
    }
  }
}

#include "generic/SpatialAdaptiveMaxPooling.cu"
#include "THCGenerateFloatTypes.h"

#undef CUDA_MAX_THREADS
