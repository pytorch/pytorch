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
                        int sizeD, int isizeH, int isizeW,
                        int osizeH, int osizeW,
                        int istrideH, int istrideW,
                        int istrideD)
{
  // iterators
  int ow, oh;

  // compute offsets based on thread/block ID
  int o_plane = blockIdx.x;
  int i_plane = o_plane;
  //int k = blockIdx.x % sizeD;

  int ostartW = threadIdx.x;
  int oendW = osizeW;
  const int ostepW = blockDim.x;

  int ostartH = blockDim.y*blockIdx.y + threadIdx.y;
  int oendH = osizeH;
  const int ostepH = blockDim.y*gridDim.y;
  // select input/output plane
  output = output + o_plane*osizeW*osizeH;
  input = input + i_plane*istrideD;
  indices = indices + o_plane*osizeW*osizeH;

  // For all output pixels...
  for(oh = ostartH; oh < oendH; oh+=ostepH) {

    int istartH = (int)floor(float(oh) / osizeH * isizeH);
    int iendH   = (int)ceil(float(oh+1) / osizeH * isizeH);
    int kH = iendH-istartH;

    for(ow = ostartW; ow < oendW; ow+=ostepW) {
      int istartW = (int)floor(float(ow) / osizeW * isizeW);
      int iendW   = (int)ceil(float(ow + 1) / osizeW * isizeW);

      int kW = iendW-istartW;

      // Compute the mean of the input image...
      T *ptr_input = input + istartH*istrideH + istartW*istrideW;
      T *ptr_output = output + oh*osizeW + ow;
      THCIndex_t *ptr_ind = indices + oh*osizeW + ow;
      int argmax = -1;
      T max = THCNumerics<T>::min();
      int iw, ih;
      for(ih = 0; ih < kH; ih++) {
        for(iw = 0; iw < kW; iw++) {
          T val = ptr_input[iw*istrideW];
          if (val > max) {
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
                             int sizeD, int isizeH, int isizeW,
                             int osizeH, int osizeW)
{
  // iterators
  int ow, oh;

  // compute offsets based on thread/block ID
  int o_plane = blockIdx.x;
  int i_plane = o_plane;
  //int k = blockIdx.x % sizeD;

  int ostartW = threadIdx.x;
  int oendW = osizeW;
  int ostepW = blockDim.x;

  int ostartH = blockDim.y*blockIdx.y + threadIdx.y;
  int oendH = osizeH;
  int ostepH = blockDim.y*gridDim.y;

  // select input/output plane
  gradOutput = gradOutput + o_plane*osizeW*osizeH;
  gradInput = gradInput + i_plane*isizeW*isizeH;
  indices = indices + o_plane*osizeW*osizeH;

  // compute gradInput
  for(oh = ostartH; oh < oendH; oh+=ostepH) {

    int istartH = (int)floor(float(oh) / osizeH * isizeH);

    for(ow = ostartW; ow < oendW; ow+=ostepW) {

      int istartW = (int)floor(float(ow) / osizeW * isizeW);

      T *ptr_gradInput = gradInput + istartH*isizeW + istartW;
      T *ptr_gradOutput = gradOutput + oh*osizeW + ow;
      THCIndex_t *ptr_ind = indices + oh*osizeW + ow;
      T z = *ptr_gradOutput;

      int argmax = (*ptr_ind) - TH_INDEX_BASE - istartW - istartH*isizeW;

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
  int sizeD, int isizeH, int isizeW, int osizeH, int osizeW
)
{
  // iterators
  int ow, oh;

  // compute offsets based on thread/block ID
  int o_plane = blockIdx.x;
  int i_plane = o_plane;

  int ostartW = threadIdx.x;
  int oendW = osizeW;
  int ostepW = blockDim.x;

  int ostartH = blockDim.y*blockIdx.y + threadIdx.y;
  int oendH = osizeH;
  int ostepH = blockDim.y*gridDim.y;

  // select input/output plane
  gradOutput = gradOutput + o_plane*osizeW*osizeH;
  gradInput = gradInput + i_plane*isizeW*isizeH;
  indices = indices + o_plane*osizeW*osizeH;

  // compute gradInput
  for(oh = ostartH; oh < oendH; oh+=ostepH) {

    int istartH = (int)floor(float(oh) / osizeH * isizeH);

    for(ow = ostartW; ow < oendW; ow+=ostepW) {

      int istartW = (int)floor(float(ow) / osizeW * isizeW);

      T *ptr_gradInput = gradInput + istartH*isizeW + istartW;
      T *ptr_gradOutput = gradOutput + oh*osizeW + ow;
      THCIndex_t *ptr_ind = indices + oh*osizeW + ow;
      T z = *ptr_gradOutput;

      int argmax = (*ptr_ind) - TH_INDEX_BASE - istartW - istartH*isizeW;

      // atomic add since different threads could update same variable
      atomicAdd(&(ptr_gradInput[argmax]), z);
    }
  }
}

#include "generic/SpatialAdaptiveMaxPooling.cu"
#include "THCGenerateFloatTypes.h"

#undef CUDA_MAX_THREADS
