#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"

#define CUDA_MAX_THREADS 1024   // this is safe, in reality 256 is our limit

#define START_IND(a,b,c) (int)floor((float)(a * c) / b)
#define END_IND(a,b,c) (int)ceil((float)((a + 1) * c) / b)
// #define START_IND(a,b,c) a * c / b
// #define END_IND(a,b,c)  (a + 1) * c / b + ((a + 1) * c % b > 0)?1:0

// 5d tensor B x D x T x H x W

/*
 * Description:
 *    this function adaptively maxpools an input 4D tensor along dimensions 2 and 3
 *    4D input, 4D output, 4D argmax x and y
 */
 template <typename T>
__global__ void cunn_VolumetricAdaptiveMaxPooling_updateOutput_kernel(
                        T *input, T *output, THCIndex_t *indices,
                        int isizeT, int isizeH, int isizeW,
                        int osizeT, int osizeH, int osizeW,
                        int64_t istrideD,
                        int64_t istrideT, int64_t istrideH, int64_t istrideW,
                        int64_t offsetZ)
{
  // iterators on output pixels
  int ot, oh, ow;

  // compute offsets based on thread/block ID
  int ostartH = blockIdx.y * blockDim.y + threadIdx.y;
  int oendH   = osizeH;
  int ostepH  = gridDim.y * blockDim.y;
  int ostartW = threadIdx.x;
  int oendW   = osizeW;
  int ostepW  = blockDim.x;

  // select output plane
  int64_t o_plane = blockIdx.x + offsetZ;
  ot = o_plane % osizeT;     // output frame/time
  int d = o_plane / osizeT;  // slice/feature

  // input frame/time ramge is fixed.
  int istartT = START_IND(ot, osizeT, isizeT);
  int iendT = END_IND(ot, osizeT, isizeT);
  int kT = iendT - istartT;

  // input offset by slice/feature and earliest relevant frame/time
  T *input_dt = input + d*istrideD + istartT*istrideT;
  // output offset by slice/feature and frame/time
  T *output_dt = output + o_plane*osizeH*osizeW;
  // indices offset by slice/feature and frame/time
  THCIndex_t *indices_dt = indices + o_plane*osizeH*osizeW;

  // For all output pixels...
  for(oh = ostartH; oh < oendH; oh += ostepH) {

    int istartH = START_IND(oh, osizeH, isizeH);
    int iendH   = END_IND(oh, osizeH, isizeH);
    int kH = iendH - istartH;

    for(ow = ostartW; ow < oendW; ow += ostepW) {

      int istartW = START_IND(ow, osizeW, isizeW);
      int iendW   = END_IND(ow, osizeW, isizeW);
      int kW = iendW - istartW;

      // Compute the average pooling from corresponding input pixels
      T *ptr_input = input_dt + istartH*istrideH + istartW*istrideW;
      T *ptr_output = output_dt + oh*osizeW + ow;
      THCIndex_t *ptr_ind = indices_dt + oh*osizeW + ow;
      int64_t argmax = -1;
      T max = THCNumerics<T>::min();

      int it, ih, iw;
      for(it = 0; it < kT; ++it) {
        for(ih = 0; ih < kH; ++ih) {
          for(iw = 0; iw < kW; ++iw) {
            T val = ptr_input[ih*istrideH + iw*istrideW];
            if (val > max) {
              max = val;
              argmax = (it+istartT)*isizeH*isizeW + (ih+istartH)*isizeW + iw+istartW;
            }
          }
        }
        ptr_input += istrideT;   // next input frame
      }
      // Update output and argmax
      *ptr_output = max;
      *ptr_ind = argmax + TH_INDEX_BASE;
    }
  }
}

/*
 * Description:
 *    This function computes the gradInput from gradOutput.
 *
 *    gridDim.y blocks work together on a single 2D output plane specified by
 *    (blockIdx.x + offsetZ).
 *
 *    Assumes that input size can be perfectly divided by output size, i.e.
 *    each input pixel can only be argmax of one output pixel.
 */
 template <typename T>
__global__ void cunn_VolumetricAdaptiveMaxPooling_updateGradInput_kernel(
  T *gradInput, T *gradOutput, THCIndex_t *indices,
  int isizeT, int isizeH, int isizeW,
  int osizeT, int osizeH, int osizeW,
  int64_t offsetZ
)
{
  // iterators on output pixels
  int oh, ow;

  // compute offsets based on thread/block ID
  int ostartH = blockIdx.y * blockDim.y + threadIdx.y;
  int oendH   = osizeH;
  int ostepH  = gridDim.y * blockDim.y;
  int ostartW = threadIdx.x;
  int oendW   = osizeW;
  int ostepW  = blockDim.x;

  // select output plane
  int64_t o_plane = blockIdx.x + offsetZ;
  int d = o_plane / osizeT;     // output slice/feature

  // gradInput offset by slice/feature
  T *gradInput_d = gradInput + d*isizeT*isizeH*isizeW;
  // gradOutput offset by slice/feature and frame/otme
  T *gradOutput_dt = gradOutput + o_plane*osizeH*osizeW;
  // indices offset by slice/feature and frame/otme
  THCIndex_t *indices_dt = indices + o_plane*osizeH*osizeW;

  // For all output pixels...
  for(oh = ostartH; oh < oendH; oh += ostepH) {
    for(ow = ostartW; ow < oendW; ow += ostepW) {
      // Compute the gradients for the argmax input pixel
      T *ptr_gradOutput = gradOutput_dt + oh*osizeW + ow;
      THCIndex_t *ptr_ind = indices_dt + oh*osizeW + ow;
      T grad_delta = *ptr_gradOutput;
      int argmax = (*ptr_ind) - TH_INDEX_BASE;
      gradInput_d[argmax] += grad_delta;
    }
  }
}


/*
 * Description:
 *    This function computes the gradInput from gradOutput.
 *
 *    gridDim.y blocks work together on a single 2D output plane specified by
 *    (blockIdx.x + offsetZ).
 *
 *    Uses atomic add.
 */
 template <typename T>
__global__ void cunn_atomic_VolumetricAdaptiveMaxPooling_updateGradInput_kernel(
  T *gradInput, T *gradOutput, THCIndex_t *indices,
  int isizeT, int isizeH, int isizeW,
  int osizeT, int osizeH, int osizeW,
  int64_t offsetZ
)
{
  // iterators on output pixels
  int oh, ow;

  // compute offsets based on thread/block ID
  int ostartH = blockIdx.y * blockDim.y + threadIdx.y;
  int oendH   = osizeH;
  int ostepH  = gridDim.y * blockDim.y;
  int ostartW = threadIdx.x;
  int oendW   = osizeW;
  int ostepW  = blockDim.x;

  // select output plane
  int64_t o_plane = blockIdx.x + offsetZ;
  int d = o_plane / osizeT;     // output slice/feature

  // gradInput offset by slice/feature
  T *gradInput_d = gradInput + d*isizeT*isizeH*isizeW;
  // gradOutput offset by slice/feature and frame/otme
  T *gradOutput_dt = gradOutput + o_plane*osizeH*osizeW;
  // indices offset by slice/feature and frame/otme
  THCIndex_t *indices_dt = indices + o_plane*osizeH*osizeW;

  // For all output pixels...
  for(oh = ostartH; oh < oendH; oh += ostepH) {
    for(ow = ostartW; ow < oendW; ow += ostepW) {
      // Compute the gradients for the argmax input pixel
      T *ptr_gradOutput = gradOutput_dt + oh*osizeW + ow;
      THCIndex_t *ptr_ind = indices_dt + oh*osizeW + ow;
      T grad_delta = *ptr_gradOutput;
      int64_t argmax = (*ptr_ind) - TH_INDEX_BASE;
      atomicAdd(&(gradInput_d[argmax]), grad_delta);
    }
  }
}

#include "generic/VolumetricAdaptiveMaxPooling.cu"
#include "THCGenerateFloatTypes.h"

#undef CUDA_MAX_THREADS
