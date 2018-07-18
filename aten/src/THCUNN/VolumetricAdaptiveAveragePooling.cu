#include "THCUNN.h"
#include "THCTensor.hpp"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"

#define START_IND(a,b,c) (int64_t)floor((float)(a * c) / b)
#define END_IND(a,b,c) (int64_t)ceil((float)((a + 1) * c) / b)
// #define START_IND(a,b,c) a * c / b
// #define END_IND(a,b,c)  (a + 1) * c / b + ((a + 1) * c % b > 0)?1:0


#define CUDA_MAX_THREADS 1024   // this is safe, in reality 256 is our limit

// 5d tensor B x D x T x H x W
// All kernels view batch dim B and feature dim D as collapsed.

/*
 * Description:
 *    This function adaptively average pools an input 5D tensor along dimensions
 *     2, 3 and 4.
 *
 *    gridDim.y blocks work together on a single 2D output plane specified by
 *    (blockIdx.x + offsetZ).
 */
 template <typename T>
__global__ void cunn_VolumetricAdaptiveAveragePooling_updateOutput_kernel(
                        T *input, T *output,
                        int64_t isizeT, int64_t isizeH, int64_t isizeW,
                        int64_t osizeT, int64_t osizeH, int64_t osizeW,
                        int64_t istrideD,
                        int64_t istrideT, int64_t istrideH, int64_t istrideW,
                        int64_t offsetZ)
{
  // iterators on output pixels
  int64_t ot, oh, ow;

  // compute offsets based on thread/block ID
  int64_t ostartH = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t oendH   = osizeH;
  int64_t ostepH  = gridDim.y * blockDim.y;
  int64_t ostartW = threadIdx.x;
  int64_t oendW   = osizeW;
  int64_t ostepW  = blockDim.x;

  // select output plane
  int64_t o_plane = blockIdx.x + offsetZ;
  ot = o_plane % osizeT;     // output frame/time
  int64_t d = o_plane / osizeT;  // slice/feature

  // input frame/time ramge is fixed.
  int64_t istartT = START_IND(ot, osizeT, isizeT);
  int64_t iendT = END_IND(ot, osizeT, isizeT);
  int64_t kT = iendT - istartT;

  // input offset by slice/feature and earliest relevant frame/time
  T *input_dt = input + d*istrideD + istartT*istrideT;
  // output offset by slice/feature and frame/time
  T *output_dt = output + o_plane*osizeH*osizeW;

  // For all output pixels...
  for(oh = ostartH; oh < oendH; oh += ostepH) {

    int64_t istartH = START_IND(oh, osizeH, isizeH);
    int64_t iendH   = END_IND(oh, osizeH, isizeH);
    int64_t kH = iendH - istartH;

    for(ow = ostartW; ow < oendW; ow += ostepW) {

      int64_t istartW = START_IND(ow, osizeW, isizeW);
      int64_t iendW   = END_IND(ow, osizeW, isizeW);
      int64_t kW = iendW - istartW;

      // Compute the average pooling from corresponding input pixels
      T *ptr_input = input_dt + istartH*istrideH + istartW*istrideW;
      T *ptr_output = output_dt + oh*osizeW + ow;
      T sum = ScalarConvert<int, T>::to(0);

      int64_t it, ih, iw;
      for(it = 0; it < kT; ++it) {
        for(ih = 0; ih < kH; ++ih) {
          for(iw = 0; iw < kW; ++iw) {
            T val = ptr_input[ih*istrideH + iw*istrideW];
            sum += val;
          }
        }
        ptr_input += istrideT;   // next input frame
      }
      // Update output
      *ptr_output = sum / kT / kH / kW;
    }
  }
}

/*
 * Description:
 *    This function computes the gradInput from gradOutput.
 *
 *    gridDim.y blocks work together on a single 2D input plane specified by
 *    (blockIdx.x + offsetZ).
 */
 template <typename T>
__global__ void cunn_VolumetricAdaptiveAveragePooling_updateGradInput_kernel(
  T *gradInput, T *gradOutput,
  int64_t isizeT, int64_t isizeH, int64_t isizeW,
  int64_t osizeT, int64_t osizeH, int64_t osizeW,
  int64_t offsetZ
)
{
  // iterators on input pixels
  int64_t it, ih, iw;

  // compute offsets based on thread/block ID
  int64_t istartH = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t iendH   = isizeH;
  int64_t istepH  = gridDim.y * blockDim.y;
  int64_t istartW = threadIdx.x;
  int64_t iendW   = isizeW;
  int64_t istepW  = blockDim.x;

  // select input plane
  int64_t i_plane = blockIdx.x + offsetZ;
  it = i_plane % isizeT;        // output frame/time
  int64_t d = i_plane / isizeT;     // slice/feature

  // output frame/time ramge is fixed.
  int64_t ostartT = START_IND(it, isizeT, osizeT);
  int64_t oendT   = END_IND(it, isizeT, osizeT);

  // gradInput offset by slice/feature and frame/time
  T *gradInput_dt = gradInput + i_plane*isizeH*isizeW;
  // gradOutput offset by slice/feature and earliest relevant frame/time
  T *gradOutput_dt = gradOutput + (d*osizeT + ostartT)*osizeH*osizeW;

  // For all input pixels...
  for(ih = istartH; ih < iendH; ih += istepH) {

    int64_t ostartH = START_IND(ih, isizeH, osizeH);
    int64_t oendH   = END_IND(ih, isizeH, osizeH);

    for(iw = istartW; iw < iendW; iw += istepW) {

      int64_t ostartW = START_IND(iw, isizeW, osizeW);
      int64_t oendW   = END_IND(iw, isizeW, osizeW);

      // Compute the gradients from corresponding output pixels
      T *ptr_gradInput = gradInput_dt + ih*isizeW + iw;
      T *ptr_gradOutput = gradOutput_dt;

      // for all relevant output pixels
      int64_t ot, oh, ow;
      for(ot = ostartT; ot < oendT; ++ot) {
        int64_t kT = END_IND(ot, osizeT, isizeT) - START_IND(ot, osizeT, isizeT);
        for(oh = ostartH; oh < oendH; ++oh) {
          int64_t kH = END_IND(oh, osizeH, isizeH) - START_IND(oh, osizeH, isizeH);
          for(ow = ostartW; ow < oendW; ++ow) {
            int64_t kW = END_IND(ow, osizeW, isizeW) - START_IND(ow, osizeW, isizeW);
            T grad_delta = ptr_gradOutput[oh*osizeW + ow] / kW / kH / kT;
            *ptr_gradInput += grad_delta;
          }
        }
        ptr_gradOutput += osizeH*osizeW;   // next output frame
      }
    }
  }
}

/*
 * Description:
 *    This function computes the gradInput from gradOutput without assuming
 *    dependencies between input pixels and output pixels.
 *
 *    gridDim.y blocks work together on a single 2D output plane specified by
 *    (blockIdx.x + offsetZ).
 *
 *    (uses atomic add)
 */
 template <typename T>
__global__ void cunn_atomic_VolumetricAdaptiveAveragePooling_updateGradInput_kernel(
  T *gradInput, T *gradOutput,
  int64_t isizeT, int64_t isizeH, int64_t isizeW,
  int64_t osizeT, int64_t osizeH, int64_t osizeW,
  int64_t offsetZ
)
{
  // iterators on output pixels
  int64_t ot, oh, ow;

  // compute offsets based on thread/block ID
  int64_t ostartH = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t oendH   = osizeH;
  int64_t ostepH  = gridDim.y * blockDim.y;
  int64_t ostartW = threadIdx.x;
  int64_t oendW   = osizeW;
  int64_t ostepW  = blockDim.x;

  // select output plane
  int64_t o_plane = blockIdx.x + offsetZ;
  ot = o_plane % osizeT;        // output frame/time
  int64_t d = o_plane / osizeT;     // output slice/feature

  // input frame/time ramge is fixed.
  int64_t istartT = START_IND(ot, osizeT, isizeT);
  int64_t iendT = END_IND(ot, osizeT, isizeT);
  int64_t kT = iendT - istartT;

  // gradInput offset by slice/feature and earliest relevant frame/time
  T *gradInput_nt = gradInput + (d*isizeT + istartT)*isizeH*isizeW;
  // gradOutput offset by slice/feature and frame/time
  T *gradOutput_nt = gradOutput + o_plane*osizeH*osizeW;

  // For all output pixels...
  for(oh = ostartH; oh < oendH; oh += ostepH) {

    int64_t istartH = START_IND(oh, osizeH, isizeH);
    int64_t iendH   = END_IND(oh, osizeH, isizeH);
    int64_t kH = iendH - istartH;

    for(ow = ostartW; ow < oendW; ow += ostepW) {

      int64_t istartW = START_IND(ow, osizeW, isizeW);
      int64_t iendW   = END_IND(ow, osizeW, isizeW);
      int64_t kW = iendW - istartW;

      // Compute the gradients from corresponding input pixels
      T *ptr_gradInput = gradInput_nt + istartH*isizeW + istartW;
      T *ptr_gradOutput = gradOutput_nt + oh*osizeW + ow;
      T grad_delta = *ptr_gradOutput / kT / kH / kW;

      int64_t it, ih, iw;
      for(it = 0; it < kT; ++it) {
        for(ih = 0; ih < kH; ++ih) {
          for(iw = 0; iw < kW; ++iw) {
            atomicAdd(&(ptr_gradInput[ih*isizeW + iw]), grad_delta);
          }
        }
        ptr_gradInput += isizeH*isizeW;   // next input frame
      }
    }
  }
}

#include "generic/VolumetricAdaptiveAveragePooling.cu"
#include "THCGenerateFloatTypes.h"

#undef CUDA_MAX_THREADS
#undef START_IND
#undef END_IND
