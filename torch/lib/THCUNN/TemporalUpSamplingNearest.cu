#include "THCUNN.h"
#include "common.h"

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

/*
 * Description:
 */

__device__ int translate_idx(int ii, int d1, int d2, int scale_factor)
{
  int x, y, z;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  z = z/scale_factor;
  d2 /= scale_factor;
  return ((x*d1+y)*d2)+z;

}
__device__ int translate_idx_inv(int ii, int d1, int d2, int scale_factor, int off_x)
{
  int x, y, z;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  z = z*scale_factor+off_x;
  d2 *= scale_factor;
  return ((x*d1+y)*d2)+z;

}

template <typename Dtype>
__global__ void upscale(Dtype *input, Dtype *output, int64_t no_elements,
                        int scale_factor, int d1, int d2)
{
  // output offset:
  int64_t ii = threadIdx.x + blockDim.x * blockIdx.x;
  ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
  if (ii >= no_elements) return;
  int ipidx = translate_idx(ii, d1, d2, scale_factor);
  output[ii]=input[ipidx];
}

/*
 * Description:
 */
template <typename Dtype, typename Acctype>
__global__ void downscale(Dtype *gradInput_data, Dtype *gradOutput_data, int64_t no_elements,
                              int scale_factor, int d1, int d2)
{
  // output offset:
  int64_t ii = threadIdx.x + blockDim.x * blockIdx.x;
  ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
  if (ii >= no_elements) return;
  Acctype sum = Acctype(0);
  for (int i=0; i < scale_factor; i++){
    int ipidx = translate_idx_inv(ii, d1, d2, scale_factor, i);
    sum += gradOutput_data[ipidx];
  }
  gradInput_data[ii] += ScalarConvert<Acctype, Dtype>::to(sum);
}

#include "generic/TemporalUpSamplingNearest.cu"
#include "THCGenerateFloatTypes.h"
