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

__device__ int translate_idx(int ii, int d1, int d2, int d3, int d4, int scale_factor)
{
  int x, y, z, w, v;
  v = ii % d4;
  ii = ii/d4;
  w = ii % d3;
  ii = ii/d3;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  v = v/scale_factor;
  w = w/scale_factor;
  z = z/scale_factor;
  d2 /= scale_factor;
  d3 /= scale_factor;
  d4 /= scale_factor;
  return ((((x*d1+y)*d2)+z)*d3+w)*d4+v;

}
__device__ int translate_idx_inv(int ii, int d1, int d2, int d3, int d4, int scale_factor, int off_x, int off_y, int off_z)
{
  int x, y, z, w, v;
  v = ii % d4;
  ii = ii/d4;
  w = ii % d3;
  ii = ii/d3;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  v = v*scale_factor+off_x;
  w = w*scale_factor+off_y;
  z = z*scale_factor+off_z;
  d2 *= scale_factor;
  d3 *= scale_factor;
  d4 *= scale_factor;
  return ((((x*d1+y)*d2)+z)*d3+w)*d4+v;

}

template <typename Dtype>
__global__ void vupscale(Dtype *input, Dtype *output, int64_t no_elements,
                         int scale_factor, int d1, int d2, int d3, int d4)
{
  // output offset:
  int64_t ii = threadIdx.x + blockDim.x * blockIdx.x;
  ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
  if (ii >= no_elements) return;
  int ipidx = translate_idx(ii, d1, d2, d3, d4, scale_factor);
  output[ii]=input[ipidx];
}

/*
 * Description:
 */
template <typename Dtype, typename Acctype>
__global__ void vdownscale(Dtype *gradInput_data, Dtype *gradOutput_data, int64_t no_elements,
                              int scale_factor, int d1, int d2, int d3, int d4)
{
  // output offset:
  int64_t ii = threadIdx.x + blockDim.x * blockIdx.x;
  ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
  if (ii >= no_elements) return;
  Acctype sum = Acctype(0);
  for (int i=0; i < scale_factor; i++){
    for(int j=0; j < scale_factor; j++){
      for(int k=0; k < scale_factor; k++){
        int ipidx = translate_idx_inv(ii, d1, d2, d3, d4, scale_factor, i, j, k);
        sum += gradOutput_data[ipidx];
      }
    }
  }
  gradInput_data[ii] += ScalarConvert<Acctype, Dtype>::to(sum);
}

#include "generic/VolumetricUpSamplingNearest.cu"
#include "THCGenerateFloatTypes.h"
