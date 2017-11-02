#include "THCUNN.h"
#include "common.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCThrustAllocator.cuh"

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

template <typename Dtype, typename Acctype>
struct smoothl1_functor
{
  smoothl1_functor() {}

  __host__ __device__ Acctype operator()(const Dtype &x, const Dtype &y) const
  {
    Acctype z = ScalarConvert<Dtype, Acctype>::to(THCNumerics<Dtype>::abs(x-y));
    return z < Acctype(1) ? 0.5f*z*z : z - 0.5f;
  }
};

template <typename Dtype>
struct smoothl1_updateOutput_no_reduce_functor
{
  smoothl1_updateOutput_no_reduce_functor() {}

  __forceinline__ __host__ __device__ void operator()(
      const Dtype *x, 
      const Dtype *y,
      Dtype *out) const
  {
    Dtype oneHalf = ScalarConvert<float, Dtype>::to(0.5f);
    Dtype z = THCNumerics<Dtype>::abs(*x - *y);
    *out = z < ScalarConvert<int, Dtype>::to(1) ? oneHalf * z * z : z - oneHalf;
  }
};

template <typename Dtype>
struct smoothl1_updateGradInput_no_reduce_functor
{
  smoothl1_updateGradInput_no_reduce_functor() {}

  __host__ __device__ void operator()(
      const Dtype *x, 
      const Dtype *y,
      Dtype *gradInput) const
  {
    Dtype z = *x - *y;
    Dtype one = ScalarConvert<int, Dtype>::to(1);
    Dtype minusOne = ScalarConvert<int, Dtype>::to(-1);
    if (z < minusOne) {
      *gradInput = minusOne;
    } else if (z > one) {
      *gradInput = one;
    } else {
      *gradInput = z;
    }
  }
};

template <typename Dtype>
struct smoothl1_updateGradInput_functor
{
  const Dtype norm;
  const Dtype gradOutput;

  smoothl1_updateGradInput_functor(Dtype norm_, Dtype gradOutput_)
    : norm(norm_), gradOutput(gradOutput_)
  {}

  __host__ __device__ Dtype operator()(const Dtype &x, const Dtype &y) const
  {
    Dtype z = x - y;
    if (z < ScalarConvert<int, Dtype>::to(-1))
      return -norm * gradOutput;
    else if (z > ScalarConvert<int, Dtype>::to(1))
      return norm * gradOutput;
    else
      return norm * z * gradOutput;
  }
};

#include "generic/SmoothL1Criterion.cu"
#include "THCGenerateFloatTypes.h"
