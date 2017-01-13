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
struct smoothl1_updateGradInput_functor
{
  const Dtype norm;

  smoothl1_updateGradInput_functor(Dtype norm_)
    : norm(norm_)
  {}

  __host__ __device__ Dtype operator()(const Dtype &x, const Dtype &y) const
  {
    Dtype z = x - y;
    if (z < ScalarConvert<int, Dtype>::to(-1))
      return -norm;
    else if (z > ScalarConvert<int, Dtype>::to(1))
      return norm;
    else
      return norm * z;
  }
};

#include "generic/SmoothL1Criterion.cu"
#include "THCGenerateFloatTypes.h"
