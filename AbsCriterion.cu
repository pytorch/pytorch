#include "THCUNN.h"
#include "common.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>

template <typename Dtype, typename Acctype>
struct abs_functor
{
  __host__ __device__ Acctype operator()(const Dtype& x, const Dtype& y) const
  {
    Dtype z = x-y;
    return ScalarConvert<Dtype, Acctype>::to(z >= 0 ? z : -z);
  }
};

template <typename Dtype>
struct abs_updateGradInput_functor
{
  const Dtype norm;

  abs_updateGradInput_functor(Dtype norm_)
    : norm(norm_)
  {}

  __host__ __device__ Dtype operator()(const Dtype& x, const Dtype& y) const
  {
    return (x - y) >= 0 ? norm : -norm;
  }
};

#include "generic/AbsCriterion.cu"
#include "THCGenerateFloatTypes.h"
