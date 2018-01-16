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
struct abs_updateOutput_no_reduce_functor
{
  __host__ __device__ void operator()(const Dtype* x, const Dtype* y, Dtype *out)
  {
    Dtype z = *x - *y;
    *out = z >= 0 ? z : -z;
  }
};

template <typename Dtype>
struct abs_updateGradInput_no_reduce_functor
{
  __forceinline__ __host__ __device__ void operator()(
      const Dtype *x,
      const Dtype *y,
      Dtype *gradInput)
  {
    *gradInput = ScalarConvert<int, Dtype>::to(*x >= *y ? 1 : -1);
  }
};

template <typename Dtype>
struct abs_updateGradInput_functor
{
  const Dtype norm;
  const Dtype gradOutput;

  abs_updateGradInput_functor(Dtype norm_, Dtype gradOutput_)
    : norm(norm_), gradOutput(gradOutput_)
  {}

  __host__ __device__ Dtype operator()(const Dtype& x, const Dtype& y) const
  {
    return ((x - y) >= 0 ? norm : -norm) * gradOutput;
  }
};

#include "generic/AbsCriterion.cu"
#include "THCGenerateFloatTypes.h"
