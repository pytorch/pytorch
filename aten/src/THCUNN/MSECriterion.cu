#include <THCUNN/THCUNN.h>
#include <THCUNN/common.h>
#include <TH/THHalf.h>
#include <THC/THCNumerics.cuh>
#include <THC/THCThrustAllocator.cuh>
#include <THC/THCApply.cuh>

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
#include <thrust/system/cuda/execution_policy.h>
#endif

template <typename Dtype, typename Acctype>
struct mse_functor
{
  mse_functor() {}

  __host__ __device__ Acctype operator()(const Dtype &x, const Dtype &y) const
  {
    Acctype z = ScalarConvert<Dtype, Acctype>::to(x)-y;
    return z*z;
  }
};


template <typename Dtype>
struct mse_updateOutput_functor
{
  mse_updateOutput_functor() {}

  __device__ void operator()(
      const Dtype *input,
      const Dtype *target,
      Dtype *output)
  {
    Dtype diff = THCNumerics<Dtype>::sub(*input, *target);
    *output = THCNumerics<Dtype>::mul(diff, diff);
  }
};


template <typename Dtype, typename Acctype>
struct mse_updateGradInput_functor
{
  const Acctype norm;

  mse_updateGradInput_functor(Acctype norm_)
    : norm(norm_)
  {}

  __host__ __device__ Dtype operator()(const Dtype &x, const Dtype &y) const
  {
    return ScalarConvert<Acctype, Dtype>::to(norm * (ScalarConvert<Dtype, Acctype>::to(x) - y));
  }
};

#include <THCUNN/generic/MSECriterion.cu>
#include <THC/THCGenerateFloatTypes.h>
