#include <THCUNN/THCUNN.h>
#include <THCUNN/common.h>
#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>
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
struct smoothl1_functor
{
  const Dtype beta;

  smoothl1_functor(Dtype beta_) : beta(beta_) {}

  __host__ __device__ Acctype operator()(const Dtype &x, const Dtype &y) const
  {
    Acctype z = ScalarConvert<Dtype, Acctype>::to(THCNumerics<Dtype>::abs(x-y));
    return z < Acctype(beta) ? 0.5f*z*z / beta : z - 0.5f * beta;
  }
};

template <typename Dtype>
struct smoothl1_updateOutput_no_reduce_functor
{
  const Dtype beta;

  smoothl1_updateOutput_no_reduce_functor(Dtype beta_) : beta(beta_) {}

  __forceinline__ __host__ __device__ void operator()(
      const Dtype *x,
      const Dtype *y,
      Dtype *out) const
  {
    Dtype beta = ScalarConvert<double, Dtype>::to(beta);
    Dtype oneHalf = ScalarConvert<float, Dtype>::to(0.5f);
    Dtype z = THCNumerics<Dtype>::abs(*x - *y);
    *out = z < beta ? oneHalf * z * z / beta : z - oneHalf * beta;
  }
};

template <typename Dtype>
struct smoothl1_updateGradInput_no_reduce_functor
{
  const Dtype beta;
  smoothl1_updateGradInput_no_reduce_functor(Dtype beta_) : beta(beta_) {}

  __host__ __device__ void operator()(
      const Dtype *x,
      const Dtype *y,
      Dtype *gradInput) const
  {
    Dtype z = *x - *y;
    Dtype beta = ScalarConvert<double, Dtype>::to(beta);
    Dtype minusBeta = ScalarConvert<double, Dtype>::to(-beta);
    Dtype one = ScalarConvert<int, Dtype>::to(1);
    Dtype minusOne = ScalarConvert<int, Dtype>::to(-1);
    if (z < minusBeta) {
      *gradInput = minusOne;
    } else if (z > beta) {
      *gradInput = one;
    } else {
      *gradInput = z / beta;
    }
  }
};

template <typename Dtype>
struct smoothl1_updateGradInput_functor
{
  const Dtype norm;
  const Dtype gradOutput;
  const Dtype beta;

  smoothl1_updateGradInput_functor(Dtype norm_, Dtype gradOutput_, Dtype beta_)
    : norm(norm_), gradOutput(gradOutput_), beta(beta_)
  {}

  __host__ __device__ Dtype operator()(const Dtype &x, const Dtype &y) const
  {
    Dtype z = x - y;
    if (z < ScalarConvert<double, Dtype>::to(-beta))
      return -norm * gradOutput;
    else if (z > ScalarConvert<double, Dtype>::to(beta))
      return norm * gradOutput;
    else
      return norm * z * gradOutput / beta;
  }
};

#include <THCUNN/generic/SmoothL1Criterion.cu>
#include <THC/THCGenerateFloatTypes.h>
