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
struct softmargin_functor
{
  __host__ __device__ Acctype operator()(const Dtype& x, const Dtype& y) const
  {
    return log(1 + exp(ScalarConvert<Dtype, Acctype>::to(-x)*y));
  }
};

template <typename Dtype, typename Acctype>
struct softmargin_no_reduce_functor
{
  __host__ __device__ void operator()(
    const Dtype *x,
    const Dtype *y,
    Dtype *out) const
  {
    *out = ScalarConvert<Acctype, Dtype>::to(log(ScalarConvert<int, Acctype>::to(1)
                                             + exp(ScalarConvert<Dtype, Acctype>::to(-*x) * *y)));
  }
};

template <typename Dtype, typename Acctype>
struct softmargin_updateGradInput_functor
{
  const Acctype norm;
  const Dtype gradOutput;

  softmargin_updateGradInput_functor(Acctype norm_, Dtype gradOutput_) :
    norm(norm_), gradOutput(gradOutput_) {}

  __host__ __device__ Dtype operator()(const Dtype& x, const Dtype& y) const
    {
      Acctype temp = exp(ScalarConvert<Dtype, Acctype>::to(-x)*y);
      return ScalarConvert<Acctype, Dtype>::to(-y*temp*norm/(ScalarConvert<int, Acctype>::to(1) + temp) * gradOutput);
    }
};

template <typename Dtype, typename Acctype>
struct softmargin_updateGradInput_no_reduce_functor
{
  __forceinline__ __host__ __device__ void operator()(
      const Dtype *x,
      const Dtype *y,
      Dtype *gradInput) const
  {
      Acctype temp = exp(ScalarConvert<Dtype, Acctype>::to(-*x) * *y);
      *gradInput = ScalarConvert<Acctype, Dtype>::to(-*y * temp / (ScalarConvert<int, Acctype>::to(1) + temp));
  }
};

#include "generic/SoftMarginCriterion.cu"
#include "THCGenerateFloatTypes.h"
