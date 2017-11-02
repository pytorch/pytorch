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
struct margin_functor
{
  margin_functor(Acctype margin)
    : margin(margin)
  {}

  __host__ __device__ Acctype operator()(const Dtype &x, const Dtype &y) const
  {
    Acctype z = margin - ScalarConvert<Dtype, Acctype>::to(x) * y;
    return z >= 0 ? z : 0;
  }

  const Acctype margin;
};

template <typename Dtype, typename Acctype>
struct margin_updateGradInput_functor
{
  const Acctype margin, norm;

  margin_updateGradInput_functor(Acctype margin_, Acctype norm_)
    : margin(margin_)
    , norm(norm_)
  {}

  __host__ __device__ Dtype operator()(const Dtype &x, const Dtype &y) const
  {
    return ScalarConvert<Acctype, Dtype>::to((ScalarConvert<Dtype, Acctype>::to(x) * y) < margin ? -norm * y : 0);
  }
};

#include "generic/MarginCriterion.cu"
#include "THCGenerateFloatTypes.h"
