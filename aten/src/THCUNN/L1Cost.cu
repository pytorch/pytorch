#include <THCUNN/THCUNN.h>
#include <THCUNN/common.h>
#include <TH/THHalf.h>
#include <THC/THCNumerics.cuh>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

template <typename Dtype, typename Acctype>
struct l1cost_functor
{
  __host__ __device__ Acctype operator()(Dtype x) const
  {
    return THCNumerics<Acctype>::abs(ScalarConvert<Dtype, Acctype>::to(x));
  }
};

template <typename Dtype>
struct l1cost_updateGradInput_functor
{
  __host__ __device__ Dtype operator()(Dtype x) const
  {
    if (x > 0)
      return ScalarConvert<int, Dtype>::to(1);
    else if (x < 0)
      return ScalarConvert<int, Dtype>::to(-1);
    else
      return ScalarConvert<int, Dtype>::to(0);
  }
};

#include <THCUNN/generic/L1Cost.cu>
#include <THC/THCGenerateFloatTypes.h>
