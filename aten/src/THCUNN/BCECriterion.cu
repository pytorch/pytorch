#include <THCUNN/THCUNN.h>
#include <THCUNN/common.h>
#include <TH/THHalf.h>
#include <THC/THCNumerics.cuh>
#include <THC/THCThrustAllocator.cuh>
#include <THC/THCApply.cuh>

#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/system/cuda/execution_policy.h>

template <typename T>
inline __host__ __device__ T eps();

template <>
inline __host__ __device__ float eps() { return 1e-12f; }

template <>
inline __host__ __device__ double eps() { return 1e-12; }

template <typename T>
inline __host__ __device__ T safe_log(T a) {
  if (a == 0.)
  {
    return std::log(eps<T>());
  }
  return std::log(a);
}

template <typename Dtype, typename Acctype>
struct bce_functor
{
  template <class Tuple>
  __host__ __device__
  Acctype operator()(Tuple x)
  {
    Dtype input = thrust::get<0>(x);
    Dtype t = thrust::get<1>(x);
    assert(input >= 0. && input <= 1.);
    return - (t * safe_log<Acctype>(ScalarConvert<Dtype, Acctype>::to(input))
        + (Acctype(1) - t) * safe_log<Acctype>(Acctype(1) - input));
  }
};

template <typename Dtype, typename Acctype>
struct bce_updateOutput_no_reduce_functor
{
  __forceinline__ __host__ __device__
  void operator()(
      const Dtype *input,
      const Dtype *target,
      Dtype *output)
  {
    assert(*input >= 0. && *input <= 1.);
    *output = ScalarConvert<Acctype, Dtype>::to(
        -(*target * safe_log<Acctype>(ScalarConvert<Dtype, Acctype>::to(*input)) +
          (Acctype(1) - *target) * safe_log<Acctype>(Acctype(1) - *input)));
  }
};

template <typename Dtype, typename Acctype>
struct bce_functor_weights
{
  template <class Tuple>
  __host__ __device__
  Acctype operator()(Tuple x)
  {
    Dtype input = thrust::get<0>(x);
    Dtype t = thrust::get<1>(x);
    Dtype w = thrust::get<2>(x);
    assert(input >= 0. && input <= 1.);
    return - w * (t * safe_log<Acctype>(ScalarConvert<Dtype, Acctype>::to(input)) +
        (Acctype(1) - t) * safe_log<Acctype>(Acctype(1) - input));
  }
};

template <typename Dtype, typename Acctype>
struct bce_updateGradInput_no_reduce_functor
{
  __forceinline__ __host__ __device__
  void operator()(
      const Dtype *x,
      const Dtype *t,
      Dtype *gradInput)
  {
      *gradInput = ScalarConvert<Acctype,Dtype>::to(
          - (*t - *x) / ((Acctype(1) - *x + eps<Acctype>()) * (*x + eps<Acctype>())));
  }
};

template <typename Dtype, typename Acctype>
struct bce_updateGradInput_functor
{
  const Dtype norm;

  bce_updateGradInput_functor(Dtype norm_)
    : norm(norm_)
  {}

  template <class Tuple>
  __host__ __device__
  Dtype operator()(Tuple x)
  {
    Dtype o = thrust::get<0>(x);
    Dtype t = thrust::get<1>(x);
    return ScalarConvert<Acctype,Dtype>::to(- (t - o) / ((Acctype(1) - o + eps<Acctype>()) * (o + eps<Acctype>())) * norm);
  }
};

template <typename Dtype, typename Acctype>
struct bce_updateGradInput_functor_weights
{
  const Dtype norm;

  bce_updateGradInput_functor_weights(Dtype norm_)
    : norm(norm_)
  {}

  template <class Tuple>
  __host__ __device__
  Dtype operator()(Tuple x)
  {
    Dtype o = thrust::get<0>(x);
    Dtype t = thrust::get<1>(x);
    Dtype w = thrust::get<2>(x);
    return ScalarConvert<Acctype, Dtype>::to(- (t - o) / ((Acctype(1) - o + eps<Acctype>()) * (o + eps<Acctype>())) * norm * w);
  }
};

#include <THCUNN/generic/BCECriterion.cu>
#include <THC/THCGenerateFloatTypes.h>
