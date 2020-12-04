#include <algorithm>
#include <utility>

#include <THCUNN/THCUNN.h>
#include <TH/THHalf.h>
#include <THC/THCNumerics.cuh>
#include <THC/THCApply.cuh>
#include <THCUNN/common.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

// copied from cutorch/lib/THC/THCTensorRandom.cu
#define MAX_NUM_BLOCKS 64
#define BLOCK_SIZE 256
#define NUM_BLOCKS(n) \
  (std::min((int)THCCeilDiv(n, (ptrdiff_t)BLOCK_SIZE), MAX_NUM_BLOCKS))

template<typename T>
inline T __device__ curand_uniform_type(curandStatePhilox4_32_10_t *state);

template <>
inline THHalf __device__ curand_uniform_type<THHalf>(curandStatePhilox4_32_10_t *state) {
  auto rand = curand_uniform4(state);
  return ScalarConvert<float, THHalf>::to(rand.x);
}

template <>
inline float __device__ curand_uniform_type<float>(curandStatePhilox4_32_10_t *state) {
  auto rand = curand_uniform4(state);
  return rand.x;
}

template <>
inline double __device__ curand_uniform_type<double>(curandStatePhilox4_32_10_t *state) {
  auto rand = curand_uniform2_double(state);
  return rand.x;
}

template <typename T>
__global__ void rreluUpdateOutputTrain(int n, std::pair<uint64_t, uint64_t> seeds,
  T *input, T* noise, T *output, double a, double b)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, idx, seeds.second, &state);
  CUDA_KERNEL_LOOP(i, n)
  {
    if (input[i] <= 0)
    {
      T r = curand_uniform_type<T>(&state);
      r = ScalarConvert<double, T>::to(r * (b-a) + a);
      output[i] = input[i] * r;
      noise[i] = r;
    }
    else
    {
      output[i] = input[i];
      noise[i] = ScalarConvert<int, T>::to(1);
    }
  }
}

template <typename T>
struct RReLUUpdateOutputEval_functor
{
  const T negSlope_;

  RReLUUpdateOutputEval_functor(T negSlope)
    : negSlope_(negSlope)
  {}

  __device__ __forceinline__ void operator()(T *out, T *in)
  {
    const T x = *in;
    const T r = x <= 0 ? negSlope_ : ScalarConvert<int, T>::to(1);
    *out = x * r;
  }
};

template <typename T>
struct RReLUUpdateOutputEvalIP_functor
{
  const T negSlope_;

  RReLUUpdateOutputEvalIP_functor(T negSlope)
    : negSlope_(negSlope)
  {}

  __device__ __forceinline__ void operator()(T *x)
  {
    if (*x <= 0)
    {
      *x = *x * negSlope_;
    }
  }
};

template <typename T>
struct RReLUupdateGradInputEval_functor
{
  const T negSlope_;

  RReLUupdateGradInputEval_functor(T negSlope)
    : negSlope_(negSlope)
  {}

  __device__ __forceinline__ void operator()(T *gradIn, T *gradOut, T *in)
  {
    *gradIn = (*in) <= 0 ? (*gradOut) * negSlope_ : (*gradOut);
  }
};

template <typename T>
struct RReLUupdateGradInputEvalIP_functor
{
  const T negSlope_;

  RReLUupdateGradInputEvalIP_functor(T negSlope)
    : negSlope_(negSlope)
  {}

  __device__ __forceinline__ void operator()(T *gradOut, T *in)
  {
    if (*in <= 0)
    {
      *gradOut = (*gradOut) * negSlope_;
    }
  }
};

#include <THCUNN/generic/RReLU.cu>
#include <THC/THCGenerateFloatTypes.h>
