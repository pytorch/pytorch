#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

#ifdef _MSC_VER
#define ZERO_MACRO zero<T>()
template <typename T>
inline __device__ typename std::enable_if<std::is_same<T, double>::value, T>::type zero() {
	return 0.;
}

template <typename T>
inline __device__ typename std::enable_if<!std::is_same<T, double>::value, T>::type zero() {
	return 0.f;
}
#else
#define ZERO_MACRO 0.f
#endif

template <typename T>
struct logSigmoid_updateOutput_functor
{
  __device__ void operator()(T *output, const T *input) const {
    const T max = fmaxType(ZERO_MACRO, -*input);
    const T z = THCNumerics<T>::exp(-max) + THCNumerics<T>::exp(-*input -max);
    *output = -(max + THCNumerics<T>::log(z));
  }
};


template <typename T>
struct logSigmoid_updateGradInput_functor
{
  __device__ void operator()(T *gradInput, const T *input, const T *gradOutput) const {
    const T max = fmaxType(ZERO_MACRO, -*input);
    const T z = THCNumerics<T>::exp(-max) + THCNumerics<T>::exp(-*input -max);
    T max_deriv = 0.f;
    T sign = -1.f;
    if (*input < 0.f){
        max_deriv = -1.f;
        sign = 1.f;
    }
    *gradInput = *gradOutput * (-max_deriv - sign*((z - 1.f)/z));
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct logSigmoid_updateOutput_functor<half> {
  __device__ __forceinline__ void operator()(half* output, const half *input) const {
#ifdef CUDA_HALF_INSTRUCTIONS
    const half max = fmaxType(__float2half(0.f), __hneg(*input));
    const half z = THCNumerics<half>::exp(__hneg(max)) + THCNumerics<half>::exp(__hneg(*input) - max);
    *output = __hneg(max + THCNumerics<half>::log(z));
#else
    float in = __half2float(*input);
    float max = fmaxType(0.f, -in);
    float z = THCNumerics<float>::exp(-max) + THCNumerics<float>::exp(-in - max);
    *output = __float2half(-(max + THCNumerics<float>::log(z)));
#endif
  }
};

template <>
struct logSigmoid_updateGradInput_functor<half> {
  __device__ __forceinline__ void operator()(half* gradInput, const half *input, const half *gradOutput) const {
#ifdef CUDA_HALF_INSTRUCTIONS
    const half one = __float2half(1.f);
    const half zero = __float2half(0.f);
    const half max = fmaxType(zero, __hneg(*input));
    const half z = THCNumerics<half>::exp(__hneg(max)) + THCNumerics<half>::exp(__hneg(*input) - max);
    half max_deriv = zero;
    half sign = __hneg(one);
    if(*input < zero){
        max_deriv = __hneg(one);
        sign = one;
    }
    *gradInput = __hmul(*gradOutput, (__hneg(max_deriv) - __hmul(sign, __hdiv(z - one, z))));
#else
    const float in = __half2float(*input);
    const float max = fmaxType(0.f, -in);
    const float z = THCNumerics<float>::exp(-max) + THCNumerics<float>::exp(-in - max);
    const float go = __half2float(*gradOutput);
    float max_deriv = 0.f;
    float sign = -1.f;
    if(in < 0.f){
        max_deriv = -1.f;
        sign = 1.f;
    }
    *gradInput = __float2half(go * (-max_deriv - sign*((z - 1.f)/z)));
#endif
  }
};
#endif

#include "generic/LogSigmoid.cu"
#include "THCGenerateFloatTypes.h"
