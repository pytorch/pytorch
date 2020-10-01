#include <THCUNN/THCUNN.h>
#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>
#include <THC/THCApply.cuh>

template <typename T>
struct logSigmoid_updateOutput_functor
{
  __device__ void operator()(T *output, const T *input) const {
    const T max = fmaxType(T{0}, -*input);
    const T z = ::exp(-max) + ::exp(-*input -max);
    *output = -(max + static_cast<T>(std::log(z)));
  }
};


template <typename T>
struct logSigmoid_updateGradInput_functor
{
  __device__ void operator()(T *gradInput, const T *input, const T *gradOutput) const {
    const T max = fmaxType(T{0}, -*input);
    const T z = ::exp(-max) + ::exp(-*input -max);
    T max_deriv = 0.f;
    T sign = -1.f;
    if (*input < 0.f){
        max_deriv = -1.f;
        sign = 1.f;
    }
    *gradInput = *gradOutput * (-max_deriv - sign*((z - 1.f)/z));
  }
};

template <>
struct logSigmoid_updateOutput_functor<half> {
  __device__ __forceinline__ void operator()(half* output, const half *input) const {
    float in = __half2float(*input);
    float max = fmaxType(0.f, -in);
    float z = ::exp(-max) + ::exp(-in - max);
    *output = __float2half(-(max + std::log(z)));
  }
};

template <>
struct logSigmoid_updateGradInput_functor<half> {
  __device__ __forceinline__ void operator()(half* gradInput, const half *input, const half *gradOutput) const {
    const float in = __half2float(*input);
    const float max = fmaxType(0.f, -in);
    const float z = ::exp(-max) + ::exp(-in - max);
    const float go = __half2float(*gradOutput);
    float max_deriv = 0.f;
    float sign = -1.f;
    if(in < 0.f){
        max_deriv = -1.f;
        sign = 1.f;
    }
    *gradInput = __float2half(go * (-max_deriv - sign*((z - 1.f)/z)));
  }
};

#include <THCUNN/generic/LogSigmoid.cu>
#include <THC/THCGenerateFloatTypes.h>
