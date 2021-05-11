#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/NumericLimits.cuh>

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <assert.h>

using namespace at;

__device__ void test(){
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  // test bfloat16 construction and implicit conversions in device
  assert(BFloat16(3) == BFloat16(3.0f));
  assert(static_cast<BFloat16>(3.0f) == BFloat16(3.0f));
  // there is no float <=> __nv_bfloat16 implicit conversion
  assert(static_cast<BFloat16>(3.0f) == 3.0f);

  __nv_bfloat16 a = __float2bfloat16(3.0f);
  __nv_bfloat16 b = __float2bfloat16(2.0f);
  __nv_bfloat16 c = a - BFloat16(b);
  assert(static_cast<BFloat16>(c) == BFloat16(1.0));

  // asserting if the functions used on
  // bfloat16 types give almost equivalent results when using
  //  functions on double.
  // The purpose of these asserts are to test the device side
  // bfloat16 API for the common mathematical functions.
  // Note: When calling std math functions from device, don't
  // use the std namespace, but just "::" so that the function
  // gets resolved from nvcc math_functions.hpp

  float threshold = 0.00001;
  assert(::abs(::lgamma(BFloat16(10.0)) - ::lgamma(10.0f)) <= threshold);
  assert(::abs(::exp(BFloat16(1.0)) - ::exp(1.0f)) <= threshold);
  assert(::abs(::log(BFloat16(1.0)) - ::log(1.0f)) <= threshold);
  assert(::abs(::log10(BFloat16(1000.0)) - ::log10(1000.0f)) <= threshold);
  assert(::abs(::log1p(BFloat16(0.0)) - ::log1p(0.0f)) <= threshold);
  assert(::abs(::log2(BFloat16(1000.0)) - ::log2(1000.0f)) <= threshold);
  assert(::abs(::expm1(BFloat16(1.0)) - ::expm1(1.0f)) <= threshold);
  assert(::abs(::cos(BFloat16(0.0)) - ::cos(0.0f)) <= threshold);
  assert(::abs(::sin(BFloat16(0.0)) - ::sin(0.0f)) <= threshold);
  assert(::abs(::sqrt(BFloat16(100.0)) - ::sqrt(100.0f)) <= threshold);
  assert(::abs(::ceil(BFloat16(2.4)) - ::ceil(2.4f)) <= threshold);
  assert(::abs(::floor(BFloat16(2.7)) - ::floor(2.7f)) <= threshold);
  assert(::abs(::trunc(BFloat16(2.7)) - ::trunc(2.7f)) <= threshold);
  assert(::abs(::acos(BFloat16(-1.0)) - ::acos(-1.0f)) <= threshold);
  assert(::abs(::cosh(BFloat16(1.0)) - ::cosh(1.0f)) <= threshold);
  assert(::abs(::acosh(BFloat16(1.0)) - ::acosh(1.0f)) <= threshold);
  assert(::abs(::acosh(BFloat16(1.0)) - ::acosh(1.0f)) <= threshold);
  assert(::abs(::asinh(BFloat16(1.0)) - ::asinh(1.0f)) <= threshold);
  assert(::abs(::atanh(BFloat16(1.0)) - ::atanh(1.0f)) <= threshold);
  assert(::abs(::asin(BFloat16(1.0)) - ::asin(1.0f)) <= threshold);
  assert(::abs(::sinh(BFloat16(1.0)) - ::sinh(1.0f)) <= threshold);
  assert(::abs(::asinh(BFloat16(1.0)) - ::asinh(1.0f)) <= threshold);
  assert(::abs(::tan(BFloat16(0.0)) - ::tan(0.0f)) <= threshold);
  assert(::abs(::atan(BFloat16(1.0)) - ::atan(1.0f)) <= threshold);
  assert(::abs(::tanh(BFloat16(1.0)) - ::tanh(1.0f)) <= threshold);
  assert(::abs(::erf(BFloat16(10.0)) - ::erf(10.0f)) <= threshold);
  assert(::abs(::erfc(BFloat16(10.0)) - ::erfc(10.0f)) <= threshold);
  assert(::abs(::abs(BFloat16(-3.0)) - ::abs(-3.0f)) <= threshold);
  assert(::abs(::round(BFloat16(2.3)) - ::round(2.3f)) <= threshold);
  assert(::abs(::pow(BFloat16(2.0), BFloat16(10.0)) - ::pow(2.0f, 10.0f)) <= threshold);
  assert(
      ::abs(::atan2(BFloat16(7.0), BFloat16(0.0)) - ::atan2(7.0f, 0.0f)) <= threshold);
  // note: can't use  namespace on isnan and isinf in device code
#ifdef _MSC_VER
  // Windows requires this explicit conversion. The reason is unclear
  // related issue with clang: https://reviews.llvm.org/D37906
  assert(::abs(::isnan((float)BFloat16(0.0)) - ::isnan(0.0f)) <= threshold);
  assert(::abs(::isinf((float)BFloat16(0.0)) - ::isinf(0.0f)) <= threshold);
#else
  assert(::abs(::isnan(BFloat16(0.0)) - ::isnan(0.0f)) <= threshold);
  assert(::abs(::isinf(BFloat16(0.0)) - ::isinf(0.0f)) <= threshold);
#endif
// #endif
}

__global__ void kernel(){
  test();
}

void launch_function(){
  kernel<<<1, 1>>>();
}

// bfloat16 common math functions tests in device
TEST(BFloat16Cuda, BFloat16Cuda) {
  if (!at::cuda::is_available()) return;
  launch_function();
  cudaError_t err = cudaDeviceSynchronize();
  bool isEQ = err == cudaSuccess;
  ASSERT_TRUE(isEQ);
}
#endif
