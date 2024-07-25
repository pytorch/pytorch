#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <assert.h>

using namespace at;

__device__ void test(){
  // test half construction and implicit conversions in device
  assert(Half(3) == Half(3.0f));
  assert(static_cast<Half>(3.0f) == Half(3.0f));
  // there is no float <=> __half implicit conversion
  assert(static_cast<Half>(3.0f) == 3.0f);

  __half a = __float2half(3.0f);
  __half b = __float2half(2.0f);
  __half c = Half(a) - Half(b);
  assert(static_cast<Half>(c) == Half(1.0));

  // asserting if the  functions used on
  // half types give almost equivalent results when using
  //  functions on double.
  // The purpose of these asserts are to test the device side
  // half API for the common mathematical functions.
  // Note: When calling std math functions from device, don't
  // use the std namespace, but just "::" so that the function
  // gets resolved from nvcc math_functions.hpp

  float threshold = 0.00001;
  assert(::abs(::lgamma(Half(10.0)) - ::lgamma(10.0f)) <= threshold);
  assert(::abs(::exp(Half(1.0)) - ::exp(1.0f)) <= threshold);
  assert(::abs(::log(Half(1.0)) - ::log(1.0f)) <= threshold);
  assert(::abs(::log10(Half(1000.0)) - ::log10(1000.0f)) <= threshold);
  assert(::abs(::log1p(Half(0.0)) - ::log1p(0.0f)) <= threshold);
  assert(::abs(::log2(Half(1000.0)) - ::log2(1000.0f)) <= threshold);
  assert(::abs(::expm1(Half(1.0)) - ::expm1(1.0f)) <= threshold);
  assert(::abs(::cos(Half(0.0)) - ::cos(0.0f)) <= threshold);
  assert(::abs(::sin(Half(0.0)) - ::sin(0.0f)) <= threshold);
  assert(::abs(::sqrt(Half(100.0)) - ::sqrt(100.0f)) <= threshold);
  assert(::abs(::ceil(Half(2.4)) - ::ceil(2.4f)) <= threshold);
  assert(::abs(::floor(Half(2.7)) - ::floor(2.7f)) <= threshold);
  assert(::abs(::trunc(Half(2.7)) - ::trunc(2.7f)) <= threshold);
  assert(::abs(::acos(Half(-1.0)) - ::acos(-1.0f)) <= threshold);
  assert(::abs(::cosh(Half(1.0)) - ::cosh(1.0f)) <= threshold);
  assert(::abs(::acosh(Half(1.0)) - ::acosh(1.0f)) <= threshold);
  assert(::abs(::acosh(Half(1.0)) - ::acosh(1.0f)) <= threshold);
  assert(::abs(::asinh(Half(1.0)) - ::asinh(1.0f)) <= threshold);
  // See note below about VC++ and isinf
#ifndef  _MSC_VER
  assert(::isinf(::atanh(Half(1.0))));
#endif
  assert(::abs(::atanh(Half(.5)) - ::atanh(.5f)) <= threshold);
  assert(::abs(::asin(Half(1.0)) - ::asin(1.0f)) <= threshold);
  assert(::abs(::sinh(Half(1.0)) - ::sinh(1.0f)) <= threshold);
  assert(::abs(::asinh(Half(1.0)) - ::asinh(1.0f)) <= threshold);
  assert(::abs(::tan(Half(0.0)) - ::tan(0.0f)) <= threshold);
  assert(::abs(::atan(Half(1.0)) - ::atan(1.0f)) <= threshold);
  assert(::abs(::tanh(Half(1.0)) - ::tanh(1.0f)) <= threshold);
  assert(::abs(::erf(Half(10.0)) - ::erf(10.0f)) <= threshold);
  assert(::abs(::erfc(Half(10.0)) - ::erfc(10.0f)) <= threshold);
  assert(::abs(::abs(Half(-3.0)) - ::abs(-3.0f)) <= threshold);
  assert(::abs(::round(Half(2.3)) - ::round(2.3f)) <= threshold);
  assert(::abs(::pow(Half(2.0), Half(10.0)) - ::pow(2.0f, 10.0f)) <= threshold);
  assert(
      ::abs(::atan2(Half(7.0), Half(0.0)) - ::atan2(7.0f, 0.0f)) <= threshold);
  // note: can't use  namespace on isnan and isinf in device code

  // Windows requires this explicit conversion. The reason is unclear
  // related issue with clang: https://reviews.llvm.org/D37906
#ifndef _MSC_VER
  assert(::abs(::isnan(Half(0.0)) - ::isnan(0.0f)) <= threshold);
  assert(::abs(::isinf(Half(0.0)) - ::isinf(0.0f)) <= threshold);
#endif

  // test complex<32>
  Half real = 3.0f;
  Half imag = -10.0f;
  auto complex = c10::complex<Half>(real, imag);
  assert(complex.real() == real);
  assert(complex.imag() == imag);
}

__global__ void kernel(){
  test();
}

void launch_function(){
  kernel<<<1, 1>>>();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// half common math functions tests in device
TEST(HalfCuda, HalfCuda) {
  if (!at::cuda::is_available()) return;
  launch_function();
  cudaError_t err = cudaDeviceSynchronize();
  bool isEQ = err == cudaSuccess;
  ASSERT_TRUE(isEQ);
}
