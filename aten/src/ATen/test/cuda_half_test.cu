#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ATen/ATen.h"
#include "ATen/cuda/CUDANumerics.cuh"
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"

#include <assert.h>

using namespace at;

__host__ __device__ void test(){
  
  // test half construction and implicit conversions in device
  assert(Half(3) == Half(3.0f));
  assert(static_cast<Half>(3.0f) == Half(3.0f));
  // there is no float <=> __half implicit conversion
  assert(static_cast<Half>(3.0f) == 3.0f);

  // asserting if the std functions used on 
  // half types give almost equivalent results when using
  // std functions on double.
  // The purpose of these asserts are to test the device side
  // half API for the common mathematical functions.

  float threshold = 0.00001;
  assert(std::abs(std::lgamma(Half(10.0)) - std::lgamma(10.0f)) <= threshold);
  assert(std::abs(std::exp(Half(1.0)) - std::exp(1.0f)) <= threshold);
  assert(std::abs(std::log(Half(1.0)) - std::log(1.0f)) <= threshold);
  assert(std::abs(std::log10(Half(1000.0)) - std::log10(1000.0f)) <= threshold);
  assert(std::abs(std::log1p(Half(0.0)) - std::log1p(0.0f)) <= threshold);
  assert(std::abs(std::log2(Half(1000.0)) - std::log2(1000.0f)) <= threshold);
  assert(std::abs(std::expm1(Half(1.0)) - std::expm1(1.0f)) <= threshold);
  assert(std::abs(std::cos(Half(0.0)) - std::cos(0.0f)) <= threshold);
  assert(std::abs(std::sin(Half(0.0)) - std::sin(0.0f)) <= threshold);
  assert(std::abs(std::sqrt(Half(100.0)) - std::sqrt(100.0f)) <= threshold);
  assert(std::abs(std::ceil(Half(2.4)) - std::ceil(2.4f)) <= threshold);
  assert(std::abs(std::floor(Half(2.7)) - std::floor(2.7f)) <= threshold);
  assert(std::abs(std::trunc(Half(2.7)) - std::trunc(2.7f)) <= threshold);
  assert(std::abs(std::acos(Half(-1.0)) - std::acos(-1.0f)) <= threshold);
  assert(std::abs(std::cosh(Half(1.0)) - std::cosh(1.0f)) <= threshold);
  assert(std::abs(std::acosh(Half(1.0)) - std::acosh(1.0f)) <= threshold);
  assert(std::abs(std::asin(Half(1.0)) - std::asin(1.0f)) <= threshold);
  assert(std::abs(std::sinh(Half(1.0)) - std::sinh(1.0f)) <= threshold);
  assert(std::abs(std::asinh(Half(1.0)) - std::asinh(1.0f)) <= threshold);
  assert(std::abs(std::tan(Half(0.0)) - std::tan(0.0f)) <= threshold);
  assert(std::abs(std::atan(Half(1.0)) - std::atan(1.0f)) <= threshold);
  assert(std::abs(std::tanh(Half(1.0)) - std::tanh(1.0f)) <= threshold);
  assert(std::abs(std::erf(Half(10.0)) - std::erf(10.0f)) <= threshold);
  assert(std::abs(std::erfc(Half(10.0)) - std::erfc(10.0f)) <= threshold);
  assert(std::abs(std::abs(Half(-3.0)) - std::abs(-3.0f)) <= threshold);
  assert(std::abs(std::round(Half(2.3)) - std::round(2.3f)) <= threshold);
  assert(std::abs(std::pow(Half(2.0), Half(10.0)) - std::pow(2.0f, 10.0f)) <= threshold);
  assert(std::abs(std::atan2(Half(7.0), Half(0.0)) - std::atan2(7.0f, 0.0f)) <= threshold);
  // note: can't use std namespace on isnan and isinf in device code
  assert(std::abs(::isnan(Half(0.0)) - ::isnan(0.0f)) <= threshold);
  assert(std::abs(::isinf(Half(0.0)) - ::isinf(0.0f)) <= threshold);
}

__global__ void kernel(){
  test();
}

void launch_function(){
  kernel<<<1,1>>>();
}

TEST_CASE( "half common math functions tests in device", "[cuda]" ) {
  launch_function();
  cudaError_t err = cudaDeviceSynchronize();
  REQUIRE(err == cudaSuccess);
}

