#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ATen/ATen.h"
#include "ATen/cuda/CUDANumerics.cuh"
#include "cuda.h"
#include "cuda_runtime.h"

#include <assert.h>

using namespace at;

__host__ __device__ void test(){
  // asserting if the std functions specialized for
  // half types give almost equivalent results when using
  // std functions on double.
  // The purpose of these asserts are to test the device side
  // half API for the common mathematical functions.
  
  assert(std::lgamma(Half(10.0)) == Half(std::lgamma(10.0)));
  assert(std::exp(Half(1.0)) == Half(std::exp(1.0)));
  assert(std::log(Half(1.0)) == Half(std::log(1.0)));
  assert(std::log10(Half(1000.0)) == Half(std::log10(1000.0)));
  assert(std::log1p(Half(0.0)) == Half(std::log1p(0.0)));
  assert(std::log2(Half(1000.0)) == Half(std::log2(1000.0)));
  assert(std::expm1(Half(1.0)) == Half(std::expm1(1.0)));
  assert(std::cos(Half(0.0)) == Half(std::cos(0.0)));
  assert(std::sin(Half(0.0)) == Half(std::sin(0.0)));
  assert(std::sqrt(Half(100.0)) == Half(std::sqrt(100.0)));
  assert(std::ceil(Half(2.4)) == Half(std::ceil(2.4)));
  assert(std::floor(Half(2.7)) == Half(std::floor(2.7)));
  assert(std::trunc(Half(2.7)) == Half(std::trunc(2.7)));
  assert(std::acos(Half(-1.0)) == Half(std::acos(-1.0)));
  assert(std::cosh(Half(1.0)) == Half(std::cosh(1.0)));
  assert(std::acosh(Half(1.0)) == Half(std::acosh(1.0)));
  assert(std::asin(Half(1.0)) == Half(std::asin(1.0)));
  assert(std::sinh(Half(1.0)) == Half(std::sinh(1.0)));
  assert(std::asinh(Half(1.0)) == Half(std::asinh(1.0)));
  assert(std::tan(Half(0.0)) == Half(std::tan(0.0)));
  assert(std::atan(Half(1.0)) == Half(std::atan(1.0)));
  assert(std::tanh(Half(1.0)) == Half(std::tanh(1.0)));
  assert(std::erf(Half(10.0)) == Half(std::erf(10.0)));
  assert(std::erfc(Half(10.0)) == Half(std::erfc(10.0)));
  assert(std::abs(Half(-3.0)) == Half(std::abs(-3.0)));
  assert(std::round(Half(2.3)) == Half(std::round(2.3)));
  assert(std::pow(Half(2.0), Half(10.0)) == Half(std::pow(2.0, 10.0)));
  assert(std::atan2(Half(7.0), Half(0.0)) == Half(std::atan2(7.0, 0.0)));
  assert(::isnan(Half(0.0)) == Half(std::isnan(0.0)));
  assert(::isinf(Half(0.0)) == Half(std::isinf(0.0)));

  // test half functions from cuda numerics
  assert(at::numerics<Half>::erfinv(Half(0.25)) == Half(at::numerics<double>::erfinv(0.25)));
  assert(at::numerics<Half>::exp10(Half(2.0)) == Half(at::numerics<double>::exp10(2.0)));
  assert(at::numerics<Half>::rsqrt(Half(100.0)) == Half(at::numerics<double>::rsqrt(100.0)));
  assert(at::numerics<Half>::frac(Half(2.5)) == Half(at::numerics<double>::frac(2.5)));
  assert(at::numerics<Half>::cinv(Half(5.0)) == Half(at::numerics<double>::cinv(5.0)));
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

