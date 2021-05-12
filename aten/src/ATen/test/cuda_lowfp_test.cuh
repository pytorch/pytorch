#pragma once

template<template T>
__device__ void test(){
  // test half construction and implicit conversions in device
  assert(T(3) == T(3.0f));
  assert(static_cast<T>(3.0f) == T(3.0f));
  // there is no float <=> __half/__nv_bfloat16 implicit conversion
  assert(static_cast<T>(3.0f) == 3.0f);

  // asserting if the functions used on
  // half types give almost equivalent results when using
  //  functions on double.
  // The purpose of these asserts are to test the device side
  // half API for the common mathematical functions.
  // Note: When calling std math functions from device, don't
  // use the std namespace, but just "::" so that the function
  // gets resolved from nvcc math_functions.hpp

  float threshold = 0.00001;
  assert(::abs(::lgamma(T(10.0)) - ::lgamma(10.0f)) <= threshold);
  assert(::abs(::exp(T(1.0)) - ::exp(1.0f)) <= threshold);
  assert(::abs(::log(T(1.0)) - ::log(1.0f)) <= threshold);
  assert(::abs(::log10(T(1000.0)) - ::log10(1000.0f)) <= threshold);
  assert(::abs(::log1p(T(0.0)) - ::log1p(0.0f)) <= threshold);
  assert(::abs(::log2(T(1000.0)) - ::log2(1000.0f)) <= threshold);
  assert(::abs(::expm1(T(1.0)) - ::expm1(1.0f)) <= threshold);
  assert(::abs(::cos(T(0.0)) - ::cos(0.0f)) <= threshold);
  assert(::abs(::sin(T(0.0)) - ::sin(0.0f)) <= threshold);
  assert(::abs(::sqrt(T(100.0)) - ::sqrt(100.0f)) <= threshold);
  assert(::abs(::ceil(T(2.4)) - ::ceil(2.4f)) <= threshold);
  assert(::abs(::floor(T(2.7)) - ::floor(2.7f)) <= threshold);
  assert(::abs(::trunc(T(2.7)) - ::trunc(2.7f)) <= threshold);
  assert(::abs(::acos(T(-1.0)) - ::acos(-1.0f)) <= threshold);
  assert(::abs(::cosh(T(1.0)) - ::cosh(1.0f)) <= threshold);
  assert(::abs(::acosh(T(1.0)) - ::acosh(1.0f)) <= threshold);
  assert(::abs(::acosh(T(1.0)) - ::acosh(1.0f)) <= threshold);
  assert(::abs(::asinh(T(1.0)) - ::asinh(1.0f)) <= threshold);
  assert(::abs(::atanh(T(0.5)) - ::atanh(0.5f)) <= threshold);
  assert(::abs(::asin(T(1.0)) - ::asin(1.0f)) <= threshold);
  assert(::abs(::sinh(T(1.0)) - ::sinh(1.0f)) <= threshold);
  assert(::abs(::asinh(T(1.0)) - ::asinh(1.0f)) <= threshold);
  assert(::abs(::tan(T(0.0)) - ::tan(0.0f)) <= threshold);
  assert(::abs(::atan(T(1.0)) - ::atan(1.0f)) <= threshold);
  assert(::abs(::tanh(T(1.0)) - ::tanh(1.0f)) <= threshold);
  assert(::abs(::erf(T(10.0)) - ::erf(10.0f)) <= threshold);
  assert(::abs(::erfc(T(10.0)) - ::erfc(10.0f)) <= threshold);
  assert(::abs(::abs(T(-3.0)) - ::abs(-3.0f)) <= threshold);
  assert(::abs(::round(T(2.3)) - ::round(2.3f)) <= threshold);
  assert(::abs(::pow(T(2.0), T(10.0)) - ::pow(2.0f, 10.0f)) <= threshold);
  assert(
      ::abs(::atan2(T(7.0), T(0.0)) - ::atan2(7.0f, 0.0f)) <= threshold);
  // note: can't use  namespace on isnan and isinf in device code
#ifdef _MSC_VER
  // Windows requires this explicit conversion. The reason is unclear
  // related issue with clang: https://reviews.llvm.org/D37906
  assert(::abs(::isnan((float)T(0.0)) - ::isnan(0.0f)) <= threshold);
  assert(::abs(::isinf((float)T(0.0)) - ::isinf(0.0f)) <= threshold);
#else
  assert(::abs(::isnan(T(0.0)) - ::isnan(0.0f)) <= threshold);
  assert(::abs(::isinf(T(0.0)) - ::isinf(0.0f)) <= threshold);
#endif
}