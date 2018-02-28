#pragma once

#include <ATen/ATenAssert.h>
#include <ATen/Dispatch.h>
#include <ATen/Type.h>
#include <ATen/cuda/CUDATensorMethods.cuh> // For half

#define AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, function)        \
  [&] {                                                                  \
    const at::Type& the_type = TYPE;                                     \
    switch (the_type.scalarType()) {                                     \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, function)     \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, function)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, half, function)         \
      default:                                                           \
        at::runtime_error(                                               \
            "%s not implemented for '%s'", (NAME), the_type.toString()); \
    }                                                                    \
  }()

#define AT_DISPATCH_ALL_TYPES_AND_HALF(TYPE, NAME, function)             \
  [&] {                                                                  \
    const at::Type& the_type = TYPE;                                     \
    switch (the_type.scalarType()) {                                     \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, function)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, uint8_t, function)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, function)     \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, function)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, function)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, function)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, function)     \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, half, function)         \
      default:                                                           \
        at::runtime_error(                                               \
            "%s not implemented for '%s'", (NAME), the_type.toString()); \
    }                                                                    \
  }()
