#pragma once

#include <ATen/ATenAssert.h>
#include <ATen/Half.h>
#include <ATen/Type.h>

#define AT_PRIVATE_CASE_TYPE(enum_type, type, function) \
  case enum_type: {                                     \
    using scalar_t = type;                              \
    return function();                                  \
  }

#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, function)                 \
  [&] {                                                                  \
    const at::Type& the_type = TYPE;                                     \
    switch (the_type.scalarType()) {                                     \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, function)     \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, function)       \
      default:                                                           \
        at::runtime_error(                                               \
            "%s not implemented for '%s'", (NAME), the_type.toString()); \
    }                                                                    \
  }()

#define AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, function)        \
  [&] {                                                                  \
    const at::Type& the_type = TYPE;                                     \
    switch (the_type.scalarType()) {                                     \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, function)     \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, function)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, Half, function)         \
      default:                                                           \
        at::runtime_error(                                               \
            "%s not implemented for '%s'", (NAME), the_type.toString()); \
    }                                                                    \
  }()

#define AT_DISPATCH_ALL_TYPES(TYPE, NAME, function)                      \
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
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, Half, function)         \
      default:                                                           \
        at::runtime_error(                                               \
            "%s not implemented for '%s'", (NAME), the_type.toString()); \
    }                                                                    \
  }()
