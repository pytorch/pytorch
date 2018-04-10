#pragma once

#include <ATen/Error.h>
#include <ATen/Half.h>
#include <ATen/Type.h>

#define AT_PRIVATE_CASE_TYPE(enum_type, type, ...) \
  case enum_type: {                                \
    using scalar_t = type;                         \
    return __VA_ARGS__();                          \
  }

#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                           \
  [&] {                                                                       \
    const at::Type& the_type = TYPE;                                          \
    switch (the_type.scalarType()) {                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)         \
      default:                                                                \
        AT_ERROR("%s not implemented for '%s'", (NAME), the_type.toString()); \
    }                                                                         \
  }()

#define AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...)                  \
  [&] {                                                                       \
    const at::Type& the_type = TYPE;                                          \
    switch (the_type.scalarType()) {                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, Half, __VA_ARGS__)           \
      default:                                                                \
        AT_ERROR("%s not implemented for '%s'", (NAME), the_type.toString()); \
    }                                                                         \
  }()

#define AT_DISPATCH_ALL_TYPES(TYPE, NAME, ...)                                \
  [&] {                                                                       \
    const at::Type& the_type = TYPE;                                          \
    switch (the_type.scalarType()) {                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)       \
      default:                                                                \
        AT_ERROR("%s not implemented for '%s'", (NAME), the_type.toString()); \
    }                                                                         \
  }()

#define AT_DISPATCH_ALL_TYPES_AND_HALF(TYPE, NAME, ...)                       \
  [&] {                                                                       \
    const at::Type& the_type = TYPE;                                          \
    switch (the_type.scalarType()) {                                          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, Half, __VA_ARGS__)           \
      default:                                                                \
        AT_ERROR("%s not implemented for '%s'", (NAME), the_type.toString()); \
    }                                                                         \
  }()
