// Macros for type dispatch and common utilities for symmetric memory
#pragma once

#include <ATen/ATen.h>

// Convert ATen floating point types to NV floating point types
// at::kBFloat16 -> __nv_bfloat16
// at::kHalf -> __half
// Float is the same.

#define AT_DISPATCH_CASE_CONVERT(enum_type, scalar_type, ...) \
  case enum_type: {                                           \
    AT_PRIVATE_CHECK_SELECTIVE_BUILD(enum_type);              \
    using scalar_t = scalar_type;                             \
    return __VA_ARGS__();                                     \
  }

#define AT_DISPATCH_NV_FLOATS(scalar_type, name, ...)                      \
  AT_DISPATCH_SWITCH(                                                      \
      scalar_type,                                                         \
      name,                                                                \
      AT_DISPATCH_CASE_CONVERT(at::kBFloat16, __nv_bfloat16, __VA_ARGS__); \
      AT_DISPATCH_CASE_CONVERT(at::kHalf, __half, __VA_ARGS__);            \
      AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__));
