#pragma once

#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/complex.h>

#ifdef __CUDACC__
#include <cuda.h> // For CUDA_VERSION
#endif

#ifdef TEMPLATE_SELECTIVE_BUILD
#include <ATen/selected_mobile_ops.h>
#else
namespace at {
/**
 * The method should_include_kernel_dtype() returns true/false
 * based on whether the switching code for a specific dtype should be
 * included based on build time constants generated from tracing model
 * execution. This method will be implemented via code-generation and
 * included in this file when code-gen is ready.
 */
inline constexpr bool should_include_kernel_dtype(
    const char* /*kernel_tag_str*/,
    at::ScalarType /*scalar_type*/
) {
  return true;
}
} // namespace at
#endif

/**
 * In the Facebook internal build (using BUCK), this macro is enabled by
 * passing in -c pt.enable_record_kernel_dtype=1 when building the tracer
 * binary.
 */
#if defined ENABLE_RECORD_KERNEL_FUNCTION_DTYPE
namespace at::detail {
TORCH_API void record_kernel_function_dtype(std::string name);
} // namespace at::detail

#define RECORD_KERNEL_FUNCTION_DTYPE(NAME, enum_type) \
  at::detail::record_kernel_function_dtype(           \
      std::string(NAME) + "$" + toString(enum_type));
#else
#define RECORD_KERNEL_FUNCTION_DTYPE(NAME, enum_type)
#endif

#define AT_PRIVATE_CHECK_SELECTIVE_BUILD(enum_type)   \
  do {                                                \
    if constexpr (!at::should_include_kernel_dtype(   \
                      at_dispatch_name, enum_type)) { \
      TORCH_CHECK(                                    \
          false,                                      \
          "dtype '",                                  \
          toString(enum_type),                        \
          "' not selected for kernel tag ",           \
          at_dispatch_name);                          \
    }                                                 \
  } while (0)

#define AT_DISPATCH_CASE_PRELUDE(ENUMTYPE) \
  AT_PRIVATE_CHECK_SELECTIVE_BUILD(ENUMTYPE)
#define AT_DISPATCH_SWITCH_PRELUDE(DISPATCHNAME, ENUMTYPE) \
  RECORD_KERNEL_FUNCTION_DTYPE(DISPATCHNAME, ENUMTYPE)
#define AT_DISPATCH_DEFAULT(DISPATCHNAME, ENUMTYPE) \
  TORCH_CHECK_NOT_IMPLEMENTED(                      \
      false, '"', DISPATCHNAME, "\" not implemented for '", ENUMTYPE, "'")
/* Include of torch/headeronly/Dispatch.h must follow the definitions
   of AT_DISPATCH_SWITCH_PRELUDE, AT_DISPATCH_CASE_PRELUDE, and
   AT_DISPATCH_DEFAULT macros. */
#include <torch/headeronly/core/Dispatch.h>

#define AT_DISPATCH_CASE_QINT(enum_type, scalar_type, ...)                  \
  case enum_type: {                                                         \
    AT_PRIVATE_CHECK_SELECTIVE_BUILD(enum_type);                            \
    using scalar_t = scalar_type;                                           \
    using underlying_t [[maybe_unused]] = typename scalar_t::underlying;    \
    [[maybe_unused]] const auto& SCALAR_TYPE = enum_type;                   \
    [[maybe_unused]] const auto& UNDERLYING_TYPE = toUnderlying(enum_type); \
    return __VA_ARGS__();                                                   \
  }

#define AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                 \
    enum_type, scalar_type, bitwidth, qmin, qmax, ...)                      \
  case enum_type: {                                                         \
    AT_PRIVATE_CHECK_SELECTIVE_BUILD(enum_type);                            \
    using scalar_t = scalar_type;                                           \
    using underlying_t [[maybe_unused]] = typename scalar_t::underlying;    \
    [[maybe_unused]] const auto& SCALAR_TYPE = enum_type;                   \
    [[maybe_unused]] const auto& UNDERLYING_TYPE = toUnderlying(enum_type); \
    [[maybe_unused]] int bit_width = bitwidth;                              \
    [[maybe_unused]] int64_t quant_min = qmin;                              \
    [[maybe_unused]] int64_t quant_max = qmax;                              \
    return __VA_ARGS__();                                                   \
  }

#define AT_DISPATCH_CASE_QINT_TYPES(...)                      \
  AT_DISPATCH_CASE_QINT(at::kQInt8, at::qint8, __VA_ARGS__)   \
  AT_DISPATCH_CASE_QINT(at::kQUInt8, at::quint8, __VA_ARGS__) \
  AT_DISPATCH_CASE_QINT(at::kQInt32, at::qint32, __VA_ARGS__)

#define AT_DISPATCH_QINT_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_QINT_TYPES(__VA_ARGS__))

#define AT_DISPATCH_CASE_QINT_TYPES_AND(SCALARTYPE, ...) \
  AT_DISPATCH_CASE_QINT_TYPES(__VA_ARGS__)               \
  AT_DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)

#define AT_DISPATCH_QINT_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                           \
      TYPE, NAME, AT_DISPATCH_CASE_QINT_TYPES_AND(SCALARTYPE, __VA_ARGS__))

#define AT_DISPATCH_CASE_QINT_BYTE_TYPES(...)               \
  AT_DISPATCH_CASE_QINT(at::kQInt8, at::qint8, __VA_ARGS__) \
  AT_DISPATCH_CASE_QINT(at::kQUInt8, at::quint8, __VA_ARGS__)

#define AT_DISPATCH_QINT_BYTE_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_QINT_BYTE_TYPES(__VA_ARGS__))

#define AT_DISPATCH_CASE_QINT_AND_SUB_BYTE_TYPES(...)                     \
  AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                     \
      at::kQInt8, at::qint8, CHAR_BIT, SCHAR_MIN, SCHAR_MAX, __VA_ARGS__) \
  AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                     \
      at::kQUInt8, at::quint8, CHAR_BIT, 0, UCHAR_MAX, __VA_ARGS__)       \
  AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                     \
      at::kQInt32,                                                        \
      at::qint32,                                                         \
      CHAR_BIT * sizeof(int),                                             \
      INT_MIN,                                                            \
      INT_MAX,                                                            \
      __VA_ARGS__)                                                        \
  AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                     \
      at::kQUInt4x2, at::quint4x2, 4, 0, 15, __VA_ARGS__)                 \
  AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                     \
      at::kQUInt2x4, at::quint2x4, 2, 0, 3, __VA_ARGS__)

#define AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                        \
      TYPE, NAME, AT_DISPATCH_CASE_QINT_AND_SUB_BYTE_TYPES(__VA_ARGS__))
