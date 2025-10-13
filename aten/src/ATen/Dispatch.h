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

#define AT_DISPATCH_CASE_FLOATING_TYPES(...)            \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_TYPES_AND_HALF(...)   \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                        \
      TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES_AND_HALF(__VA_ARGS__))

#define AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES(...)  \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define AT_DISPATCH_REDUCED_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                       \
      TYPE, NAME, AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES(__VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_TYPES_AND(SCALARTYPE, ...) \
  AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)               \
  AT_DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                               \
      TYPE,                                                         \
      NAME,                                                         \
      AT_DISPATCH_CASE_FLOATING_TYPES_AND(SCALARTYPE, __VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, ...) \
  AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)                              \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                                \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES_AND2(       \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                          \
      TYPE,                                    \
      NAME,                                    \
      AT_DISPATCH_CASE_FLOATING_TYPES_AND2(    \
          SCALARTYPE1, SCALARTYPE2, __VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_TYPES_AND3(   \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, ...) \
  AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)  \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)    \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)    \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES_AND3(                    \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                       \
      TYPE,                                                 \
      NAME,                                                 \
      AT_DISPATCH_CASE_FLOATING_TYPES_AND3(                 \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, __VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_TYPES_AND4(                \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, ...) \
  AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)               \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                 \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                 \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)                 \
  AT_DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)

#define AT_DISPATCH_CASE_FLOATING_TYPES_AND5(                             \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, SCALARTYPE5, ...) \
  AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)                            \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                              \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                              \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)                              \
  AT_DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)                              \
  AT_DISPATCH_CASE(SCALARTYPE5, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES_AND4(                                 \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                                    \
      TYPE,                                                              \
      NAME,                                                              \
      AT_DISPATCH_CASE_FLOATING_TYPES_AND4(                              \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, __VA_ARGS__))

#define AT_DISPATCH_FLOATING_TYPES_AND5(    \
    SCALARTYPE1,                            \
    SCALARTYPE2,                            \
    SCALARTYPE3,                            \
    SCALARTYPE4,                            \
    SCALARTYPE5,                            \
    TYPE,                                   \
    NAME,                                   \
    ...)                                    \
  AT_DISPATCH_SWITCH(                       \
      TYPE,                                 \
      NAME,                                 \
      AT_DISPATCH_CASE_FLOATING_TYPES_AND5( \
          SCALARTYPE1,                      \
          SCALARTYPE2,                      \
          SCALARTYPE3,                      \
          SCALARTYPE4,                      \
          SCALARTYPE5,                      \
          __VA_ARGS__))

#define AT_DISPATCH_CASE_COMPLEX_TYPES(...)                    \
  AT_DISPATCH_CASE(at::ScalarType::ComplexDouble, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::ComplexFloat, __VA_ARGS__)

#define AT_DISPATCH_COMPLEX_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_COMPLEX_TYPES(__VA_ARGS__))

#define AT_DISPATCH_CASE_COMPLEX_TYPES_AND(SCALARTYPE, ...) \
  AT_DISPATCH_CASE_COMPLEX_TYPES(__VA_ARGS__)               \
  AT_DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)

#define AT_DISPATCH_COMPLEX_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                              \
      TYPE, NAME, AT_DISPATCH_CASE_COMPLEX_TYPES_AND(SCALARTYPE, __VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(...) \
  AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)           \
  AT_DISPATCH_CASE_COMPLEX_TYPES(__VA_ARGS__)

#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                           \
      TYPE, NAME, AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(__VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND1(SCALARTYPE, ...) \
  AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(__VA_ARGS__)                \
  AT_DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(    \
    SCALARTYPE, TYPE, NAME, ...)                        \
  AT_DISPATCH_SWITCH(                                   \
      TYPE,                                             \
      NAME,                                             \
      AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND1( \
          SCALARTYPE, __VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND2(  \
    SCALARTYPE1, SCALARTYPE2, ...)                         \
  AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(__VA_ARGS__) \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)               \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(    \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)          \
  AT_DISPATCH_SWITCH(                                   \
      TYPE,                                             \
      NAME,                                             \
      AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND2( \
          SCALARTYPE1, SCALARTYPE2, __VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND3(  \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, ...)            \
  AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(__VA_ARGS__) \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)               \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)               \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(        \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                       \
      TYPE,                                                 \
      NAME,                                                 \
      AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND3(     \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, __VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND4(    \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, ...) \
  AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(__VA_ARGS__)   \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                 \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                 \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)                 \
  AT_DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND4(                     \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                                    \
      TYPE,                                                              \
      NAME,                                                              \
      AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND4(                  \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, __VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND5(                 \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, SCALARTYPE5, ...) \
  AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(__VA_ARGS__)                \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                              \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                              \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)                              \
  AT_DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)                              \
  AT_DISPATCH_CASE(SCALARTYPE5, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND5(    \
    SCALARTYPE1,                                        \
    SCALARTYPE2,                                        \
    SCALARTYPE3,                                        \
    SCALARTYPE4,                                        \
    SCALARTYPE5,                                        \
    TYPE,                                               \
    NAME,                                               \
    ...)                                                \
  AT_DISPATCH_SWITCH(                                   \
      TYPE,                                             \
      NAME,                                             \
      AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND5( \
          SCALARTYPE1,                                  \
          SCALARTYPE2,                                  \
          SCALARTYPE3,                                  \
          SCALARTYPE4,                                  \
          SCALARTYPE5,                                  \
          __VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND6(  \
    SCALARTYPE1,                                           \
    SCALARTYPE2,                                           \
    SCALARTYPE3,                                           \
    SCALARTYPE4,                                           \
    SCALARTYPE5,                                           \
    SCALARTYPE6,                                           \
    ...)                                                   \
  AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(__VA_ARGS__) \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)               \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)               \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)               \
  AT_DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)               \
  AT_DISPATCH_CASE(SCALARTYPE5, __VA_ARGS__)               \
  AT_DISPATCH_CASE(SCALARTYPE6, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND6(    \
    SCALARTYPE1,                                        \
    SCALARTYPE2,                                        \
    SCALARTYPE3,                                        \
    SCALARTYPE4,                                        \
    SCALARTYPE5,                                        \
    SCALARTYPE6,                                        \
    TYPE,                                               \
    NAME,                                               \
    ...)                                                \
  AT_DISPATCH_SWITCH(                                   \
      TYPE,                                             \
      NAME,                                             \
      AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND6( \
          SCALARTYPE1,                                  \
          SCALARTYPE2,                                  \
          SCALARTYPE3,                                  \
          SCALARTYPE4,                                  \
          SCALARTYPE5,                                  \
          SCALARTYPE6,                                  \
          __VA_ARGS__))

#define AT_DISPATCH_CASE_INTEGRAL_TYPES(...)          \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__)

#define AT_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))

#define AT_DISPATCH_CASE_INTEGRAL_TYPES_AND(SCALARTYPE, ...) \
  AT_DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__)               \
  AT_DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)

#define AT_DISPATCH_INTEGRAL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                               \
      TYPE,                                                         \
      NAME,                                                         \
      AT_DISPATCH_CASE_INTEGRAL_TYPES_AND(SCALARTYPE, __VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES(...)        \
  AT_DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__) \
  AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__))

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

#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(...) \
  AT_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__)           \
  AT_DISPATCH_CASE_COMPLEX_TYPES(__VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                      \
      TYPE, NAME, AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(__VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES_AND(SCALARTYPE, ...) \
  AT_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__)               \
  AT_DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                          \
      TYPE, NAME, AT_DISPATCH_CASE_ALL_TYPES_AND(SCALARTYPE, __VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND(SCALARTYPE, ...) \
  AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(__VA_ARGS__)               \
  AT_DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                                      \
      TYPE,                                                                \
      NAME,                                                                \
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND(SCALARTYPE, __VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, ...) \
  AT_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__)                              \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                           \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                                         \
      TYPE,                                                                   \
      NAME,                                                                   \
      AT_DISPATCH_CASE_ALL_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, __VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND2(  \
    SCALARTYPE1, SCALARTYPE2, ...)                    \
  AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(__VA_ARGS__) \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(    \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)     \
  AT_DISPATCH_SWITCH(                              \
      TYPE,                                        \
      NAME,                                        \
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND2( \
          SCALARTYPE1, SCALARTYPE2, __VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES_AND3(        \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, ...) \
  AT_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__)       \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)    \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)    \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES_AND3(                         \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                       \
      TYPE,                                                 \
      NAME,                                                 \
      AT_DISPATCH_CASE_ALL_TYPES_AND3(                      \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, __VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND3(  \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, ...)       \
  AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(__VA_ARGS__) \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(             \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                       \
      TYPE,                                                 \
      NAME,                                                 \
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND3(          \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, __VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND4(         \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, ...) \
  AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(__VA_ARGS__)        \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                 \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                 \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)                 \
  AT_DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(                          \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                                    \
      TYPE,                                                              \
      NAME,                                                              \
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND4(                       \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, __VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND5(                      \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, SCALARTYPE5, ...) \
  AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(__VA_ARGS__)                     \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                              \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                              \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)                              \
  AT_DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)                              \
  AT_DISPATCH_CASE(SCALARTYPE5, __VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND5(    \
    SCALARTYPE1,                                   \
    SCALARTYPE2,                                   \
    SCALARTYPE3,                                   \
    SCALARTYPE4,                                   \
    SCALARTYPE5,                                   \
    TYPE,                                          \
    NAME,                                          \
    ...)                                           \
  AT_DISPATCH_SWITCH(                              \
      TYPE,                                        \
      NAME,                                        \
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND5( \
          SCALARTYPE1,                             \
          SCALARTYPE2,                             \
          SCALARTYPE3,                             \
          SCALARTYPE4,                             \
          SCALARTYPE5,                             \
          __VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND6(  \
    SCALARTYPE1,                                      \
    SCALARTYPE2,                                      \
    SCALARTYPE3,                                      \
    SCALARTYPE4,                                      \
    SCALARTYPE5,                                      \
    SCALARTYPE6,                                      \
    ...)                                              \
  AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(__VA_ARGS__) \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE5, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE6, __VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND6(    \
    SCALARTYPE1,                                   \
    SCALARTYPE2,                                   \
    SCALARTYPE3,                                   \
    SCALARTYPE4,                                   \
    SCALARTYPE5,                                   \
    SCALARTYPE6,                                   \
    TYPE,                                          \
    NAME,                                          \
    ...)                                           \
  AT_DISPATCH_SWITCH(                              \
      TYPE,                                        \
      NAME,                                        \
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND6( \
          SCALARTYPE1,                             \
          SCALARTYPE2,                             \
          SCALARTYPE3,                             \
          SCALARTYPE4,                             \
          SCALARTYPE5,                             \
          SCALARTYPE6,                             \
          __VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND7(  \
    SCALARTYPE1,                                      \
    SCALARTYPE2,                                      \
    SCALARTYPE3,                                      \
    SCALARTYPE4,                                      \
    SCALARTYPE5,                                      \
    SCALARTYPE6,                                      \
    SCALARTYPE7,                                      \
    ...)                                              \
  AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(__VA_ARGS__) \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE5, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE6, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE7, __VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND7(    \
    SCALARTYPE1,                                   \
    SCALARTYPE2,                                   \
    SCALARTYPE3,                                   \
    SCALARTYPE4,                                   \
    SCALARTYPE5,                                   \
    SCALARTYPE6,                                   \
    SCALARTYPE7,                                   \
    TYPE,                                          \
    NAME,                                          \
    ...)                                           \
  AT_DISPATCH_SWITCH(                              \
      TYPE,                                        \
      NAME,                                        \
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND7( \
          SCALARTYPE1,                             \
          SCALARTYPE2,                             \
          SCALARTYPE3,                             \
          SCALARTYPE4,                             \
          SCALARTYPE5,                             \
          SCALARTYPE6,                             \
          SCALARTYPE7,                             \
          __VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND8(  \
    SCALARTYPE1,                                      \
    SCALARTYPE2,                                      \
    SCALARTYPE3,                                      \
    SCALARTYPE4,                                      \
    SCALARTYPE5,                                      \
    SCALARTYPE6,                                      \
    SCALARTYPE7,                                      \
    SCALARTYPE8,                                      \
    ...)                                              \
  AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(__VA_ARGS__) \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE5, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE6, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE7, __VA_ARGS__)          \
  AT_DISPATCH_CASE(SCALARTYPE8, __VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND8(    \
    SCALARTYPE1,                                   \
    SCALARTYPE2,                                   \
    SCALARTYPE3,                                   \
    SCALARTYPE4,                                   \
    SCALARTYPE5,                                   \
    SCALARTYPE6,                                   \
    SCALARTYPE7,                                   \
    SCALARTYPE8,                                   \
    TYPE,                                          \
    NAME,                                          \
    ...)                                           \
  AT_DISPATCH_SWITCH(                              \
      TYPE,                                        \
      NAME,                                        \
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND8( \
          SCALARTYPE1,                             \
          SCALARTYPE2,                             \
          SCALARTYPE3,                             \
          SCALARTYPE4,                             \
          SCALARTYPE5,                             \
          SCALARTYPE6,                             \
          SCALARTYPE7,                             \
          SCALARTYPE8,                             \
          __VA_ARGS__))

#define AT_DISPATCH_CASE_BIT_TYPES(...)                  \
  AT_DISPATCH_CASE(at::ScalarType::Bits1x8, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Bits2x4, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Bits4x2, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Bits8, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::Bits16, __VA_ARGS__)

#define AT_DISPATCH_BIT_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_BIT_TYPES(__VA_ARGS__))

#define AT_DISPATCH_INDEX_TYPES(TYPE, NAME, ...)     \
  AT_DISPATCH_SWITCH(                                \
      TYPE,                                          \
      NAME,                                          \
      AT_PRIVATE_CASE_TYPE_USING_HINT(               \
          at::ScalarType::Int, index_t, __VA_ARGS__) \
          AT_PRIVATE_CASE_TYPE_USING_HINT(           \
              at::ScalarType::Long, index_t, __VA_ARGS__))
