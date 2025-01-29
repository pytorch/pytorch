#pragma once

#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/complex.h>
#include <c10/util/string_view.h>

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

#define AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, HINT, ...)                 \
  case enum_type: {                                                           \
    AT_PRIVATE_CHECK_SELECTIVE_BUILD(enum_type);                              \
    using HINT [[maybe_unused]] = c10::impl::ScalarTypeToCPPTypeT<enum_type>; \
    return __VA_ARGS__();                                                     \
  }

#define AT_DISPATCH_CASE(enum_type, ...) \
  AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, scalar_t, __VA_ARGS__)

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

// The AT_DISPATCH_* family of macros provides the ability to
// conveniently generate specializations of a kernel over all of the
// dtypes we care about in PyTorch.  We call it "dispatch" because
// we are "dispatching" to the correct, dtype-specific kernel.
//
// A standard usage looks like:
//
//      AT_DISPATCH_ALL_TYPES(self.scalar_type(), "op_name", [&] {
//          // Your code here, with 'scalar_t' now defined to
//          // be the dtype in question
//      });
//
// There are many variations of this macro, so it's important to
// understand exactly /which/ dtypes you want to get instantiated, as
// well as what the "default" set is.
//
// The default set of dtypes that are instantiated (e.g., by
// AT_DISPATCH_ALL_TYPES) are floating point types (float, double),
// and integral types (int32_t, int64_t, int16_t, int8_t, uint8_t),
// but NOT booleans (bool), half-precision floats (Half) or
// complex number (c10::complex<float>, c10::complex<double>).
// This "cut" is somewhat historical (the default types are the
// ones that TH historically supported), but it also reflects the
// fact that the non-default types are "poorly" behaved (booleans
// are NOT integers mod 2, half precision operations ~essentially
// don't exist on CPU, complex numbers are an experimental application).
//
// Here are the questions you should generally ask to decide which
// dispatch you want:
//
// 1. Is this an integral or floating point specific operation?
//    (If so, you'll want one of the FLOATING or INTEGRAL macros.)
//
// 2. Should half be supported?  (If you're on CPU, the answer is almost
//    definitely no.  If you do want support, use one of the AND_HALF
//    macros)
//
// Much rarer situations:
//
// 3. Should bool be supported?  (You often have to write your kernel
//    differently if arithmetic operations are involved.)  If so,
//    Use AT_DISPATCH_ALL_TYPES_AND along with ScalarType::Bool
//
// 4. Should complex be supported?  The answer is almost always no,
//    unless you are working on "generic" code that should work on
//    all dtypes.
//
// Parameters:
// -----------
//
// 1. The NAME argument is a "tag" that is used to trace and then
//    conditionally compile fragments of the case statements such
//    that the kernel functions are specialized only for the dtypes
//    that are needed. The NAME parameter *must* be a build time
//    const char* (can't be std::string, etc...)
//
// Please ensure that the NAME is unique for every implementation
// or you run the risk of over-including code for the kernel
// functions. There is no risk of missing out on any code, so
// it's mostly a risk of a Type-2 error, and not a Type-1 error.
//
// Switch-like syntax:
// -------------------
// There is also a switch-case like syntax which is useful if a kernel
// needs to be specialized for particular scalar types
//
//      AT_DISPATCH_SWITCH(self.scalar_type(), "op_name",
//          AT_DISPATCH_CASE_INTEGRAL_TYPES([&] {
//            op_integral<scalar_t>(iter);
//          })
//          AT_DISPATCH_CASE_FLOATING_TYPES([&] {
//            op_floating<scalar_t>(iter);
//          })
//          AT_DISPATCH_CASE(kBool, [&] {
//            op_bool(iter);
//          })
//      );
//
// For each AT_DISPATCH_FOO macro, there is a corresponding
// AT_DISPATCH_CASE_FOO macro which can be used inside of an
// AT_DISPATCH_SWITCH block.

// NB: the the_type variable is not used, but we have kept it for
// backwards compatibility.  It's probably not used by anyone though;
// but we're just being safe (and it doesn't hurt.)  Note we must
// use it to shut up warnings about unused store.

#define AT_DISPATCH_SWITCH(TYPE, NAME, ...)                                 \
  [&] {                                                                     \
    const at::ScalarType _st = TYPE;                                        \
    constexpr const char* at_dispatch_name = NAME;                          \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    RECORD_KERNEL_FUNCTION_DTYPE(at_dispatch_name, _st);                    \
    switch (_st) {                                                          \
      __VA_ARGS__                                                           \
      default:                                                              \
        TORCH_CHECK(                                                        \
            false,                                                          \
            '"',                                                            \
            at_dispatch_name,                                               \
            "\" not implemented for '",                                     \
            toString(_st),                                                  \
            "'");                                                           \
    }                                                                       \
  }()

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
