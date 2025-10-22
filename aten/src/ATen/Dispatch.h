#pragma once

#include <torch/headeronly/core/Dispatch.h>

#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/complex.h>
#include <torch/headeronly/core/Dispatch_v2.h>

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

#define AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, HINT, ...) \
  AT_PRIVATE_CASE_TYPE_USING_HINT_TMPL(                       \
      AT_PRIVATE_CHECK_SELECTIVE_BUILD, enum_type, HINT, __VA_ARGS__)

#define AT_DISPATCH_CASE(enum_type, ...) \
  AT_DISPATCH_CASE_TMPL(AT_PRIVATE_CASE_TYPE_USING_HINT, enum_type, __VA_ARGS__)

#define AT_DISPATCH_CASE_QINT(enum_type, scalar_type, ...) \
  THO_DISPATCH_CASE_QINT(                                  \
      AT_PRIVATE_CHECK_SELECTIVE_BUILD, enum_type, scalar_type, __VA_ARGS__)

#define AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(            \
    enum_type, scalar_type, bitwidth, qmin, qmax, ...) \
  THO_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                 \
      AT_PRIVATE_CHECK_SELECTIVE_BUILD,                \
      enum_type,                                       \
      scalar_type,                                     \
      bitwidth,                                        \
      qmin,                                            \
      qmax,                                            \
      __VA_ARGS__)

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

#define AT_DISPATCH_SWITCH(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH_TMPL(                  \
      RECORD_KERNEL_FUNCTION_DTYPE,         \
      TORCH_CHECK_NOT_IMPLEMENTED,          \
      TYPE,                                 \
      NAME,                                 \
      __VA_ARGS__)

#define AT_DISPATCH_CASE_FLOATING_TYPES(...) \
  THO_DISPATCH_CASE_FLOATING_TYPES(AT_DISPATCH_CASE, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_TYPES_AND_HALF(...) \
  THO_DISPATCH_CASE_FLOATING_TYPES_AND_HALF(AT_DISPATCH_CASE, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                        \
      TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES_AND_HALF(__VA_ARGS__))

#define AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES(...) \
  THO_DISPATCH_CASE_REDUCED_FLOATING_TYPES(AT_DISPATCH_CASE, __VA_ARGS__)

#define AT_DISPATCH_REDUCED_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                       \
      TYPE, NAME, AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES(__VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_TYPES_AND(SCALARTYPE, ...) \
  THO_DISPATCH_CASE_FLOATING_TYPES_AND(                      \
      AT_DISPATCH_CASE, SCALARTYPE, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                               \
      TYPE,                                                         \
      NAME,                                                         \
      AT_DISPATCH_CASE_FLOATING_TYPES_AND(SCALARTYPE, __VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, ...) \
  THO_DISPATCH_CASE_FLOATING_TYPES_AND2(                                    \
      AT_DISPATCH_CASE, SCALARTYPE1, SCALARTYPE2, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES_AND2(       \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                          \
      TYPE,                                    \
      NAME,                                    \
      AT_DISPATCH_CASE_FLOATING_TYPES_AND2(    \
          SCALARTYPE1, SCALARTYPE2, __VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_TYPES_AND3(   \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, ...) \
  THO_DISPATCH_CASE_FLOATING_TYPES_AND3(        \
      AT_DISPATCH_CASE, SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES_AND3(                    \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                       \
      TYPE,                                                 \
      NAME,                                                 \
      AT_DISPATCH_CASE_FLOATING_TYPES_AND3(                 \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, __VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_TYPES_AND4(                \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, ...) \
  THO_DISPATCH_CASE_FLOATING_TYPES_AND4(                     \
      AT_DISPATCH_CASE,                                      \
      SCALARTYPE1,                                           \
      SCALARTYPE2,                                           \
      SCALARTYPE3,                                           \
      SCALARTYPE4,                                           \
      __VA_ARGS__)

#define AT_DISPATCH_CASE_FLOATING_TYPES_AND5(                             \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, SCALARTYPE5, ...) \
  THO_DISPATCH_CASE_FLOATING_TYPES_AND5(                                  \
      AT_DISPATCH_CASE,                                                   \
      SCALARTYPE1,                                                        \
      SCALARTYPE2,                                                        \
      SCALARTYPE3,                                                        \
      SCALARTYPE4,                                                        \
      SCALARTYPE5,                                                        \
      __VA_ARGS__)

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

#define AT_DISPATCH_CASE_COMPLEX_TYPES(...) \
  THO_DISPATCH_CASE_COMPLEX_TYPES(AT_DISPATCH_CASE, __VA_ARGS__)

#define AT_DISPATCH_COMPLEX_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_COMPLEX_TYPES(__VA_ARGS__))

#define AT_DISPATCH_CASE_COMPLEX_TYPES_AND(SCALARTYPE, ...) \
  THO_DISPATCH_CASE_COMPLEX_TYPES_AND(AT_DISPATCH_CASE, SCALARTYPE, __VA_ARGS__)

#define AT_DISPATCH_COMPLEX_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                              \
      TYPE, NAME, AT_DISPATCH_CASE_COMPLEX_TYPES_AND(SCALARTYPE, __VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(...) \
  THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(AT_DISPATCH_CASE, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                           \
      TYPE, NAME, AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(__VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND1(SCALARTYPE, ...) \
  THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND1(                      \
      AT_DISPATCH_CASE, SCALARTYPE, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(    \
    SCALARTYPE, TYPE, NAME, ...)                        \
  AT_DISPATCH_SWITCH(                                   \
      TYPE,                                             \
      NAME,                                             \
      AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND1( \
          SCALARTYPE, __VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND2( \
    SCALARTYPE1, SCALARTYPE2, ...)                        \
  THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND2(      \
      AT_DISPATCH_CASE, SCALARTYPE1, SCALARTYPE2, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(    \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)          \
  AT_DISPATCH_SWITCH(                                   \
      TYPE,                                             \
      NAME,                                             \
      AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND2( \
          SCALARTYPE1, SCALARTYPE2, __VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND3( \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, ...)           \
  THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND3(      \
      AT_DISPATCH_CASE, SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(        \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                       \
      TYPE,                                                 \
      NAME,                                                 \
      AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND3(     \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, __VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND4(    \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, ...) \
  THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND4(         \
      AT_DISPATCH_CASE,                                      \
      SCALARTYPE1,                                           \
      SCALARTYPE2,                                           \
      SCALARTYPE3,                                           \
      SCALARTYPE4,                                           \
      __VA_ARGS__)

#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND4(                     \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                                    \
      TYPE,                                                              \
      NAME,                                                              \
      AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND4(                  \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, __VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND5(                 \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, SCALARTYPE5, ...) \
  THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND5(                      \
      AT_DISPATCH_CASE,                                                   \
      SCALARTYPE1,                                                        \
      SCALARTYPE2,                                                        \
      SCALARTYPE3,                                                        \
      SCALARTYPE4,                                                        \
      SCALARTYPE5,                                                        \
      __VA_ARGS__)

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

#define AT_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND6( \
    SCALARTYPE1,                                          \
    SCALARTYPE2,                                          \
    SCALARTYPE3,                                          \
    SCALARTYPE4,                                          \
    SCALARTYPE5,                                          \
    SCALARTYPE6,                                          \
    ...)                                                  \
  THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND6(      \
      AT_DISPATCH_CASE,                                   \
      SCALARTYPE1,                                        \
      SCALARTYPE2,                                        \
      SCALARTYPE3,                                        \
      SCALARTYPE4,                                        \
      SCALARTYPE5,                                        \
      SCALARTYPE6,                                        \
      __VA_ARGS__)

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

#define AT_DISPATCH_CASE_INTEGRAL_TYPES(...) \
  THO_DISPATCH_CASE_INTEGRAL_TYPES(AT_DISPATCH_CASE, __VA_ARGS__)

#define AT_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))

#define AT_DISPATCH_CASE_INTEGRAL_TYPES_AND(SCALARTYPE, ...) \
  THO_DISPATCH_CASE_INTEGRAL_TYPES_AND(                      \
      AT_DISPATCH_CASE, SCALARTYPE, __VA_ARGS__)

#define AT_DISPATCH_INTEGRAL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                               \
      TYPE,                                                         \
      NAME,                                                         \
      AT_DISPATCH_CASE_INTEGRAL_TYPES_AND(SCALARTYPE, __VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES(...) \
  THO_DISPATCH_CASE_ALL_TYPES(AT_DISPATCH_CASE, __VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__))

#define AT_DISPATCH_CASE_QINT_TYPES(...) \
  THO_DISPATCH_CASE_QINT_TYPES(AT_DISPATCH_CASE_QINT, __VA_ARGS__)

#define AT_DISPATCH_QINT_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_QINT_TYPES(__VA_ARGS__))

#define AT_DISPATCH_CASE_QINT_TYPES_AND(SCALARTYPE, ...) \
  THO_DISPATCH_CASE_QINT_TYPES_AND(                      \
      AT_DISPATCH_CASE_QINT, AT_DISPATCH_CASE, SCALARTYPE, __VA_ARGS__)

#define AT_DISPATCH_QINT_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                           \
      TYPE, NAME, AT_DISPATCH_CASE_QINT_TYPES_AND(SCALARTYPE, __VA_ARGS__))

#define AT_DISPATCH_CASE_QINT_BYTE_TYPES(...) \
  THO_DISPATCH_CASE_QINT_BYTE_TYPES(AT_DISPATCH_CASE_QINT, __VA_ARGS__)

#define AT_DISPATCH_QINT_BYTE_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_QINT_BYTE_TYPES(__VA_ARGS__))

#define AT_DISPATCH_CASE_QINT_AND_SUB_BYTE_TYPES(...) \
  THO_DISPATCH_CASE_QINT_AND_SUB_BYTE_TYPES(          \
      AT_QINT_SUB_BYTE_PRIVATE_CASE_TYPE, __VA_ARGS__)

#define AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                        \
      TYPE, NAME, AT_DISPATCH_CASE_QINT_AND_SUB_BYTE_TYPES(__VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(...) \
  THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(AT_DISPATCH_CASE, __VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                      \
      TYPE, NAME, AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(__VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES_AND(SCALARTYPE, ...) \
  THO_DISPATCH_CASE_ALL_TYPES_AND(AT_DISPATCH_CASE, SCALARTYPE, __VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                          \
      TYPE, NAME, AT_DISPATCH_CASE_ALL_TYPES_AND(SCALARTYPE, __VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND(SCALARTYPE, ...) \
  THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND(                      \
      AT_DISPATCH_CASE, SCALARTYPE, __VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                                      \
      TYPE,                                                                \
      NAME,                                                                \
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND(SCALARTYPE, __VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, ...) \
  THO_DISPATCH_CASE_ALL_TYPES_AND2(                                    \
      AT_DISPATCH_CASE, SCALARTYPE1, SCALARTYPE2, __VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                                         \
      TYPE,                                                                   \
      NAME,                                                                   \
      AT_DISPATCH_CASE_ALL_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, __VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND2( \
    SCALARTYPE1, SCALARTYPE2, ...)                   \
  THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND2(      \
      AT_DISPATCH_CASE, SCALARTYPE1, SCALARTYPE2, __VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(    \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)     \
  AT_DISPATCH_SWITCH(                              \
      TYPE,                                        \
      NAME,                                        \
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND2( \
          SCALARTYPE1, SCALARTYPE2, __VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES_AND3(        \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, ...) \
  THO_DISPATCH_CASE_ALL_TYPES_AND3(             \
      AT_DISPATCH_CASE, SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, __VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES_AND3(                         \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                       \
      TYPE,                                                 \
      NAME,                                                 \
      AT_DISPATCH_CASE_ALL_TYPES_AND3(                      \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, __VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND3( \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, ...)      \
  THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND3(      \
      AT_DISPATCH_CASE, SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, __VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(             \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                       \
      TYPE,                                                 \
      NAME,                                                 \
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND3(          \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, __VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND4(         \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, ...) \
  THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND4(              \
      AT_DISPATCH_CASE,                                      \
      SCALARTYPE1,                                           \
      SCALARTYPE2,                                           \
      SCALARTYPE3,                                           \
      SCALARTYPE4,                                           \
      __VA_ARGS__)

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(                          \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                                    \
      TYPE,                                                              \
      NAME,                                                              \
      AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND4(                       \
          SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, __VA_ARGS__))

#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND5(                      \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, SCALARTYPE5, ...) \
  THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND5(                           \
      AT_DISPATCH_CASE,                                                   \
      SCALARTYPE1,                                                        \
      SCALARTYPE2,                                                        \
      SCALARTYPE3,                                                        \
      SCALARTYPE4,                                                        \
      SCALARTYPE5,                                                        \
      __VA_ARGS__)

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

#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND6( \
    SCALARTYPE1,                                     \
    SCALARTYPE2,                                     \
    SCALARTYPE3,                                     \
    SCALARTYPE4,                                     \
    SCALARTYPE5,                                     \
    SCALARTYPE6,                                     \
    ...)                                             \
  THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND6(      \
      AT_DISPATCH_CASE,                              \
      SCALARTYPE1,                                   \
      SCALARTYPE2,                                   \
      SCALARTYPE3,                                   \
      SCALARTYPE4,                                   \
      SCALARTYPE5,                                   \
      SCALARTYPE6,                                   \
      __VA_ARGS__)

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

#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND7( \
    SCALARTYPE1,                                     \
    SCALARTYPE2,                                     \
    SCALARTYPE3,                                     \
    SCALARTYPE4,                                     \
    SCALARTYPE5,                                     \
    SCALARTYPE6,                                     \
    SCALARTYPE7,                                     \
    ...)                                             \
  THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND7(      \
      AT_DISPATCH_CASE,                              \
      SCALARTYPE1,                                   \
      SCALARTYPE2,                                   \
      SCALARTYPE3,                                   \
      SCALARTYPE4,                                   \
      SCALARTYPE5,                                   \
      SCALARTYPE6,                                   \
      SCALARTYPE7,                                   \
      __VA_ARGS__)

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

#define AT_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND8( \
    SCALARTYPE1,                                     \
    SCALARTYPE2,                                     \
    SCALARTYPE3,                                     \
    SCALARTYPE4,                                     \
    SCALARTYPE5,                                     \
    SCALARTYPE6,                                     \
    SCALARTYPE7,                                     \
    SCALARTYPE8,                                     \
    ...)                                             \
  THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND8(      \
      AT_DISPATCH_CASE,                              \
      SCALARTYPE1,                                   \
      SCALARTYPE2,                                   \
      SCALARTYPE3,                                   \
      SCALARTYPE4,                                   \
      SCALARTYPE5,                                   \
      SCALARTYPE6,                                   \
      SCALARTYPE7,                                   \
      SCALARTYPE8,                                   \
      __VA_ARGS__)

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

#define AT_DISPATCH_CASE_BIT_TYPES(...) \
  THO_DISPATCH_CASE_BIT_TYPES(AT_DISPATCH_CASE, __VA_ARGS__)

#define AT_DISPATCH_BIT_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_BIT_TYPES(__VA_ARGS__))

#define AT_DISPATCH_INDEX_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                            \
      TYPE,                                      \
      NAME,                                      \
      THO_DISPATCH_CASE_INDEX_TYPES(             \
          AT_PRIVATE_CASE_TYPE_USING_HINT, __VA_ARGS__))
