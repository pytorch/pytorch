#pragma once

#include <torch/headeronly/core/ScalarType.h>

/*
  When defined, AT_DISPATCH_SWITCH_PRELUDE macro is inserted before
  the dispatch switch statement.
*/
#ifndef AT_DISPATCH_SWITCH_PRELUDE
#define AT_DISPATCH_SWITCH_PRELUDE(DISPATCHNAME, ENUMTYPE)
#endif

/*
  When defined, AT_DISPATCH_CASE_PRELUDE macro is prepended to the
  case statement of the dispatch switch statement.
*/
#ifndef AT_DISPATCH_CASE_PRELUDE
#define AT_DISPATCH_CASE_PRELUDE(ENUMTYPE)
#endif

/*
  When defined, AT_DISPATCH_DEFAULT macro is used in the default
  statement of the dispatch switch statement. By default,
  AT_DISPATCH_DEFAULT contains a failing check about not implemented
  enum type.
*/
#ifndef AT_DISPATCH_DEFAULT
#include <torch/headeronly/util/Exception.h>
#define AT_DISPATCH_DEFAULT(DISPATCHNAME, ENUMTYPE) \
  STD_TORCH_CHECK(                                  \
      false, '"', DISPATCHNAME, "\" not implemented for '", ENUMTYPE, "'")
#endif

#define AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, HINT, ...)     \
  case enum_type: {                                               \
    AT_DISPATCH_CASE_PRELUDE(enum_type);                          \
    using HINT [[maybe_unused]] =                                 \
        torch::headeronly::impl::ScalarTypeToCPPTypeT<enum_type>; \
    return __VA_ARGS__();                                         \
  }

#define AT_DISPATCH_CASE(enum_type, ...) \
  AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, scalar_t, __VA_ARGS__)

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

namespace detail {

inline at::ScalarType scalar_type(at::ScalarType s) {
  return s;
}

} // namespace detail

#define AT_DISPATCH_SWITCH(TYPE, NAME, ...)                                 \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    constexpr const char* at_dispatch_name = NAME;                          \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    torch::headeronly::ScalarType _st = ::detail::scalar_type(the_type);    \
    AT_DISPATCH_SWITCH_PRELUDE(at_dispatch_name, _st);                      \
    switch (_st) {                                                          \
      __VA_ARGS__                                                           \
      default:                                                              \
        AT_DISPATCH_DEFAULT(at_dispatch_name, _st);                         \
    }                                                                       \
  }()

#define AT_DISPATCH_CASE_FLOATING_TYPES(...)                           \
  AT_DISPATCH_CASE(torch::headeronly::ScalarType::Double, __VA_ARGS__) \
  AT_DISPATCH_CASE(torch::headeronly::ScalarType::Float, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#define AT_DISPATCH_CASE_FLOATING_TYPES_AND_HALF(...)                  \
  AT_DISPATCH_CASE(torch::headeronly::ScalarType::Double, __VA_ARGS__) \
  AT_DISPATCH_CASE(torch::headeronly::ScalarType::Float, __VA_ARGS__)  \
  AT_DISPATCH_CASE(torch::headeronly::ScalarType::Half, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                        \
      TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES_AND_HALF(__VA_ARGS__))

#define AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES(...)                 \
  AT_DISPATCH_CASE(torch::headeronly::ScalarType::Half, __VA_ARGS__) \
  AT_DISPATCH_CASE(torch::headeronly::ScalarType::BFloat16, __VA_ARGS__)

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

#define AT_DISPATCH_CASE_COMPLEX_TYPES(...)                                   \
  AT_DISPATCH_CASE(torch::headeronly::ScalarType::ComplexDouble, __VA_ARGS__) \
  AT_DISPATCH_CASE(torch::headeronly::ScalarType::ComplexFloat, __VA_ARGS__)

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

#define AT_DISPATCH_CASE_INTEGRAL_TYPES(...)                         \
  AT_DISPATCH_CASE(torch::headeronly::ScalarType::Byte, __VA_ARGS__) \
  AT_DISPATCH_CASE(torch::headeronly::ScalarType::Char, __VA_ARGS__) \
  AT_DISPATCH_CASE(torch::headeronly::ScalarType::Int, __VA_ARGS__)  \
  AT_DISPATCH_CASE(torch::headeronly::ScalarType::Long, __VA_ARGS__) \
  AT_DISPATCH_CASE(torch::headeronly::ScalarType::Short, __VA_ARGS__)

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

#define AT_DISPATCH_CASE_BIT_TYPES(...)                                 \
  AT_DISPATCH_CASE(torch::headeronly::ScalarType::Bits1x8, __VA_ARGS__) \
  AT_DISPATCH_CASE(torch::headeronly::ScalarType::Bits2x4, __VA_ARGS__) \
  AT_DISPATCH_CASE(torch::headeronly::ScalarType::Bits4x2, __VA_ARGS__) \
  AT_DISPATCH_CASE(torch::headeronly::ScalarType::Bits8, __VA_ARGS__)   \
  AT_DISPATCH_CASE(torch::headeronly::ScalarType::Bits16, __VA_ARGS__)

#define AT_DISPATCH_BIT_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_BIT_TYPES(__VA_ARGS__))

#define AT_DISPATCH_INDEX_TYPES(TYPE, NAME, ...)                    \
  AT_DISPATCH_SWITCH(                                               \
      TYPE,                                                         \
      NAME,                                                         \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                              \
          torch::headeronly::ScalarType::Int, index_t, __VA_ARGS__) \
          AT_PRIVATE_CASE_TYPE_USING_HINT(                          \
              torch::headeronly::ScalarType::Long, index_t, __VA_ARGS__))
