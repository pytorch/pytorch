#pragma once

#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>
#include <climits>

/*
  This header file provides a set of THO_... macros that can be used
  to define custom dispatch macros as follows:

  1. Define a DISPATCH_CASE macro using AT_DISPATCH_CASE_TMPL as a
    template, for instance:

      #define MY_DISPATCH_CASE(enum_type, ...) \
        AT_DISPATCH_CASE_TMPL(MY_CASE_PRELUDE, enum_type, __VA_ARGS__)

    where MY_CASE_PRELUDE(enum_type) is a CPP macro that will be
    expanded inside each case block of a switch statement.

  2. Define a DISPATCH_SWITCH macro using AT_DISPATCH_SWITCH_TMPL as a
    template, for instance:

      #define MY_DISPATCH_SWITCH(TYPE, NAME, ...)                  \
        AT_DISPATCH_TMPL(MY_SWITCH_PRELUDE, MY_CHECK_NOT_IMPLEMENTED, \
                      TYPE, NAME, __VA_ARGS__)

    where

    - MY_DISPATCH_PRELUDE(dispatch_name, enum_type) is a CPP macro
      that will be expanded before the switch statement;
    - MY_CHECK_NOT_IMPLEMENTED(...) is a CPP macro that will be
      expanded in the switch default block.

  3. Define DISPATCH_CASE macros using
     THO_DISPATCH_CASE_..._TYPES... macros as templates, for instance

       #define MY_DISPATCH_CASE_FLOATING_TYPES(...) \
         THO_DISPATCH_CASE_FLOATING_TYPES(MY_DISPATCH_CASE, __VA_ARGS__)

       #define MY_DISPATCH_CASE_FLOATING_TYPES_AND(SCALARTYPE, ...) \
         THO_DISPATCH_CASE_FLOATING_TYPES_AND(                      \
           MY_DISPATCH_CASE, SCALARTYPE, __VA_ARGS__)

       etc.

  4. Finally, define DISPATCH_..._TYPES macros, for instance:

       #define MY_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
         MY_DISPATCH_SWITCH(TYPE, NAME,
  MY_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

       #define MY_DISPATCH_FLOATING_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
         MY_DISPATCH_SWITCH(                                               \
             TYPE,                                                         \
             NAME,                                                         \
             MY_DISPATCH_CASE_FLOATING_TYPES_AND(SCALARTYPE, __VA_ARGS__))

       etc.

  Now, in a code, one can use

    MY_DISPATCH_FLOATING_TYPES(self.scalar_type(), "op_name", [&] {...});

  for instance.

  See also ATen/Dispatch.h that defines a complete set of
  AT_DISPATCH_...  macros (hint: for custom dispatch macros,
  copy-paste AT_DISPATCH_... macros and replace `AT_` prefixes with a
  custom prefix).

  Note that while AT_DISPATCH_..._TYPES macros may depend on the torch
  library (for example, when using selective build) then
  `THO_... macros do not. The torch library dependency may exists only
  within user-defined DISPATCH_CASE and DISPATCH_SWITCH macros.
*/

#define THO_DISPATCH_CASE_QINT(PRELUDE, enum_type, scalar_type, ...)     \
  case enum_type: {                                                      \
    PRELUDE(enum_type);                                                  \
    using scalar_t = scalar_type;                                        \
    using underlying_t [[maybe_unused]] = typename scalar_t::underlying; \
    [[maybe_unused]] const auto& SCALAR_TYPE = enum_type;                \
    [[maybe_unused]] const auto& UNDERLYING_TYPE =                       \
        c10::toUnderlying(enum_type);                                    \
    return __VA_ARGS__();                                                \
  }

#define THO_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                             \
    PRELUDE, enum_type, scalar_type, bitwidth, qmin, qmax, ...)          \
  case enum_type: {                                                      \
    PRELUDE(enum_type);                                                  \
    using scalar_t = scalar_type;                                        \
    using underlying_t [[maybe_unused]] = typename scalar_t::underlying; \
    [[maybe_unused]] const auto& SCALAR_TYPE = enum_type;                \
    [[maybe_unused]] const auto& UNDERLYING_TYPE =                       \
        c10::toUnderlying(enum_type);                                    \
    [[maybe_unused]] int bit_width = bitwidth;                           \
    [[maybe_unused]] int64_t quant_min = qmin;                           \
    [[maybe_unused]] int64_t quant_max = qmax;                           \
    return __VA_ARGS__();                                                \
  }

#define THO_DISPATCH_TYPES(                                \
    DISPATCH_SWITCH, DISPATCH_CASE_TYPES, TYPE, NAME, ...) \
  DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_TYPES(__VA_ARGS__))

#define THO_DISPATCH_CASE_FLOATING_TYPES(DISPATCH_CASE, ...) \
  DISPATCH_CASE(c10::ScalarType::Double, __VA_ARGS__)        \
  DISPATCH_CASE(c10::ScalarType::Float, __VA_ARGS__)

#define THO_DISPATCH_CASE_FLOATING_TYPES_AND_HALF(DISPATCH_CASE, ...) \
  DISPATCH_CASE(c10::ScalarType::Double, __VA_ARGS__)                 \
  DISPATCH_CASE(c10::ScalarType::Float, __VA_ARGS__)                  \
  DISPATCH_CASE(c10::ScalarType::Half, __VA_ARGS__)

#define THO_DISPATCH_CASE_REDUCED_FLOATING_TYPES(DISPATCH_CASE, ...) \
  DISPATCH_CASE(c10::ScalarType::Half, __VA_ARGS__)                  \
  DISPATCH_CASE(c10::ScalarType::BFloat16, __VA_ARGS__)

#define THO_DISPATCH_CASE_FLOATING_TYPES_AND(DISPATCH_CASE, SCALARTYPE, ...) \
  THO_DISPATCH_CASE_FLOATING_TYPES(DISPATCH_CASE, __VA_ARGS__)               \
  DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)

#define THO_DISPATCH_CASE_FLOATING_TYPES_AND2(                 \
    DISPATCH_CASE, SCALARTYPE1, SCALARTYPE2, ...)              \
  THO_DISPATCH_CASE_FLOATING_TYPES(DISPATCH_CASE, __VA_ARGS__) \
  DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                      \
  DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)

#define THO_DISPATCH_CASE_FLOATING_TYPES_AND3(                 \
    DISPATCH_CASE, SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, ...) \
  THO_DISPATCH_CASE_FLOATING_TYPES(DISPATCH_CASE, __VA_ARGS__) \
  DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                      \
  DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                      \
  DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)

#define THO_DISPATCH_CASE_FLOATING_TYPES_AND4(                              \
    DISPATCH_CASE, SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, ...) \
  THO_DISPATCH_CASE_FLOATING_TYPES(DISPATCH_CASE, __VA_ARGS__)              \
  DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                                   \
  DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                                   \
  DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)                                   \
  DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)

#define THO_DISPATCH_CASE_FLOATING_TYPES_AND5(                 \
    DISPATCH_CASE,                                             \
    SCALARTYPE1,                                               \
    SCALARTYPE2,                                               \
    SCALARTYPE3,                                               \
    SCALARTYPE4,                                               \
    SCALARTYPE5,                                               \
    ...)                                                       \
  THO_DISPATCH_CASE_FLOATING_TYPES(DISPATCH_CASE, __VA_ARGS__) \
  DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                      \
  DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                      \
  DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)                      \
  DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)                      \
  DISPATCH_CASE(SCALARTYPE5, __VA_ARGS__)

#define THO_DISPATCH_CASE_COMPLEX_TYPES(DISPATCH_CASE, ...)  \
  DISPATCH_CASE(c10::ScalarType::ComplexDouble, __VA_ARGS__) \
  DISPATCH_CASE(c10::ScalarType::ComplexFloat, __VA_ARGS__)

#define THO_DISPATCH_CASE_COMPLEX_TYPES_AND(DISPATCH_CASE, SCALARTYPE, ...) \
  THO_DISPATCH_CASE_COMPLEX_TYPES(DISPATCH_CASE, __VA_ARGS__)               \
  DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)

#define THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(DISPATCH_CASE, ...) \
  THO_DISPATCH_CASE_FLOATING_TYPES(DISPATCH_CASE, __VA_ARGS__)           \
  THO_DISPATCH_CASE_COMPLEX_TYPES(DISPATCH_CASE, __VA_ARGS__)

#define THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND1(                 \
    DISPATCH_CASE, SCALARTYPE, ...)                                        \
  THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(DISPATCH_CASE, __VA_ARGS__) \
  DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)

#define THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND2(                 \
    DISPATCH_CASE, SCALARTYPE1, SCALARTYPE2, ...)                          \
  THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(DISPATCH_CASE, __VA_ARGS__) \
  DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                                  \
  DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)

#define THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND3(                 \
    DISPATCH_CASE, SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, ...)             \
  THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(DISPATCH_CASE, __VA_ARGS__) \
  DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                                  \
  DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                                  \
  DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)

#define THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND4(                  \
    DISPATCH_CASE, SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, ...) \
  THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(DISPATCH_CASE, __VA_ARGS__)  \
  DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                                   \
  DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                                   \
  DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)                                   \
  DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)

#define THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND5(                 \
    DISPATCH_CASE,                                                         \
    SCALARTYPE1,                                                           \
    SCALARTYPE2,                                                           \
    SCALARTYPE3,                                                           \
    SCALARTYPE4,                                                           \
    SCALARTYPE5,                                                           \
    ...)                                                                   \
  THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(DISPATCH_CASE, __VA_ARGS__) \
  DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                                  \
  DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                                  \
  DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)                                  \
  DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)                                  \
  DISPATCH_CASE(SCALARTYPE5, __VA_ARGS__)

#define THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND6(                 \
    DISPATCH_CASE,                                                         \
    SCALARTYPE1,                                                           \
    SCALARTYPE2,                                                           \
    SCALARTYPE3,                                                           \
    SCALARTYPE4,                                                           \
    SCALARTYPE5,                                                           \
    SCALARTYPE6,                                                           \
    ...)                                                                   \
  THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(DISPATCH_CASE, __VA_ARGS__) \
  DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                                  \
  DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                                  \
  DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)                                  \
  DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)                                  \
  DISPATCH_CASE(SCALARTYPE5, __VA_ARGS__)                                  \
  DISPATCH_CASE(SCALARTYPE6, __VA_ARGS__)

#define THO_DISPATCH_CASE_INTEGRAL_TYPES(DISPATCH_CASE, ...) \
  DISPATCH_CASE(c10::ScalarType::Byte, __VA_ARGS__)          \
  DISPATCH_CASE(c10::ScalarType::Char, __VA_ARGS__)          \
  DISPATCH_CASE(c10::ScalarType::Int, __VA_ARGS__)           \
  DISPATCH_CASE(c10::ScalarType::Long, __VA_ARGS__)          \
  DISPATCH_CASE(c10::ScalarType::Short, __VA_ARGS__)

#define THO_DISPATCH_CASE_INTEGRAL_TYPES_AND(DISPATCH_CASE, SCALARTYPE, ...) \
  THO_DISPATCH_CASE_INTEGRAL_TYPES(DISPATCH_CASE, __VA_ARGS__)               \
  DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)

#define THO_DISPATCH_CASE_ALL_TYPES(DISPATCH_CASE, ...)        \
  THO_DISPATCH_CASE_INTEGRAL_TYPES(DISPATCH_CASE, __VA_ARGS__) \
  THO_DISPATCH_CASE_FLOATING_TYPES(DISPATCH_CASE, __VA_ARGS__)

#define THO_DISPATCH_CASE_QINT_TYPES(DISPATCH_CASE_QINT, ...)           \
  DISPATCH_CASE_QINT(c10::ScalarType::QInt8, c10::qint8, __VA_ARGS__)   \
  DISPATCH_CASE_QINT(c10::ScalarType::QUInt8, c10::quint8, __VA_ARGS__) \
  DISPATCH_CASE_QINT(c10::ScalarType::QInt32, c10::qint32, __VA_ARGS__)

#define THO_DISPATCH_CASE_QINT_TYPES_AND(                       \
    DISPATCH_CASE_QINT, DISPATCH_CASE, SCALARTYPE, ...)         \
  THO_DISPATCH_CASE_QINT_TYPES(DISPATCH_CASE_QINT, __VA_ARGS__) \
  DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)

#define THO_DISPATCH_CASE_QINT_BYTE_TYPES(DISPATCH_CASE_QINT, ...)    \
  DISPATCH_CASE_QINT(c10::ScalarType::QInt8, c10::qint8, __VA_ARGS__) \
  DISPATCH_CASE_QINT(c10::ScalarType::QUInt8, c10::quint8, __VA_ARGS__)

#define THO_DISPATCH_CASE_QINT_AND_SUB_BYTE_TYPES(                     \
    QINT_SUB_BYTE_PRIVATE_CASE_TYPE, ...)                              \
  QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                     \
      c10::ScalarType::QInt8,                                          \
      c10::qint8,                                                      \
      CHAR_BIT,                                                        \
      SCHAR_MIN,                                                       \
      SCHAR_MAX,                                                       \
      __VA_ARGS__)                                                     \
  QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                     \
      c10::ScalarType::QUInt8,                                         \
      c10::quint8,                                                     \
      CHAR_BIT,                                                        \
      0,                                                               \
      UCHAR_MAX,                                                       \
      __VA_ARGS__)                                                     \
  QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                     \
      c10::ScalarType::QInt32,                                         \
      c10::qint32,                                                     \
      CHAR_BIT * sizeof(int),                                          \
      INT_MIN,                                                         \
      INT_MAX,                                                         \
      __VA_ARGS__)                                                     \
  QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                     \
      c10::ScalarType::QUInt4x2, c10::quint4x2, 4, 0, 15, __VA_ARGS__) \
  QINT_SUB_BYTE_PRIVATE_CASE_TYPE(                                     \
      c10::ScalarType::QUInt2x4, c10::quint2x4, 2, 0, 3, __VA_ARGS__)

#define THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(DISPATCH_CASE, ...) \
  THO_DISPATCH_CASE_ALL_TYPES(DISPATCH_CASE, __VA_ARGS__)           \
  THO_DISPATCH_CASE_COMPLEX_TYPES(DISPATCH_CASE, __VA_ARGS__)

#define THO_DISPATCH_CASE_ALL_TYPES_AND(DISPATCH_CASE, SCALARTYPE, ...) \
  THO_DISPATCH_CASE_ALL_TYPES(DISPATCH_CASE, __VA_ARGS__)               \
  DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)

#define THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND(                  \
    DISPATCH_CASE, SCALARTYPE, ...)                                   \
  THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(DISPATCH_CASE, __VA_ARGS__) \
  DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)

#define THO_DISPATCH_CASE_ALL_TYPES_AND2(                 \
    DISPATCH_CASE, SCALARTYPE1, SCALARTYPE2, ...)         \
  THO_DISPATCH_CASE_ALL_TYPES(DISPATCH_CASE, __VA_ARGS__) \
  DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                 \
  DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)

#define THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND2(                 \
    DISPATCH_CASE, SCALARTYPE1, SCALARTYPE2, ...)                     \
  THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(DISPATCH_CASE, __VA_ARGS__) \
  DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)

#define THO_DISPATCH_CASE_ALL_TYPES_AND3(                      \
    DISPATCH_CASE, SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, ...) \
  THO_DISPATCH_CASE_ALL_TYPES(DISPATCH_CASE, __VA_ARGS__)      \
  DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                      \
  DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                      \
  DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)

#define THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND3(                 \
    DISPATCH_CASE, SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, ...)        \
  THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(DISPATCH_CASE, __VA_ARGS__) \
  DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)

#define THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND4(                       \
    DISPATCH_CASE, SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, ...) \
  THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(DISPATCH_CASE, __VA_ARGS__)       \
  DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                                   \
  DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                                   \
  DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)                                   \
  DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)

#define THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND5(                 \
    DISPATCH_CASE,                                                    \
    SCALARTYPE1,                                                      \
    SCALARTYPE2,                                                      \
    SCALARTYPE3,                                                      \
    SCALARTYPE4,                                                      \
    SCALARTYPE5,                                                      \
    ...)                                                              \
  THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(DISPATCH_CASE, __VA_ARGS__) \
  DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE5, __VA_ARGS__)

#define THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND6(                 \
    DISPATCH_CASE,                                                    \
    SCALARTYPE1,                                                      \
    SCALARTYPE2,                                                      \
    SCALARTYPE3,                                                      \
    SCALARTYPE4,                                                      \
    SCALARTYPE5,                                                      \
    SCALARTYPE6,                                                      \
    ...)                                                              \
  THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(DISPATCH_CASE, __VA_ARGS__) \
  DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE5, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE6, __VA_ARGS__)

#define THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND7(                 \
    DISPATCH_CASE,                                                    \
    SCALARTYPE1,                                                      \
    SCALARTYPE2,                                                      \
    SCALARTYPE3,                                                      \
    SCALARTYPE4,                                                      \
    SCALARTYPE5,                                                      \
    SCALARTYPE6,                                                      \
    SCALARTYPE7,                                                      \
    ...)                                                              \
  THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(DISPATCH_CASE, __VA_ARGS__) \
  DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE5, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE6, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE7, __VA_ARGS__)

#define THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND8(                 \
    DISPATCH_CASE,                                                    \
    SCALARTYPE1,                                                      \
    SCALARTYPE2,                                                      \
    SCALARTYPE3,                                                      \
    SCALARTYPE4,                                                      \
    SCALARTYPE5,                                                      \
    SCALARTYPE6,                                                      \
    SCALARTYPE7,                                                      \
    SCALARTYPE8,                                                      \
    ...)                                                              \
  THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(DISPATCH_CASE, __VA_ARGS__) \
  DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE4, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE5, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE6, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE7, __VA_ARGS__)                             \
  DISPATCH_CASE(SCALARTYPE8, __VA_ARGS__)

#define THO_DISPATCH_CASE_BIT_TYPES(DISPATCH_CASE, ...) \
  DISPATCH_CASE(c10::ScalarType::Bits1x8, __VA_ARGS__)  \
  DISPATCH_CASE(c10::ScalarType::Bits2x4, __VA_ARGS__)  \
  DISPATCH_CASE(c10::ScalarType::Bits4x2, __VA_ARGS__)  \
  DISPATCH_CASE(c10::ScalarType::Bits8, __VA_ARGS__)    \
  DISPATCH_CASE(c10::ScalarType::Bits16, __VA_ARGS__)

#define THO_DISPATCH_CASE_INDEX_TYPES(PRIVATE_CASE_TYPE_USING_HINT, ...)   \
  PRIVATE_CASE_TYPE_USING_HINT(c10::ScalarType::Int, index_t, __VA_ARGS__) \
  PRIVATE_CASE_TYPE_USING_HINT(c10::ScalarType::Long, index_t, __VA_ARGS__)
