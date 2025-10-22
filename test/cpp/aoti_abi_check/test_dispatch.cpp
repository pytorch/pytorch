#include <gtest/gtest.h>

#if defined(ENABLE_RECORD_KERNEL_FUNCTION_DTYPE) || \
    defined(TEMPLATE_SELECTIVE_BUILD)
#error "selective build not supported when testing AT_DISPATCH macros"
#endif

#include <torch/headeronly/core/Dispatch.h>
#include <torch/headeronly/core/Dispatch_v2.h>

// MY_PRIVATE_CHECK_SELECTIVE_BUILD is a prelude to case block. For
// testing, we do nothing:
#define MY_PRIVATE_CHECK_SELECTIVE_BUILD(enum_type) /* empty */

#define MY_PRIVATE_CASE_TYPE_USING_HINT(...) \
  AT_PRIVATE_CASE_TYPE_USING_HINT_TMPL(      \
      MY_PRIVATE_CHECK_SELECTIVE_BUILD, __VA_ARGS__)

#define MY_DISPATCH_CASE(...) \
  AT_DISPATCH_CASE_TMPL(MY_PRIVATE_CASE_TYPE_USING_HINT, __VA_ARGS__)

// MY_RECORD_KERNEL_FUNCTION_DTYPE is a prelude to switch
// statement. For testing, we just avoid unused variable warning:
#define MY_RECORD_KERNEL_FUNCTION_DTYPE(DISPATCHNAME, ENUMTYPE) \
  (void)DISPATCHNAME

// MY_CHECK_NOT_IMPLEMENTED is called in switch default block. For
// testing, we count case mismatches:
#define MY_CHECK_NOT_IMPLEMENTED(...) default_count++

#define MY_DISPATCH_SWITCH(...) \
  AT_DISPATCH_SWITCH_TMPL(      \
      MY_RECORD_KERNEL_FUNCTION_DTYPE, MY_CHECK_NOT_IMPLEMENTED, __VA_ARGS__)

// MY_CASE_FUNCTION is called in a case block. For testing, we count
// case matches and ensure that scalar_t/index_t type is defined:
#define MY_CASE_FUNCTION \
  [&] {                  \
    count++;             \
    scalar_t tmp;        \
    (void)tmp;           \
  }

#define DEFINE_ITEM(TYPE, SCALARTYPE) ScalarType::SCALARTYPE,

// Test V2 dispatch macros:

#define MY_DISPATCH_V2(TYPE, NAME, BODY, ...) \
  AT_DISPATCH_V2_TMPL(                        \
      MY_DISPATCH_SWITCH,                     \
      MY_DISPATCH_CASE,                       \
      TYPE,                                   \
      NAME,                                   \
      AT_WRAP(BODY),                          \
      __VA_ARGS__)

#define TEST_DISPATCH_V2(NAME, EXPECTEDCOUNT, ...)                             \
  TEST(TestDispatchV2, NAME) {                                                 \
    using torch::headeronly::ScalarType;                                       \
    using torch::headeronly::impl::ScalarTypeToCPPTypeT;                       \
    int8_t total_count = 0;                                                    \
    int8_t count = 0;                                                          \
    int8_t default_count = 0;                                                  \
    for (ScalarType t :                                                        \
         {AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ITEM)}) {       \
      total_count++;                                                           \
      MY_DISPATCH_V2(t, "test_my_dispatch_v2", MY_CASE_FUNCTION, __VA_ARGS__); \
    }                                                                          \
    EXPECT_EQ(count, EXPECTEDCOUNT);                                           \
    EXPECT_EQ(default_count + count, total_count);                             \
  }

TEST_DISPATCH_V2(AT_FLOAT8_TYPES_, 5, AT_FLOAT8_TYPES);
TEST_DISPATCH_V2(AT_INTEGRAL_TYPES_, 5, AT_INTEGRAL_TYPES);
TEST_DISPATCH_V2(AT_FLOATING_TYPES_, 2, AT_FLOATING_TYPES);
TEST_DISPATCH_V2(AT_BAREBONES_UNSIGNED_TYPES_, 3, AT_BAREBONES_UNSIGNED_TYPES);
TEST_DISPATCH_V2(AT_INTEGRAL_TYPES_V2_, 8, AT_INTEGRAL_TYPES_V2);
TEST_DISPATCH_V2(AT_COMPLEX_TYPES_, 2, AT_COMPLEX_TYPES);
TEST_DISPATCH_V2(AT_QINT_TYPES_, 3, AT_QINT_TYPES);
TEST_DISPATCH_V2(AT_ALL_TYPES_, 7, AT_ALL_TYPES);
TEST_DISPATCH_V2(AT_ALL_TYPES_AND_COMPLEX_, 9, AT_ALL_TYPES_AND_COMPLEX);

#define TEST_DISPATCH_TMPL(M, EXPECTEDCOUNT, ...)                        \
  TEST(TestDispatch, M) {                                                \
    using torch::headeronly::ScalarType;                                 \
    using torch::headeronly::impl::ScalarTypeToCPPTypeT;                 \
    int8_t total_count = 0;                                              \
    int8_t count = 0;                                                    \
    int8_t default_count = 0;                                            \
    for (ScalarType t :                                                  \
         {AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ITEM)}) { \
      total_count++;                                                     \
      MY_DISPATCH_SWITCH(t, "test_my_dispatch", M(__VA_ARGS__));         \
    }                                                                    \
    EXPECT_EQ(count, EXPECTEDCOUNT);                                     \
    EXPECT_EQ(default_count + count, total_count);                       \
  }

#define TEST_DISPATCH(M, EXPECTEDCOUNT, ...) \
  TEST_DISPATCH_TMPL(M, EXPECTEDCOUNT, MY_DISPATCH_CASE, __VA_ARGS__)

#define MY_DISPATCH_CASE_QINT(...) \
  THO_DISPATCH_CASE_QINT(MY_PRIVATE_CHECK_SELECTIVE_BUILD, __VA_ARGS__)

#define TEST_DISPATCH_QINT(M, EXPECTEDCOUNT, ...) \
  TEST_DISPATCH_TMPL(M, EXPECTEDCOUNT, MY_DISPATCH_CASE_QINT, __VA_ARGS__)

#define TEST_DISPATCH_QINT_AND(M, EXPECTEDCOUNT, ...) \
  TEST_DISPATCH_TMPL(                                 \
      M, EXPECTEDCOUNT, MY_DISPATCH_CASE_QINT, MY_DISPATCH_CASE, __VA_ARGS__)

#define MY_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(...) \
  THO_QINT_SUB_BYTE_PRIVATE_CASE_TYPE(          \
      MY_PRIVATE_CHECK_SELECTIVE_BUILD, __VA_ARGS__)

#define TEST_DISPATCH_QINT_AND_SUB_BYTE(M, EXPECTEDCOUNT, ...) \
  TEST_DISPATCH_TMPL(                                          \
      M, EXPECTEDCOUNT, MY_QINT_SUB_BYTE_PRIVATE_CASE_TYPE, __VA_ARGS__)

TEST_DISPATCH(THO_DISPATCH_CASE_FLOATING_TYPES, 2, MY_CASE_FUNCTION)
TEST_DISPATCH(THO_DISPATCH_CASE_FLOATING_TYPES_AND_HALF, 3, MY_CASE_FUNCTION);
TEST_DISPATCH(THO_DISPATCH_CASE_REDUCED_FLOATING_TYPES, 2, MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_FLOATING_TYPES_AND,
    3,
    ScalarType::Byte,
    MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_FLOATING_TYPES_AND2,
    4,
    ScalarType::Byte,
    ScalarType::Short,
    MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_FLOATING_TYPES_AND3,
    5,
    ScalarType::Byte,
    ScalarType::Short,
    ScalarType::Int,
    MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_FLOATING_TYPES_AND4,
    6,
    ScalarType::Byte,
    ScalarType::Short,
    ScalarType::Int,
    ScalarType::Long,
    MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_FLOATING_TYPES_AND5,
    7,
    ScalarType::Byte,
    ScalarType::Short,
    ScalarType::Int,
    ScalarType::Long,
    ScalarType::Char,
    MY_CASE_FUNCTION);

TEST_DISPATCH(THO_DISPATCH_CASE_COMPLEX_TYPES, 2, MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_COMPLEX_TYPES_AND,
    3,
    ScalarType::Byte,
    MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES,
    4,
    MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND1,
    5,
    ScalarType::Byte,
    MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND2,
    6,
    ScalarType::Byte,
    ScalarType::Short,
    MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND3,
    7,
    ScalarType::Byte,
    ScalarType::Short,
    ScalarType::Int,
    MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND4,
    8,
    ScalarType::Byte,
    ScalarType::Short,
    ScalarType::Int,
    ScalarType::Long,
    MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND5,
    9,
    ScalarType::Byte,
    ScalarType::Short,
    ScalarType::Int,
    ScalarType::Long,
    ScalarType::Char,
    MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES_AND6,
    10,
    ScalarType::Byte,
    ScalarType::Short,
    ScalarType::Int,
    ScalarType::Long,
    ScalarType::Char,
    ScalarType::Half,
    MY_CASE_FUNCTION);

TEST_DISPATCH(THO_DISPATCH_CASE_INTEGRAL_TYPES, 5, MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_INTEGRAL_TYPES_AND,
    6,
    ScalarType::Float,
    MY_CASE_FUNCTION);
TEST_DISPATCH(THO_DISPATCH_CASE_ALL_TYPES, 7, MY_CASE_FUNCTION);

TEST_DISPATCH_QINT(THO_DISPATCH_CASE_QINT_TYPES, 3, MY_CASE_FUNCTION);

TEST_DISPATCH_QINT_AND(
    THO_DISPATCH_CASE_QINT_TYPES_AND,
    4,
    ScalarType::Float,
    MY_CASE_FUNCTION);

TEST_DISPATCH_QINT(THO_DISPATCH_CASE_QINT_BYTE_TYPES, 2, MY_CASE_FUNCTION);
TEST_DISPATCH_QINT_AND_SUB_BYTE(
    THO_DISPATCH_CASE_QINT_AND_SUB_BYTE_TYPES,
    5,
    MY_CASE_FUNCTION);

TEST_DISPATCH(THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX, 9, MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_ALL_TYPES_AND,
    8,
    ScalarType::Half,
    MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND,
    10,
    ScalarType::Half,
    MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_ALL_TYPES_AND2,
    9,
    ScalarType::Half,
    ScalarType::ComplexHalf,
    MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND2,
    11,
    ScalarType::Half,
    ScalarType::BFloat16,
    MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND3,
    12,
    ScalarType::Half,
    ScalarType::BFloat16,
    ScalarType::ComplexHalf,
    MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND4,
    13,
    ScalarType::Half,
    ScalarType::BFloat16,
    ScalarType::ComplexHalf,
    ScalarType::Bits1x8,
    MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND5,
    14,
    ScalarType::Half,
    ScalarType::BFloat16,
    ScalarType::ComplexHalf,
    ScalarType::Bits1x8,
    ScalarType::Bits2x4,
    MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND6,
    15,
    ScalarType::Half,
    ScalarType::BFloat16,
    ScalarType::ComplexHalf,
    ScalarType::Bits1x8,
    ScalarType::Bits2x4,
    ScalarType::Bits4x2,
    MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND7,
    16,
    ScalarType::Half,
    ScalarType::BFloat16,
    ScalarType::ComplexHalf,
    ScalarType::Bits1x8,
    ScalarType::Bits2x4,
    ScalarType::Bits4x2,
    ScalarType::Bits8,
    MY_CASE_FUNCTION);
TEST_DISPATCH(
    THO_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX_AND8,
    17,
    ScalarType::Half,
    ScalarType::BFloat16,
    ScalarType::ComplexHalf,
    ScalarType::Bits1x8,
    ScalarType::Bits2x4,
    ScalarType::Bits4x2,
    ScalarType::Bits8,
    ScalarType::Bits16,
    MY_CASE_FUNCTION);
TEST_DISPATCH(THO_DISPATCH_CASE_BIT_TYPES, 5, MY_CASE_FUNCTION);

// THO_DISPATCH_CASE_INDEX_TYPES already contains case blocks, so
// ignoring DISPATCH_CASE:
#define MY_DISPATCH_CASE_INDEX_TYPES(DISPATCH_CASE, ...) \
  THO_DISPATCH_CASE_INDEX_TYPES(MY_PRIVATE_CASE_TYPE_USING_HINT, __VA_ARGS__)
#define MY_INDEX_CASE_FUNCTION \
  [&] {                        \
    count++;                   \
    index_t tmp;               \
    (void)tmp;                 \
  }

TEST_DISPATCH(MY_DISPATCH_CASE_INDEX_TYPES, 2, MY_INDEX_CASE_FUNCTION);
#undef MY_DISPATCH_CASE_INDEX_TYPES

#undef DEFINE_ITEM
