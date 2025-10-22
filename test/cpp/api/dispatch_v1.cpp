#include <gtest/gtest.h>

// This file contains tests for AT_DISPATCH macros.
#include <ATen/Dispatch.h>

// MY_CASE_FUNCTION is called in a case block. For testing, we count
// case matches and ensure that scalar_t/index_t type is defined:
#define MY_CASE_FUNCTION \
  [&] {                  \
    count++;             \
    scalar_t tmp;        \
    (void)tmp;           \
  }

#define DEFINE_ITEM(TYPE, SCALARTYPE) ScalarType::SCALARTYPE,

#define TEST_DISPATCH_V1(M, EXPECTEDCOUNT, CASE_FUNCTION, ...)           \
  TEST(TestDispatchV1, M) {                                              \
    using torch::headeronly::ScalarType;                                 \
    using torch::headeronly::impl::ScalarTypeToCPPTypeT;                 \
    int8_t total_count = 0;                                              \
    int8_t count = 0;                                                    \
    int8_t default_count = 0;                                            \
    for (ScalarType t :                                                  \
         {AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ITEM)}) { \
      total_count++;                                                     \
      try {                                                              \
        M(__VA_ARGS__ t, "test_at_dispatch_v1", CASE_FUNCTION);          \
      } catch (...) {                                                    \
        default_count++; /* counts mismatches */                         \
      }                                                                  \
    }                                                                    \
    EXPECT_EQ(count, EXPECTEDCOUNT);                                     \
    EXPECT_EQ(default_count + count, total_count);                       \
  }

TEST_DISPATCH_V1(AT_DISPATCH_FLOATING_TYPES, 2, MY_CASE_FUNCTION);
TEST_DISPATCH_V1(AT_DISPATCH_FLOATING_TYPES_AND_HALF, 3, MY_CASE_FUNCTION);
TEST_DISPATCH_V1(AT_DISPATCH_REDUCED_FLOATING_TYPES, 2, MY_CASE_FUNCTION);
TEST_DISPATCH_V1(
    AT_DISPATCH_FLOATING_TYPES_AND,
    3,
    MY_CASE_FUNCTION,
    ScalarType::Byte, );

TEST_DISPATCH_V1(
    AT_DISPATCH_FLOATING_TYPES_AND2,
    4,
    MY_CASE_FUNCTION,
    ScalarType::Byte,
    ScalarType::Short, );
TEST_DISPATCH_V1(
    AT_DISPATCH_FLOATING_TYPES_AND3,
    5,
    MY_CASE_FUNCTION,
    ScalarType::Byte,
    ScalarType::Short,
    ScalarType::Int, );
TEST_DISPATCH_V1(
    AT_DISPATCH_FLOATING_TYPES_AND4,
    6,
    MY_CASE_FUNCTION,
    ScalarType::Byte,
    ScalarType::Short,
    ScalarType::Int,
    ScalarType::Long, );
TEST_DISPATCH_V1(
    AT_DISPATCH_FLOATING_TYPES_AND5,
    7,
    MY_CASE_FUNCTION,
    ScalarType::Byte,
    ScalarType::Short,
    ScalarType::Int,
    ScalarType::Long,
    ScalarType::Char, );

TEST_DISPATCH_V1(AT_DISPATCH_COMPLEX_TYPES, 2, MY_CASE_FUNCTION);
TEST_DISPATCH_V1(
    AT_DISPATCH_COMPLEX_TYPES_AND,
    3,
    MY_CASE_FUNCTION,
    ScalarType::Byte, );
TEST_DISPATCH_V1(AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES, 4, MY_CASE_FUNCTION, );
TEST_DISPATCH_V1(
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1,
    5,
    MY_CASE_FUNCTION,
    ScalarType::Byte, );
TEST_DISPATCH_V1(
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2,
    6,
    MY_CASE_FUNCTION,
    ScalarType::Byte,
    ScalarType::Short, );
TEST_DISPATCH_V1(
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3,
    7,
    MY_CASE_FUNCTION,
    ScalarType::Byte,
    ScalarType::Short,
    ScalarType::Int, );
TEST_DISPATCH_V1(
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND4,
    8,
    MY_CASE_FUNCTION,
    ScalarType::Byte,
    ScalarType::Short,
    ScalarType::Int,
    ScalarType::Long, );
TEST_DISPATCH_V1(
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND5,
    9,
    MY_CASE_FUNCTION,
    ScalarType::Byte,
    ScalarType::Short,
    ScalarType::Int,
    ScalarType::Long,
    ScalarType::Char, );
TEST_DISPATCH_V1(
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND6,
    10,
    MY_CASE_FUNCTION,
    ScalarType::Byte,
    ScalarType::Short,
    ScalarType::Int,
    ScalarType::Long,
    ScalarType::Char,
    ScalarType::Half, );

TEST_DISPATCH_V1(AT_DISPATCH_INTEGRAL_TYPES, 5, MY_CASE_FUNCTION);
TEST_DISPATCH_V1(
    AT_DISPATCH_INTEGRAL_TYPES_AND,
    6,
    MY_CASE_FUNCTION,
    ScalarType::Float, );
TEST_DISPATCH_V1(AT_DISPATCH_ALL_TYPES, 7, MY_CASE_FUNCTION);

TEST_DISPATCH_V1(AT_DISPATCH_QINT_TYPES, 3, MY_CASE_FUNCTION);

TEST_DISPATCH_V1(
    AT_DISPATCH_QINT_TYPES_AND,
    4,
    MY_CASE_FUNCTION,
    ScalarType::Float, );

TEST_DISPATCH_V1(AT_DISPATCH_QINT_BYTE_TYPES, 2, MY_CASE_FUNCTION);
TEST_DISPATCH_V1(AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES, 5, MY_CASE_FUNCTION);

TEST_DISPATCH_V1(AT_DISPATCH_ALL_TYPES_AND_COMPLEX, 9, MY_CASE_FUNCTION);
TEST_DISPATCH_V1(
    AT_DISPATCH_ALL_TYPES_AND,
    8,
    MY_CASE_FUNCTION,
    ScalarType::Half, );
TEST_DISPATCH_V1(
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND,
    10,
    MY_CASE_FUNCTION,
    ScalarType::Half, );
TEST_DISPATCH_V1(
    AT_DISPATCH_ALL_TYPES_AND2,
    9,
    MY_CASE_FUNCTION,
    ScalarType::Half,
    ScalarType::ComplexHalf, );
TEST_DISPATCH_V1(
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2,
    11,
    MY_CASE_FUNCTION,
    ScalarType::Half,
    ScalarType::BFloat16, );
TEST_DISPATCH_V1(
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3,
    12,
    MY_CASE_FUNCTION,
    ScalarType::Half,
    ScalarType::BFloat16,
    ScalarType::ComplexHalf, );
TEST_DISPATCH_V1(
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4,
    13,
    MY_CASE_FUNCTION,
    ScalarType::Half,
    ScalarType::BFloat16,
    ScalarType::ComplexHalf,
    ScalarType::Bits1x8, );
TEST_DISPATCH_V1(
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND5,
    14,
    MY_CASE_FUNCTION,
    ScalarType::Half,
    ScalarType::BFloat16,
    ScalarType::ComplexHalf,
    ScalarType::Bits1x8,
    ScalarType::Bits2x4, );
TEST_DISPATCH_V1(
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND6,
    15,
    MY_CASE_FUNCTION,
    ScalarType::Half,
    ScalarType::BFloat16,
    ScalarType::ComplexHalf,
    ScalarType::Bits1x8,
    ScalarType::Bits2x4,
    ScalarType::Bits4x2, );
TEST_DISPATCH_V1(
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND7,
    16,
    MY_CASE_FUNCTION,
    ScalarType::Half,
    ScalarType::BFloat16,
    ScalarType::ComplexHalf,
    ScalarType::Bits1x8,
    ScalarType::Bits2x4,
    ScalarType::Bits4x2,
    ScalarType::Bits8, );
TEST_DISPATCH_V1(
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND8,
    17,
    MY_CASE_FUNCTION,
    ScalarType::Half,
    ScalarType::BFloat16,
    ScalarType::ComplexHalf,
    ScalarType::Bits1x8,
    ScalarType::Bits2x4,
    ScalarType::Bits4x2,
    ScalarType::Bits8,
    ScalarType::Bits16, );
TEST_DISPATCH_V1(AT_DISPATCH_BIT_TYPES, 5, MY_CASE_FUNCTION);

#define MY_INDEX_FUNCTION \
  [&] {                   \
    count++;              \
    index_t tmp;          \
    (void)tmp;            \
  }

TEST_DISPATCH_V1(AT_DISPATCH_INDEX_TYPES, 2, MY_INDEX_FUNCTION);

#undef DEFINE_ITEM
