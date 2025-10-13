#include <gtest/gtest.h>

#include <torch/headeronly/core/ScalarType.h>

/* skip runtime exception in default, just count the number of types
   which support has not been implemented */
#define AT_DISPATCH_DEFAULT(NAME, ENUMTYPE) \
  (void)NAME;                               \
  default_count++;
#include <torch/headeronly/core/Dispatch.h>

#define DEFINE_ITEM(TYPE, SCALARTYPE) ScalarType::SCALARTYPE,

#define TEST_DISPATCH(M, EXPECTEDCOUNT, ...)                             \
  TEST(TestDispatch, M) {                                                \
    using torch::headeronly::ScalarType;                                 \
    int8_t total_count = 0;                                              \
    int8_t count = 0;                                                    \
    int8_t default_count = 0;                                            \
    for (ScalarType t :                                                  \
         {AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ITEM)}) { \
      total_count++;                                                     \
      M(__VA_ARGS__ t, "test_dispatch", [&] { count++; });               \
    }                                                                    \
    EXPECT_EQ(count, EXPECTEDCOUNT);                                     \
    EXPECT_EQ(default_count + count, total_count);                       \
  }

TEST_DISPATCH(AT_DISPATCH_FLOATING_TYPES, 2);
TEST_DISPATCH(AT_DISPATCH_FLOATING_TYPES_AND_HALF, 3);
TEST_DISPATCH(AT_DISPATCH_REDUCED_FLOATING_TYPES, 2);
TEST_DISPATCH(AT_DISPATCH_FLOATING_TYPES_AND, 3, ScalarType::Int, );
TEST_DISPATCH(
    AT_DISPATCH_FLOATING_TYPES_AND2,
    4,
    ScalarType::Int,
    ScalarType::Long, );
TEST_DISPATCH(
    AT_DISPATCH_FLOATING_TYPES_AND3,
    5,
    ScalarType::Half,
    ScalarType::Int,
    ScalarType::Long, );
TEST_DISPATCH(
    AT_DISPATCH_FLOATING_TYPES_AND4,
    6,
    ScalarType::Half,
    ScalarType::Short,
    ScalarType::Int,
    ScalarType::Long, );
TEST_DISPATCH(
    AT_DISPATCH_FLOATING_TYPES_AND5,
    7,
    ScalarType::Half,
    ScalarType::Char,
    ScalarType::Short,
    ScalarType::Int,
    ScalarType::Long, );
TEST_DISPATCH(AT_DISPATCH_COMPLEX_TYPES, 2);
TEST_DISPATCH(AT_DISPATCH_COMPLEX_TYPES_AND, 3, ScalarType::Float, );
TEST_DISPATCH(AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES, 4);
TEST_DISPATCH(
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1,
    5,
    ScalarType::Int, );
TEST_DISPATCH(
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2,
    6,
    ScalarType::Int,
    ScalarType::Long, );
TEST_DISPATCH(
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3,
    7,
    ScalarType::Short,
    ScalarType::Int,
    ScalarType::Long, );
TEST_DISPATCH(
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND4,
    8,
    ScalarType::Char,
    ScalarType::Short,
    ScalarType::Int,
    ScalarType::Long, );
TEST_DISPATCH(
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND5,
    9,
    ScalarType::Half,
    ScalarType::Char,
    ScalarType::Short,
    ScalarType::Int,
    ScalarType::Long, );
TEST_DISPATCH(
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND6,
    10,
    ScalarType::Byte,
    ScalarType::Half,
    ScalarType::Char,
    ScalarType::Short,
    ScalarType::Int,
    ScalarType::Long, );
TEST_DISPATCH(AT_DISPATCH_INTEGRAL_TYPES, 5);
TEST_DISPATCH(AT_DISPATCH_INTEGRAL_TYPES_AND, 6, ScalarType::Float, );
TEST_DISPATCH(AT_DISPATCH_ALL_TYPES, 7);
TEST_DISPATCH(AT_DISPATCH_ALL_TYPES_AND_COMPLEX, 9);
TEST_DISPATCH(AT_DISPATCH_ALL_TYPES_AND, 8, ScalarType::ComplexFloat, );
TEST_DISPATCH(
    AT_DISPATCH_ALL_TYPES_AND2,
    9,
    ScalarType::ComplexFloat,
    ScalarType::ComplexDouble, );
TEST_DISPATCH(AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND, 10, ScalarType::Half, );
TEST_DISPATCH(
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2,
    11,
    ScalarType::Half,
    ScalarType::BFloat16, );
TEST_DISPATCH(
    AT_DISPATCH_ALL_TYPES_AND3,
    10,
    ScalarType::Half,
    ScalarType::ComplexFloat,
    ScalarType::ComplexDouble, );
TEST_DISPATCH(
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3,
    12,
    ScalarType::Half,
    ScalarType::BFloat16,
    ScalarType::ComplexHalf, );
TEST_DISPATCH(
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4,
    13,
    ScalarType::Half,
    ScalarType::ComplexHalf,
    ScalarType::BFloat16,
    ScalarType::Bits1x8, );
TEST_DISPATCH(
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND5,
    14,
    ScalarType::Half,
    ScalarType::ComplexHalf,
    ScalarType::BFloat16,
    ScalarType::Bits2x4,
    ScalarType::Bits4x2, );
TEST_DISPATCH(
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND6,
    15,
    ScalarType::Half,
    ScalarType::ComplexHalf,
    ScalarType::BFloat16,
    ScalarType::Bits1x8,
    ScalarType::Bits2x4,
    ScalarType::Bits4x2, );
TEST_DISPATCH(
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND7,
    16,
    ScalarType::Half,
    ScalarType::ComplexHalf,
    ScalarType::BFloat16,
    ScalarType::Bits1x8,
    ScalarType::Bits2x4,
    ScalarType::Bits4x2,
    ScalarType::Bits8, );
TEST_DISPATCH(
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND8,
    17,
    ScalarType::Half,
    ScalarType::ComplexHalf,
    ScalarType::BFloat16,
    ScalarType::Bits1x8,
    ScalarType::Bits2x4,
    ScalarType::Bits4x2,
    ScalarType::Bits8,
    ScalarType::Bits16, );
TEST_DISPATCH(AT_DISPATCH_BIT_TYPES, 5);
TEST_DISPATCH(AT_DISPATCH_INDEX_TYPES, 2);
