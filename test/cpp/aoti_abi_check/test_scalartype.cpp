#include <gtest/gtest.h>

#include <torch/headeronly/core/ScalarType.h>

TEST(TestScalarType, AT_FORALL) {
  using torch::headeronly::ScalarType;

#define DEFINE_CHECK(TYPE, SCALARTYPE)                               \
  {                                                                  \
    TYPE value{}; /* sanity check that type is defined */            \
    (void)value; /* avoids unused variable warning */                \
    EXPECT_EQ(                                                       \
        toString(ScalarType::SCALARTYPE),                            \
        #SCALARTYPE); /* sanity check that scalar type is defined */ \
    count++;                                                         \
  }

#define CHECK_FORALL_MACRO(M, EXPECTEDCOUNT) \
  {                                          \
    int8_t count = 0;                        \
    M(DEFINE_CHECK);                         \
    EXPECT_EQ(count, EXPECTEDCOUNT);         \
  }

#define CHECK_FORALL_AND_MACRO(M, EXPECTEDCOUNT, ...) \
  {                                                   \
    int8_t count = 0;                                 \
    M(__VA_ARGS__, DEFINE_CHECK);                     \
    EXPECT_EQ(count, EXPECTEDCOUNT);                  \
  }

  CHECK_FORALL_MACRO(
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF_F8NZ, 14);
  CHECK_FORALL_MACRO(AT_FORALL_SCALAR_TYPES_WITH_COMPLEX, 18);
  CHECK_FORALL_MACRO(AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS, 46);
  CHECK_FORALL_MACRO(AT_FORALL_INT_TYPES, 5);
  CHECK_FORALL_MACRO(AT_FORALL_SCALAR_TYPES, 7);
  CHECK_FORALL_AND_MACRO(AT_FORALL_SCALAR_TYPES_AND, 8, Bool);
  CHECK_FORALL_AND_MACRO(AT_FORALL_SCALAR_TYPES_AND2, 9, Bool, Half);
  CHECK_FORALL_AND_MACRO(
      AT_FORALL_SCALAR_TYPES_AND3, 10, Bool, Half, ComplexFloat);
  CHECK_FORALL_AND_MACRO(
      AT_FORALL_SCALAR_TYPES_AND7,
      14,
      Bool,
      Half,
      ComplexHalf,
      ComplexFloat,
      ComplexDouble,
      UInt16,
      UInt32);

#undef CHECK_FORALL_MACRO
#undef CHECK_FORALL_AND_MACRO
#undef DEFINE_CHECK
}

TEST(TestScalarType, toString) {
  using torch::headeronly::ScalarType;

#define DEFINE_CHECK(_, name) EXPECT_EQ(toString(ScalarType::name), #name);
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CHECK);
#undef DEFINE_CHECK
}
