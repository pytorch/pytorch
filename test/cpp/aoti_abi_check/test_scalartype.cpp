#include <gtest/gtest.h>

#include <torch/headeronly/core/ScalarType.h>

TEST(TestScalarType, ScalarTypeToCPPTypeT) {
  using torch::headeronly::ScalarType;
  using torch::headeronly::impl::ScalarTypeToCPPTypeT;

#define DEFINE_CHECK(TYPE, SCALARTYPE) \
  EXPECT_EQ(typeid(ScalarTypeToCPPTypeT<ScalarType::SCALARTYPE>), typeid(TYPE));

  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CHECK);
#undef DEFINE_CHECK
}

TEST(TestScalarType, CppTypeToScalarType) {
  using torch::headeronly::CppTypeToScalarType;
  using torch::headeronly::ScalarType;

#define DEFINE_CHECK(TYPE, SCALARTYPE) \
  EXPECT_EQ(CppTypeToScalarType<TYPE>::value, ScalarType::SCALARTYPE);

  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CHECK);
#undef DEFINE_CHECK
}

#define DEFINE_CHECK(TYPE, SCALARTYPE)                                       \
  {                                                                          \
    EXPECT_EQ(                                                               \
        typeid(ScalarTypeToCPPTypeT<ScalarType::SCALARTYPE>), typeid(TYPE)); \
    count++;                                                                 \
  }

#define TEST_FORALL(M, EXPECTEDCOUNT, ...)               \
  TEST(TestScalarType, M) {                              \
    using torch::headeronly::ScalarType;                 \
    using torch::headeronly::impl::ScalarTypeToCPPTypeT; \
    int8_t count = 0;                                    \
    M(__VA_ARGS__ DEFINE_CHECK);                         \
    EXPECT_EQ(count, EXPECTEDCOUNT);                     \
  }

TEST_FORALL(AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF_F8NZ, 14)
TEST_FORALL(AT_FORALL_SCALAR_TYPES_WITH_COMPLEX, 18)
TEST_FORALL(AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS, 46)
TEST_FORALL(AT_FORALL_INT_TYPES, 5)
TEST_FORALL(AT_FORALL_SCALAR_TYPES, 7)
TEST_FORALL(AT_FORALL_SCALAR_TYPES_AND, 8, Bool, )
TEST_FORALL(AT_FORALL_SCALAR_TYPES_AND2, 9, Bool, Half, )
TEST_FORALL(AT_FORALL_SCALAR_TYPES_AND3, 10, Bool, Half, ComplexFloat, )
TEST_FORALL(
    AT_FORALL_SCALAR_TYPES_AND7,
    14,
    Bool,
    Half,
    ComplexHalf,
    ComplexFloat,
    ComplexDouble,
    UInt16,
    UInt32, )
TEST_FORALL(AT_FORALL_QINT_TYPES, 5)
TEST_FORALL(AT_FORALL_FLOAT8_TYPES, 5)
TEST_FORALL(AT_FORALL_COMPLEX_TYPES, 2)

#undef DEFINE_CHECK
#undef TEST_FORALL

TEST(TestScalarType, toString) {
  using torch::headeronly::ScalarType;

#define DEFINE_CHECK(_, name) EXPECT_EQ(toString(ScalarType::name), #name);
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CHECK);
#undef DEFINE_CHECK
}

TEST(TestScalarType, operator_left_shift) {
  using torch::headeronly::ScalarType;

#define DEFINE_CHECK(_, name)   \
  {                             \
    std::stringstream ss;       \
    ss << ScalarType::name;     \
    EXPECT_EQ(ss.str(), #name); \
  }
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CHECK);
#undef DEFINE_CHECK
}

TEST(TestScalarType, toUnderlying) {
  using torch::headeronly::ScalarType;
  using torch::headeronly::toUnderlying;

  EXPECT_EQ(toUnderlying(ScalarType::QUInt8), ScalarType::Byte);
  EXPECT_EQ(toUnderlying(ScalarType::QUInt4x2), ScalarType::Byte);
  EXPECT_EQ(toUnderlying(ScalarType::QUInt2x4), ScalarType::Byte);
  EXPECT_EQ(toUnderlying(ScalarType::QInt8), ScalarType::Char);
  EXPECT_EQ(toUnderlying(ScalarType::QInt32), ScalarType::Int);
#define DEFINE_CHECK(_, name) \
  EXPECT_EQ(toUnderlying(ScalarType::name), ScalarType::name);
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CHECK);
  AT_FORALL_FLOAT8_TYPES(DEFINE_CHECK);
#undef DEFINE_CHECK
}

TEST(TestScalarType, isQIntType) {
  using torch::headeronly::isQIntType;
  using torch::headeronly::ScalarType;
#define DEFINE_CHECK(_, name) EXPECT_TRUE(isQIntType(ScalarType::name));
  AT_FORALL_QINT_TYPES(DEFINE_CHECK);
#undef DEFINE_CHECK
#define DEFINE_CHECK(_, name) EXPECT_FALSE(isQIntType(ScalarType::name));
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CHECK);
#undef DEFINE_CHECK
}
