#include <gtest/gtest.h>

#include <torch/headeronly/core/ScalarType.h>

TEST(TestScalarType, toString) {
  using torch::headeronly::ScalarType;

#define DEFINE_CHECK(_, name) EXPECT_EQ(toString(ScalarType::name), #name);
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CHECK);
#undef DEFINE_CHECK
}
