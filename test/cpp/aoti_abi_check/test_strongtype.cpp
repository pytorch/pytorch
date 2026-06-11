#include <gtest/gtest.h>

#include <torch/headeronly/util/strong_type.h>

namespace {
using MyInt = strong::type<int, struct MyIntTag, strong::equality>;
} // namespace

TEST(TestStrongType, TestStrongType) {
  MyInt a{1};
  MyInt b{1};
  MyInt c{2};
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a == c);
  EXPECT_EQ(value_of(a), 1);
}
