#include <gtest/gtest.h>
#include <torch/nativert/common/Conv.h>

namespace torch::nativert {

TEST(TryToTest, Int64T) {
  // Test valid conversions
  EXPECT_EQ(tryTo<int64_t>("123"), 123);
  EXPECT_EQ(tryTo<int64_t>("456"), 456);
  // Test invalid conversions
  EXPECT_FALSE(tryTo<int64_t>("0x123").has_value());
  EXPECT_FALSE(tryTo<int64_t>("123abc").has_value());
  EXPECT_FALSE(tryTo<int64_t>("123.45").has_value());
  EXPECT_FALSE(tryTo<int64_t>("").has_value());
  // Test overflow
  EXPECT_FALSE(tryTo<int64_t>("12345678901234567890").has_value());
}

TEST(TryToTest, Double) {
  // Test valid conversions
  EXPECT_EQ(tryTo<double>("123.45"), 123.45);
  EXPECT_EQ(tryTo<double>("-123.45"), -123.45);
  // Test invalid conversions
  EXPECT_FALSE(tryTo<double>("0x123").has_value());
  EXPECT_FALSE(tryTo<double>("123abc").has_value());
  EXPECT_FALSE(tryTo<double>("").has_value());
  // Test overflow
  EXPECT_FALSE(tryTo<double>("1e309").has_value());
}

} // namespace torch::nativert
