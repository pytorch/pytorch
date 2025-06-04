#include <gtest/gtest.h>

#include <torch/standalone/macros/Export.h>
#include <torch/standalone/util/Exception.h>

namespace torch {
namespace aot_inductor {

TORCH_STANDALONE_API bool equal(int a, int b) {
  return a == b;
}

TEST(TestMacros, TestAPI) {
  EXPECT_TRUE(equal(1, 1));
  EXPECT_FALSE(equal(1, 2));
}

TEST(TestMacros, TestTorchCheck) {
  EXPECT_NO_THROW(TORCH_STANDALONE_CHECK(true, "dummy true message"));
  EXPECT_NO_THROW(TORCH_STANDALONE_CHECK(true, "dummy ", "true ", "message"));
  EXPECT_THROW(
      TORCH_STANDALONE_CHECK(false, "dummy false message"), std::runtime_error);
  EXPECT_THROW(
      TORCH_STANDALONE_CHECK(false, "dummy ", "false ", "message"),
      std::runtime_error);
}

} // namespace aot_inductor
} // namespace torch
