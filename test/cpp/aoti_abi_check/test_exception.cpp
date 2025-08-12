#include <gtest/gtest.h>

#include <torch/headeronly/util/Exception.h>

namespace torch {
namespace aot_inductor {

TEST(TestExceptions, TestStdTorchCheck) {
  EXPECT_NO_THROW(STD_TORCH_CHECK(true, "dummy true message"));
  EXPECT_NO_THROW(STD_TORCH_CHECK(true, "dummy ", "true ", "message"));
  EXPECT_THROW(
      STD_TORCH_CHECK(false, "dummy false message"), std::runtime_error);
  EXPECT_THROW(
      STD_TORCH_CHECK(false, "dummy ", "false ", "message"),
      std::runtime_error);
}

} // namespace aot_inductor
} // namespace torch
