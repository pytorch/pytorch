#include <fmt/format.h>
#include <gtest/gtest.h>
#include <torch/csrc/deploy/unity/tests/test_unity.h>
#include <torch/csrc/deploy/unity/unity.h>

namespace torch {
namespace deploy {

TEST(UnityTest, TestUnitySum) {
  // use a different path for unit test. Normally don't specify the path will
  // use the default one.
  Unity unity(2, TEST_PYTHON_APP_DIR);
  unity.runMainModule();

  auto I = unity.getInterpreterManager().acquireOne();
  auto result = I.global("sum", "func")({1, 2, 3, 4});
  EXPECT_EQ(10, result.toIValue().toInt());
}

} // namespace deploy
} // namespace torch
