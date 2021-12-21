#include <fmt/format.h>
#include <gtest/gtest.h>
#include <torch/csrc/deploy/unity/tests/test_unity.h>
#include <torch/csrc/deploy/unity/xar_environment.h>

namespace torch {
namespace deploy {

const char* exePath = nullptr;

TEST(UnityTest, TestUnitySimpleModel) {
  // use a different path for unit test. Normally don't specify the path will
  // use the default one.
  mkdtemp(TEST_PYTHON_APP_DIR_TEMP);
  std::shared_ptr<Environment> env =
      std::make_shared<XarEnvironment>(exePath, TEST_PYTHON_APP_DIR_TEMP);
  InterpreterManager m(2, env);

  auto I = m.acquireOne();

  auto noArgs = at::ArrayRef<Obj>();
  auto input = I.global("torch", "randn")({32, 256});
  auto model = I.global("simple_model", "SimpleModel")(noArgs);

  auto output = model({input}); // implicitly calls model's forward method
  EXPECT_EQ(2, output.attr("shape").attr("__len__")(noArgs).toIValue().toInt());
  EXPECT_EQ(
      32, output.attr("shape").attr("__getitem__")({0}).toIValue().toInt());
  EXPECT_EQ(
      10, output.attr("shape").attr("__getitem__")({1}).toIValue().toInt());
}

} // namespace deploy
} // namespace torch

int main(int argc, char** argv) {
  torch::deploy::exePath = argv[0];
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
