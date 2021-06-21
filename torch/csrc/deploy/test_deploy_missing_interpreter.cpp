#include <gtest/gtest.h>
#include <torch/csrc/deploy/deploy.h>
#include <torch/torch.h>

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  return rc;
}

TEST(TorchDeployMissingInterpreter, Throws) {
  EXPECT_THROW(torch::deploy::InterpreterManager(1), c10::Error);
}
