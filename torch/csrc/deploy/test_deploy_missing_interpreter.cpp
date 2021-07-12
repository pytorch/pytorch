#include <gtest/gtest.h>
#include <torch/csrc/deploy/deploy.h>
#include <torch/torch.h>

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  return rc;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TorchDeployMissingInterpreter, Throws) {
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(torch::deploy::InterpreterManager(1), c10::Error);
}
