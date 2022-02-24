#include <gtest/gtest.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>

namespace torch {
namespace jit {
namespace mobile {

TEST(LiteInterpreterTest, Ones) {
  // Load check in model: sequence.ptl
  std::string filePath(__FILE__);
  auto testModelFile = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  testModelFile.append("ones.ptl");

  //  class Model(torch.nn.Module):
  //    def forward(self):
  //        a = torch.ones(3, 4, dtype=torch.int64, layout=torch.strided, device="cpu", requires_grad=False)
  //        return a
  Module bc = _load_for_mobile(testModelFile);
  auto forward_method = bc.find_method("forward");
  std::vector<c10::IValue> input{c10::IValue(at::tensor(1))};
  const auto result = bc.forward(input);
  EXPECT_EQ(result.toList().get(0).toTensor().sizes()[0], 3);
  EXPECT_EQ(result.toList().get(0).toTensor().sizes()[1], 4);
}

} // namespace mobile
} // namespace jit
} // namespace torch