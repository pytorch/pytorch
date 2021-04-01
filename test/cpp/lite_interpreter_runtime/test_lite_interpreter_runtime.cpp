#include <gtest/gtest.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace mobile {

TEST(RunTimeTest, LoadAndForward) {
  // Load check in model: sequence.ptl
  std::string filePath(__FILE__);
  auto testModelFile = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  testModelFile.append("sequence.ptl");

  //  sequence.ptl source code:
  //  class A(torch.nn.Module):
  //    def __init__(self):
  //      super(A, self).__init__()
  //
  //    def forward(self, x):
  //      return x + 1
  //
  //  class B(torch.nn.Module):
  //    def __init__(self):
  //      super(B, self).__init__()
  //
  //    def forward(self, x):
  //      return x + 2
  //
  //  class C(torch.nn.Module):
  //    def __init__(self):
  //      super(C, self).__init__()
  //      self.A0 = A()
  //      self.B0 = B()
  //
  //    def forward(self, x):
  //      return self.A0.forward(self.B0.forward(x))

  Module bc = _load_for_mobile(testModelFile);
  auto forward_method = bc.find_method("forward");
  std::vector<c10::IValue> input{c10::IValue(at::tensor(1))};
  const auto result = bc.forward(input);
  const auto expected_result = c10::IValue(at::tensor(4));
  ASSERT_EQ(result, expected_result);
}

} // namespace mobile
} // namespace jit
} // namespace torch
