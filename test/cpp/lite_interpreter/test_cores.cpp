#include <gtest/gtest.h>

#include <c10/core/TensorOptions.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/custom_class.h>
#include <torch/torch.h>

#include <unordered_set>

#define ASSERT_THROWS_WITH(statement, substring)                         \
  try {                                                                  \
    (void)statement;                                                     \
    ASSERT_TRUE(false);                                                  \
  } catch (const std::exception& e) {                                    \
    ASSERT_NE(std::string(e.what()).find(substring), std::string::npos); \
  }

// Tests go in torch::jit
namespace torch {
namespace jit {

TEST(CoresTest, Load) {
  std::string filePath(__FILE__);
  auto testModelFile = filePath.substr(0, filePath.find_last_of("/\\") + 1);

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

  testModelFile.append("sequence.ptl");
  mobile::Module bc = _load_for_mobile(testModelFile);
  auto forward_method = bc.find_method("forward");
  std::vector<c10::IValue> input{c10::IValue(at::ones(1))};
  auto result = bc.forward(input);
  auto expected_result = c10::IValue(4);
  ASSERT_EQ(result.toInt(), expected_result.toInt());

}

} // namespace jit
} // namespace torch
