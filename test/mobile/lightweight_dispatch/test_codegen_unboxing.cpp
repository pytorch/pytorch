#include <gtest/gtest.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
// Cover codegen'd unboxing logic for these types:
//'Device',
//'Device?',
//'Dimname',
//'Dimname[1]',
//'Dimname[]',
//'Dimname[]?',
//'Generator?',
//'Layout?',
//'MemoryFormat',
//'MemoryFormat?',
//'Scalar',
//'Scalar?',
//'ScalarType',
//'ScalarType?',
//'Scalar[]',
//'Storage',
//'Stream',
//'Tensor',
//'Tensor(a!)',
//'Tensor(a!)[]',
//'Tensor(a)',
//'Tensor(b!)',
//'Tensor(c!)',
//'Tensor(d!)',
//'Tensor?',
//'Tensor?[]',
//'Tensor[]',
//'bool',
//'bool?',
//'bool[2]',
//'bool[3]',
//'bool[4]',
//'float',
//'float?',
//'float[]?',
//'int',
//'int?',
//'int[1]',
//'int[1]?',
//'int[2]',
//'int[2]?',
//'int[3]',
//'int[4]',
//'int[5]',
//'int[6]',
//'int[]',
//'int[]?',
//'str',
//'str?'
namespace torch {
namespace jit {
namespace mobile {
// covers int[], ScalarType?, Layout?, Device?, bool?
TEST(LiteInterpreterTest, Ones) {
  // Load check in model: ones.ptl
  auto testModelFile = "ones.ptl";

  //  class Model(torch.nn.Module):
  //    def forward(self, x: int):
  //        a = torch.ones([3, x], dtype=torch.int64, layout=torch.strided, device="cpu")
  //        return a
  Module bc = _load_for_mobile(testModelFile);
  std::vector<c10::IValue> input{c10::IValue(4)};
  const auto result = bc.forward(input);
  ASSERT_EQ(result.toTensor().size(0), 3);
  ASSERT_EQ(result.toTensor().size(1), 4);
}

TEST(LiteInterpreterTest, Index) {
  // Load check in model: index.ptl
  auto testModelFile = "index.ptl";

  //    class Model(torch.nn.Module):
  //      def forward(self, index):
  //        a = torch.zeros(2, 2)
  //        a[0][1] = 1
  //        a[1][0] = 2
  //        a[1][1] = 3
  //        return a[index]
  Module bc = _load_for_mobile(testModelFile);
  int64_t ind_1 = 0;

  const auto result_1 = bc.forward({at::tensor(ind_1)});

  at::Tensor expected = at::empty({1, 2}, c10::TensorOptions(c10::ScalarType::Float));
  expected[0][1] = 1;

  AT_ASSERT(result_1.toTensor().equal(expected));
}
} // namespace mobile
} // namespace jit
} // namespace torch