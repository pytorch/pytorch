#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/passes/eliminate_list_construct_unpack.h>

namespace torch {
namespace jit {

namespace {

void TestEliminateListConstructUnpack(
    const std::string& src,
    const std::vector<c10::IValue>& args,
    const std::string& expected_string) {
  testGraphPass(src, args, expected_string, EliminateListConstructUnpack);
}

} // namespace

TEST(EliminateListConstructUnpack, OptimizationApplied_SingleArg) {
  const std::string src = R"IR(
        graph(%0 : Tensor):
            %1 : Tensor[] = prim::ListConstruct(%0)
            %2 : Tensor = prim::ListUnpack(%1)
            return (%2)
    )IR";
  const std::vector<IValue> args{at::randn({1})};
  const std::string expected_string = "return (%0)";
  TestEliminateListConstructUnpack(src, args, expected_string);
}

TEST(EliminateListConstructUnpack, OptimizationApplied_MultiArg) {
  const std::string src = R"IR(
        graph(%0 : Tensor, %1 : Tensor):
            %2 : Tensor[] = prim::ListConstruct(%0, %1)
            %3 : Tensor, %4 : Tensor = prim::ListUnpack(%2)
            return (%3, %4)
    )IR";
  const std::vector<IValue> args{at::randn({1}), at::randn({1})};
  const std::string expected_string = "return (%0, %1)";
  TestEliminateListConstructUnpack(src, args, expected_string);
}

TEST(EliminateListConstructUnpack, OptimizationNotApplied_ListMutated) {
  const std::string src = R"IR(
        graph(%0 : Tensor, %1 : Tensor):
            %2 : int = prim::Constant[value=0]()
            %3 : Tensor[] = prim::ListConstruct(%0, %1)
            %4 : Tensor[] = aten::_set_item(%3, %2, %1)
            %5 : Tensor, %6 : Tensor = prim::ListUnpack(%3)
            return (%5, %6)
    )IR";
  const std::vector<IValue> args{at::randn({1}), at::randn({1})};
  const std::string expected_string = "return (%5, %6)";
  TestEliminateListConstructUnpack(src, args, expected_string);
}

} // namespace jit
} // namespace torch
