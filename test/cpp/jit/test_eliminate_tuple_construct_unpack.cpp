#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/passes/eliminate_tuple_construct_unpack.h>

namespace torch {
namespace jit {

namespace {

void TestEliminateTupleConstructUnpack(
    const std::string& src,
    const std::vector<c10::IValue>& args,
    const std::string& expected_string) {
  testGraphPass(src, args, expected_string, EliminateTupleConstructUnpack);
}

void TestEliminateTupleUnpackConstruct(
    const std::string& src,
    const std::vector<c10::IValue>& args,
    const std::string& expected_string) {
  testGraphPass(src, args, expected_string, EliminateTupleUnpackConstruct);
}

} // namespace

TEST(EliminateTupleConstructUnpack, SingleArg) {
  const std::string src = R"IR(
        graph(%0 : Tensor):
            %1 : Tensor[] = prim::TupleConstruct(%0)
            %2 : Tensor = prim::TupleUnpack(%1)
            return (%2)
    )IR";
  const std::vector<IValue> args{at::randn({2})};
  const std::string expected_string = "return (%0)";
  TestEliminateTupleConstructUnpack(src, args, expected_string);
}

TEST(EliminateTupleConstructUnpack, MultiArg) {
  const std::string src = R"IR(
        graph(%0 : Tensor, %1 : Tensor):
            %2 : Tensor[] = prim::TupleConstruct(%0, %1)
            %3 : Tensor, %4 : Tensor = prim::TupleUnpack(%2)
            return (%3, %4)
    )IR";
  const std::vector<IValue> args{at::randn({2}), at::randn({2})};
  const std::string expected_string = "return (%0, %1)";
  TestEliminateTupleConstructUnpack(src, args, expected_string);
}

TEST(EliminateTupleUnpackConstruct, OptimizationApplied) {
  const std::string src = R"IR(
        graph(%0 : Tensor, %1 : Tensor):
            %2 : (Tensor, Tensor) = prim::TupleConstruct(%0, %1)

            %3 : Tensor, %4 : Tensor = prim::TupleUnpack(%2)
            %5 : (Tensor, Tensor) = prim::TupleConstruct(%3, %4)
            return (%5)
        )IR";

  const std::vector<IValue> args{at::randn({1}), at::randn({1})};
  const std::string expected_string = "return (%2)";
  TestEliminateTupleUnpackConstruct(src, args, expected_string);
}

TEST(EliminateTupleUnpackConstruct, NoOptimization_Reorder) {
  const std::string src = R"IR(
        graph(%0 : Tensor, %1 : Tensor):
            %2 : (Tensor, Tensor) = prim::TupleConstruct(%0, %1)

            %3 : Tensor, %4 : Tensor = prim::TupleUnpack(%2)
            %5 : (Tensor, Tensor) = prim::TupleConstruct(%4, %3)
            return (%5)
    )IR";

  const std::vector<IValue> args{at::randn({1}), at::randn({1})};
  const std::string expected_string = "return (%5)";
  TestEliminateTupleUnpackConstruct(src, args, expected_string);
}

TEST(EliminateTupleUnpackConstruct, NoOptimization_SubsetConstructed) {
  const std::string src = R"IR(
        graph(%0 : Tensor, %1 : Tensor, %2 : Tensor):
            %3 : (Tensor, Tensor, Tensor) = prim::TupleConstruct(%0, %1, %2)

            %4 : Tensor, %5 : Tensor, %6 : Tensor = prim::TupleUnpack(%3)
            %7 : (Tensor, Tensor) = prim::TupleConstruct(%4, %5)
            return (%7)
    )IR";

  const std::vector<IValue> args{
      at::randn({1}), at::randn({1}), at::randn({1})};
  const std::string expected_string = "return (%7)";
  TestEliminateTupleUnpackConstruct(src, args, expected_string);
}

TEST(EliminateTupleUnpackConstruct, NoOptimization_DifferentTuples) {
  const std::string src = R"IR(
        graph(%0 : Tensor, %1 : Tensor, %2 : Tensor, %3 : Tensor):
            %4 : (Tensor, Tensor) = prim::TupleConstruct(%0, %1)
            %5 : (Tensor, Tensor) = prim::TupleConstruct(%3, %4)

            %6 : Tensor, %7 : Tensor = prim::TupleUnpack(%4)
            %8 : Tensor, %9 : Tensor = prim::TupleUnpack(%5)

            %10 : (Tensor, Tensor) = prim::TupleConstruct(%6, %8)
            return (%10)
    )IR";

  const std::vector<IValue> args{
      at::randn({1}), at::randn({1}), at::randn({1}), at::randn({1})};
  const std::string expected_string = "return (%10)";
  TestEliminateTupleUnpackConstruct(src, args, expected_string);
}

} // namespace jit
} // namespace torch
