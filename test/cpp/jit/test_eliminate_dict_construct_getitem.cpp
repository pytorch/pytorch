#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/passes/eliminate_dict_construct_getitem.h>

namespace torch {
namespace jit {

namespace {

void TestDictConstructGetItem(
    const std::string& src,
    const std::vector<IValue>& args,
    const std::string& expected_string) {
  testGraphPass(src, args, expected_string, EliminateDictConstructGetItem);
}

} // namespace

TEST(EliminateDictConstructGetItemTest, SingleKeyGet_SingleKeyDict) {
  const auto src = R"IR(
    graph(%0: Tensor):
        %1 : int = prim::Constant[value=1]()
        %2 : Dict(int, Tensor) = prim::DictConstruct(%1, %0)
        %3 : Tensor = aten::__getitem__(%2, %1)
        return (%3)
    )IR";
  const std::vector<c10::IValue> args{at::randn({2, 2})};
  const std::string expected_string = "return (%0)";
  TestDictConstructGetItem(src, args, expected_string);
}

TEST(EliminateDictConstructGetItemTest, SingleKeyGet_MultiKeyDict) {
  const auto src = R"IR(
    graph(%0: Tensor, %1: Tensor):
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=2]()
        %4 : Dict(int, Tensor) = prim::DictConstruct(%2, %0, %3, %1)
        %5 : Tensor = aten::__getitem__(%4, %3)
        return (%5)
    )IR";
  const std::vector<c10::IValue> args{at::randn({2, 2}), at::randn({2, 2})};
  const std::string& expected_string = "return (%1)";
  TestDictConstructGetItem(src, args, expected_string);
}

TEST(EliminateDictConstructGetItemTest, SingleKeyGet_InputArgKey) {
  const auto src = R"IR(
    graph(%0: Tensor, %1: int):
        %2 : Dict(int, Tensor) = prim::DictConstruct(%1, %0)
        %3 : Tensor = aten::__getitem__(%2, %1)
        return (%3)
    )IR";
  const std::vector<c10::IValue> args{at::randn({2, 2}), 1};
  const std::string expected_string = "return (%0)";
  TestDictConstructGetItem(src, args, expected_string);
}

TEST(EliminateDictConstructGetItemTest, SingleKeyGet_ManyPureOps) {
  // This src has ops other than __getitem__ that do not mutate
  // the dict.
  const auto src = R"IR(
    graph(%0: Tensor):
        %1 : int = prim::Constant[value=1]()
        %2 : Dict(int, Tensor) = prim::DictConstruct(%1, %0)
        %3 : Tensor = aten::__getitem__(%2, %1)
        %4 : int = aten::len(%2)
        %5 : Dict(int, Tensor) = aten::copy(%2)
        %6 : int[] = aten::keys(%2)
        %7 : Tensor[] = aten::values(%2)
        %8 : bool = aten::__contains__(%2, %1)
        return (%3)
    )IR";
  const std::vector<c10::IValue> args{at::randn({2, 2})};
  const std::string expected_string = "return (%0)";
  TestDictConstructGetItem(src, args, expected_string);
}

TEST(EliminateDictConstructGetItemTest, MultiKeyGet_MultiKeyDict) {
  const auto src = R"IR(
    graph(%0: Tensor, %1: Tensor):
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=2]()
        %4 : Dict(int, Tensor) = prim::DictConstruct(%2, %0, %3, %1)
        %5 : Tensor = aten::__getitem__(%4, %2)
        %6 : Tensor = aten::__getitem__(%4, %3)
        %7 : Tensor = aten::matmul(%5, %6)
        return (%7)
    )IR";
  const std::vector<c10::IValue> args{at::randn({2, 2}), at::randn({2, 2})};
  const std::string expected_string = "aten::matmul(%0, %1)";
  TestDictConstructGetItem(src, args, expected_string);
}

TEST(EliminateDictConstructGetItemTest, NoOptimization_MissingKey) {
  const auto src = R"IR(
    graph(%0: Tensor):
        %1 : int = prim::Constant[value=1]()
        %2 : int = prim::Constant[value=2]()
        %3 : Dict(int, Tensor) = prim::DictConstruct(%1, %0)
        %4 : Tensor = aten::__getitem__(%3, %2)
        return (%4)
    )IR";
  // Can't use TestDictConstructGetItem; the script will throw an error
  // since the key is missing.
  auto graph = makeGraph(src);

  EliminateDictConstructGetItem(graph);
  graph->lint();

  // The return value should not change.
  testing::FileCheck().check("return (%4)")->run(*graph);
}

TEST(EliminateDictConstructGetItemTest, NoOptimization_InputArgKey) {
  const auto src = R"IR(
    graph(%0: Tensor, %1: int):
        %2 : int = prim::Constant[value=1]()
        %3 : Dict(int, Tensor) = prim::DictConstruct(%2, %0)
        %4 : Tensor = aten::__getitem__(%3, %1)
        return (%4)
    )IR";
  const std::vector<c10::IValue> args{at::randn({2, 2}), 1};
  // The return value should not change.
  const std::string expected_string = "return (%4)";
  TestDictConstructGetItem(src, args, expected_string);
}

TEST(EliminateDictConstructGetItem, NoOptimization_DictModified) {
  // Making the optimization isn't safe here because the dict is mutated
  // between constructing the dict and accessing it.
  const auto src = R"IR(
    graph(%0: Tensor, %1: Tensor):
        %2 : int = prim::Constant[value=1]()
        %3 : Dict(int, Tensor) = prim::DictConstruct(%2, %0)
        aten::_set_item(%3, %2, %1)
        %4 : Tensor = aten::__getitem__(%3, %2)
        return (%4)
    )IR";
  const std::vector<c10::IValue> args{at::randn({2, 2}), at::randn({2, 2})};
  // The return value should not change.
  const std::string expected_string = "return (%4)";
  TestDictConstructGetItem(src, args, expected_string);
}

} // namespace jit
} // namespace torch
