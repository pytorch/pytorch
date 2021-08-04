#include <gtest/gtest.h>

#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/eliminate_dict_construct_getitem.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {

namespace {

void compareTensors(const IValue& expected, const IValue& actual) {
  EXPECT_TRUE(expected.isTensor());
  EXPECT_TRUE(actual.isTensor());
  EXPECT_TRUE(expected.toTensor().equal(actual.toTensor()));
}

c10::IValue runGraph(
    std::shared_ptr<Graph> graph,
    const std::vector<c10::IValue>& args) {
  auto graph_exec = GraphExecutor(graph, "");

  Stack stack(args);
  graph_exec.run(stack);

  if (stack.size() == 1) {
    return stack[0];
  }
  return c10::ivalue::Tuple::create(stack);
}

std::shared_ptr<Graph> makeGraph(const std::string& ir_src) {
  auto graph = std::make_shared<Graph>();
  parseIR(ir_src, graph.get());
  return graph;
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

  auto graph = makeGraph(src);

  std::vector<c10::IValue> args{at::randn({2, 2})};

  auto expected = runGraph(graph, args);

  EliminateDictConstructGetItem(graph);
  graph->lint();

  auto actual = runGraph(graph, args);
  compareTensors(expected, actual);

  testing::FileCheck().check("return (%0)")->run(*graph);
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

  auto graph = makeGraph(src);

  std::vector<c10::IValue> args{at::randn({2, 2}), at::randn({2, 2})};

  auto expected = runGraph(graph, args);

  EliminateDictConstructGetItem(graph);
  graph->lint();

  auto actual = runGraph(graph, args);
  compareTensors(expected, actual);

  testing::FileCheck().check("return (%1)")->run(*graph);
}

TEST(EliminateDictConstructGetItemTest, SingleKeyGet_InputArgKey) {
  const auto src = R"IR(
    graph(%0: Tensor, %1: int):
        %2 : Dict(int, Tensor) = prim::DictConstruct(%1, %0)
        %3 : Tensor = aten::__getitem__(%2, %1)
        return (%3)
    )IR";

  auto graph = makeGraph(src);

  std::vector<c10::IValue> args{at::randn({2, 2}), 1};

  auto expected = runGraph(graph, args);

  EliminateDictConstructGetItem(graph);
  graph->lint();

  auto actual = runGraph(graph, args);
  compareTensors(expected, actual);

  testing::FileCheck().check("return (%0)")->run(*graph);
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

  auto graph = makeGraph(src);

  std::vector<c10::IValue> args{at::randn({2, 2})};

  auto expected = runGraph(graph, args);

  EliminateDictConstructGetItem(graph);
  graph->lint();

  auto actual = runGraph(graph, args);
  compareTensors(expected, actual);

  testing::FileCheck().check("return (%0)")->run(*graph);
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

  auto graph = makeGraph(src);

  std::vector<c10::IValue> args{at::randn({2, 2}), at::randn({2, 2})};

  auto expected = runGraph(graph, args);

  EliminateDictConstructGetItem(graph);
  graph->lint();

  auto actual = runGraph(graph, args);
  compareTensors(expected, actual);

  testing::FileCheck().check("aten::matmul(%0, %1)")->run(*graph);
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

  auto graph = makeGraph(src);

  EliminateDictConstructGetItem(graph);
  graph->lint();

  // The return value should not change.
  testing::FileCheck().check("return (%4)")->run(*graph);
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

  auto graph = makeGraph(src);

  EliminateDictConstructGetItem(graph);
  graph->lint();

  // The return value should not change.
  testing::FileCheck().check("return (%4)")->run(*graph);
}

} // namespace jit
} // namespace torch
