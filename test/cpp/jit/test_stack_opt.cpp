#include <ATen/Functions.h>
#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/variadic_ops.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {

TEST(StackOptTest, UseVariadicStack) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(56, 56, 56),
              %1: Float(56, 56, 56),
              %2: Float(56, 56, 56),
              %3: Float(56, 56, 56),
              %4: Float(56, 56, 56),
              %5: Float(56, 56, 56)):
          %10 : int = prim::Constant[value=0]()
          %input : Tensor[] = prim::ListConstruct(%0, %1, %2, %3, %4, %5)
          %stack : Float(5, 56, 56, 56) = aten::stack(%input, %10)
          return (%stack)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(UseVariadicStack(graph));
  graph->lint();
  auto opt_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // After replacing `aten::stack` with `prim::VarStack` we should have the
  // following graph:
  //
  //  graph(%0 : ...,
  //        %1 : ...):
  //    %zero : int = prim:Constant[value=0]()
  //    %varstack : Tensor = prim::VarStack(%0, %1, %2, %3, %4, %5, %zero)
  //    return (%varstack)
  testing::FileCheck()
      .check_count("= prim::VarStack(", 1, /*exactly*/ true)
      ->check_count("= aten::stack(", 0, /*exactly*/ true)
      ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
      ->run(*graph);
}

TEST(StackOptTest, UseVariadicStackReplaceMultiple) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(56, 56, 56),
              %1: Float(56, 56, 56),
              %2: Float(56, 56, 56),
              %3: Float(56, 56, 56)):
          %10 : int = prim::Constant[value=0]()
          %input1 : Tensor[] = prim::ListConstruct(%0, %1)
          %stack1 : Float(4, 56, 56, 56) = aten::stack(%input1, %10)
          %input2 : Tensor[] = prim::ListConstruct(%2, %3)
          %stack2 : Float(4, 56, 56, 56) = aten::stack(%input2, %10)
          return (%stack1, %stack2)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(UseVariadicStack(graph));
  graph->lint();
  auto opt_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // After full stack optimization we should have the following graph:
  //
  //  graph(%0 : ...,
  //        %1 : ...,
  //        %2 : ...,
  //        %3 : ....):
  //    %zero : int = prim:Constant[value=0]()
  //    %varcat1 : Tensor = prim::VarStack(%0, %1, %zero)
  //    %varcat2 : Tensor = prim::VarStack(%2, %3, %zero)
  //    return (%varcat1, %varcat2)
  testing::FileCheck()
      .check_count("= prim::VarStack(", 2, /*exactly*/ true)
      ->check_count("= aten::stack(", 0, /*exactly*/ true)
      ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
      ->run(*graph);
}

TEST(StackOptTest, UseVariadicStackWithMultipleListUses) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(56, 56, 56),
              %1: Float(56, 56, 56)):
          %2 : int = prim::Constant[value=0]()
          %input : Tensor[] = prim::ListConstruct(%0, %1)
          %stack : Float(2, 56, 56, 56) = aten::stack(%input, %2)
          return (%stack, %input)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({56, 56, 56}, at::kCPU), at::rand({56, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(UseVariadicStack(graph));
  graph->lint();
  auto opt_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // After replacing `aten::stack` with `prim::VarStack` we should have the
  // following graph:
  //
  //  graph(%0 : ...,
  //        %1 : ...):
  //    %zero : int = prim:Constant[value=0]()
  //    %input : Tensor[] = prim::ListConstruct(%0, %1)
  //    %varcat : Tensor = prim::VarStack(%0, %1, %zero)
  //    return (%varcat, %input)
  testing::FileCheck()
      .check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
      ->check_count("= prim::VarStack(", 1, /*exactly*/ true)
      ->check_count("= aten::stack(", 0, /*exactly*/ true)
      ->run(*graph);
}

TEST(StackOptTest, UseVariadicStackWithListMutationAfterCat) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(56, 56, 56),
              %1: Float(56, 56, 56),
              %2: Float(56, 56, 56)):
          %10 : int = prim::Constant[value=0]()
          %input : Tensor[] = prim::ListConstruct(%0, %1)
          %stack : Float(3, 56, 56, 56) = aten::stack(%input, %10)
          %11 : Tensor = aten::append(%input, %2)
          return (%stack, %input)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(UseVariadicStack(graph));
  graph->lint();
  auto opt_outputs = runGraph(graph, inputs);
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // The input list to `aten::stack` is mutated only after `aten::stack` op. So,
  // it should have been replaced with `prim::VarStack`. The transformed graph
  // should look like the following:
  //
  //  graph(%0 : ...,
  //        %1 : ...,
  //        %2 : ...):
  //    %3 : int = prim:Constant[value=0]()
  //    %4 : Tensor[] = prim::ListConstruct(%0, %1)
  //    %7 : Tensor = prim::VarStack(%0, %1, %3)
  //    %6 : Tensor = aten::append(%4, %2)
  //    return (%7, %4)
  testing::FileCheck()
      .check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
      ->check_count("= prim::VarStack(", 1, /*exactly*/ true)
      ->check_count("= aten::stack(", 0, /*exactly*/ true)
      ->run(*graph);
}

TEST(StackOptTest, UseVariadicStackWithListMutationBeforeCat) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(56, 56, 56),
              %1: Float(56, 56, 56),
              %2: Float(56, 56, 56)):
          %10 : int = prim::Constant[value=0]()
          %input : Tensor[] = prim::ListConstruct(%0, %1)
          %11 : Tensor = aten::append(%input, %2)
          %stack : Float(3, 56, 56, 56) = aten::stack(%input, %10)
          return (%stack)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  {
    ASSERT_FALSE(UseVariadicStack(graph));
    graph->lint();
    auto opt_outputs = runGraph(graph, inputs);
    ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

    // No transformation should have happened since the `prim::ListConstruct` is
    // mutated before `aten::stack`.
    testing::FileCheck()
        .check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
        ->check_count("= aten::stack(", 1, /*exactly*/ true)
        ->check_count("= prim::VarStack(", 0, /*exactly*/ true)
        ->run(*graph);
  }

  {
    ASSERT_TRUE(RemoveListMutationAndUseVariadicStack(graph));
    graph->lint();
    auto opt_outputs = runGraph(graph, inputs);
    ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

    // The mutation of the list must be removed and the `aten::stack` op must
    // be replaced with the `prim::VarStack` op in the graph. The transformed
    // graph should look like the following:
    //
    //  graph(%0 : ...,
    //        %1 : ...,
    //        %2 : ...):
    //    %3 : int = prim:Constant[value=0]()
    //    %7 : Tensor = prim::VarStack(%0, %1, %2, %3)
    //    return (%7)
    testing::FileCheck()
        .check_count("= prim::VarStack(", 1, /*exactly*/ true)
        ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
        ->check_count("= aten::stack(", 0, /*exactly*/ true)
        ->run(*graph);
  }
}

TEST(StackOptTest, UseVariadicStackWithMultipleListMutations) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(56, 56, 56),
              %1: Float(56, 56, 56),
              %2: Float(56, 56, 56),
              %3: Float(56, 56, 56),
              %4: Float(56, 56, 56)):
          %10 : int = prim::Constant[value=0]()
          %input : Tensor[] = prim::ListConstruct(%0, %1)
          %stack.1 : Float(5, 56, 56, 56) = aten::stack(%input, %10)
          %11 : Tensor = aten::append(%input, %2)
          %stack.2 : Float(5, 56, 56, 56) = aten::stack(%input, %10)
          %12 : Tensor = aten::append(%input, %3)
          %stack.3 : Float(5, 56, 56, 56) = aten::stack(%input, %10)
          %13 : Tensor = aten::append(%input, %4)
          %stack.4 : Float(5, 56, 56, 56) = aten::stack(%input, %10)
          return (%stack.1, %stack.2, %stack.3, %stack.4)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU),
      at::rand({56, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(RemoveListMutationAndUseVariadicStack(graph));
  graph->lint();
  auto opt_outputs = runGraph(graph, inputs);
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // All the mutations of the list must be removed and the `aten::stack` ops
  // must be replaced with `prim::VarStack` ops in the graph. The transformed
  // graph should look like the following:
  //
  //  graph(%0 : ...,
  //        %1 : ...,
  //        %2 : ...,
  //        %3 : ...,
  //        %4 : ...):
  //    %10 : int = prim:Constant[value=0]()
  //    %5 : Tensor = prim::VarStack(%0, %1, %10)
  //    %6 : Tensor = prim::VarStack(%0, %1, %2, %10)
  //    %7 : Tensor = prim::VarStack(%0, %1, %2, %3, %10)
  //    %8 : Tensor = prim::VarStack(%0, %1, %2, %3, %4, %10)
  //    return (%5, %6, %7, %8)
  testing::FileCheck()
      .check_count("= prim::VarStack(", 4, /*exactly*/ true)
      ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
      ->check_count("= aten::stack(", 0, /*exactly*/ true)
      ->run(*graph);
}

} // namespace jit
} // namespace torch
