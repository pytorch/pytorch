#include <gtest/gtest.h>

#include <ATen/Functions.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/concat_opt.h>
#include <torch/csrc/jit/passes/variadic_ops.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {

TEST(ConcatOptTest, SimpleCommonInputsEliminationPrefix) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %5 : int = prim::Constant[value=0]()
          %concat.2 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = prim::VarConcat(%0, %1, %5)
          %concat.3 : Float(128, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = prim::VarConcat(%0, %1, %2, %5)
          %res : Tensor[] = prim::ListConstruct(%concat.2, %concat.3)
          return (%res)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(EliminateConcatCommonInputs(graph));
  graph->lint();
  auto opt_outputs = runGraph(graph, inputs);
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // Graph after EliminateConcatCommonInputs:
  //  graph(%0 : ...,
  //        %1 : ...,
  //        %2 : ...):
  //    %3 : int = prim::Constant[value=0]()
  //    %4 : Tensor = prim::VarConcat(%0, %1, %3)
  //    %7 : Tensor = prim::VarConcat(%4, %2, %3) // UPDATED
  //    %8 : Tensor[] = prim::ListConstruct(%4, %7)
  //    return (%8)

  testing::FileCheck()
      .check_count("= prim::VarConcat(%0, %1, %3)", 1, /*exactly*/ true)
      ->check_count("= prim::VarConcat(%4, %2, %3)", 1, /*exactly*/ true)
      ->check_count("= prim::ListConstruct(%4, %7)", 1, /*exactly*/ true)
      ->check_count("= aten::cat(", 0, /*exactly*/ true)
      ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
      ->run(*graph);
}

TEST(ConcatOptTest, SimpleCommonInputsEliminationSuffix) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %5 : int = prim::Constant[value=0]()
          %concat.2 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = prim::VarConcat(%1, %2, %5)
          %concat.3 : Float(128, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = prim::VarConcat(%0, %1, %2, %5)
          %res : Tensor[] = prim::ListConstruct(%concat.2, %concat.3)
          return (%res)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(EliminateConcatCommonInputs(graph));
  graph->lint();
  auto opt_outputs = runGraph(graph, inputs);
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // Graph after EliminateConcatCommonInputs:
  //  graph(%0 : ...,
  //        %1 : ...,
  //        %2 : ...):
  //    %3 : int = prim::Constant[value=0]()
  //    %4 : Tensor = prim::VarConcat(%1, %2, %3)
  //    %7 : Tensor = prim::VarConcat(%0, %4, %3) // UPDATED
  //    %8 : Tensor[] = prim::ListConstruct(%4, %7)
  //    return (%8)

  testing::FileCheck()
      .check_count("= prim::VarConcat(%1, %2, %3)", 1, /*exactly*/ true)
      ->check_count("= prim::VarConcat(%0, %4, %3)", 1, /*exactly*/ true)
      ->check_count("= prim::ListConstruct(%4, %7)", 1, /*exactly*/ true)
      ->check_count("= aten::cat(", 0, /*exactly*/ true)
      ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
      ->run(*graph);
}

TEST(ConcatOptTest, CommonInputsEliminationWithDifferentOrderInputs) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %5 : int = prim::Constant[value=0]()

          #CHECK: prim::VarConcat
          %concat.1 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = prim::VarConcat(%0, %1, %5)

          #CHECK: prim::VarConcat
          %concat.2 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = prim::VarConcat(%1, %0, %2, %5)

          #CHECK: prim::ListConstruct
          %res : Tensor[] = prim::ListConstruct(%concat.1, %concat.2)
          return (%res)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  ASSERT_FALSE(EliminateConcatCommonInputs(graph));
  graph->lint();
  auto opt_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // No optimizations should have happened in this case since the inputs
  // to the `cat` are in different order.
  testing::FileCheck().run(input, *graph);
}

TEST(ConcatOptTest, MoreCommonInputsElimination) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %3: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %4: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %5 : int = prim::Constant[value=0]()
          %concat.1 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = prim::VarConcat(%0, %1, %5)
          %concat.2 : Float(128, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = prim::VarConcat(%0, %1, %2, %5)
          %concat.3 : Float(160, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = prim::VarConcat(%0, %1, %2, %3, %5)
          %concat.4 : Float(192, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = prim::VarConcat(%0, %1, %2, %3, %4, %5)
          %res : Tensor[] = prim::ListConstruct(%concat.1, %concat.2, %concat.3, %concat.4)
          return (%res)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(EliminateConcatCommonInputs(graph));
  graph->lint();
  auto opt_outputs = runGraph(graph, inputs);
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  testing::FileCheck()
      .check_count("= prim::VarConcat(%0, %1, %5)", 1, /*exactly*/ true)
      ->check_count("= prim::VarConcat(%6, %2, %5)", 1, /*exactly*/ true)
      ->check_count("= prim::VarConcat(%11, %3, %5)", 1, /*exactly*/ true)
      ->check_count("= prim::VarConcat(%12, %4, %5)", 1, /*exactly*/ true)
      ->check_count("= aten::cat(", 0, /*exactly*/ true)
      ->run(*graph);
}

TEST(ConcatOptTest, ExpandConcat) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %2 : int = prim::Constant[value=0]()
          %3 : float = prim::Constant[value=0.5]()
          %4 : Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::clamp_max(%0, %3)
          %5 : Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::clamp_max(%1, %3)
          %input : Tensor[] = prim::ListConstruct(%4, %5)
          %concat : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%input, %2)
          return (%concat)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU), at::rand({32, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  ExpandConcatAndEliminateRedundancy(graph);
  graph->lint();
  auto opt_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // After full concat optimization we should have the following graph:
  //
  //  graph(%0 : ...,
  //        %1 : ...):
  //    ...
  //    %4 : Tensor = aten::clamp_max(...)
  //    %5 : Tensor = aten::clamp_max(...)
  //    %13 : int[] = prim::ListConstruct(...)
  //    %14 : Tensor = aten::empty(%13, ...)    // concat buffer
  //    %17 : Tensor = aten::slice(%14, ...)    // slice for %4
  //    %18 : Tensor = aten::copy_(%17, %4)
  //    %20 : Tensor = aten::slice(%14, ...)    // slice for %5
  //    %21 : Tensor = aten::copy_(%20, %5)
  //    return (%14)
  testing::FileCheck()
      .check_count("= aten::cat(", 0, /*exactly*/ true)
      ->check_count("= aten::clamp_max(", 2, /*exactly*/ true)
      ->check_count("= aten::empty(", 1, /*exactly*/ true)
      ->check_count("= aten::slice(", 1, /*exactly*/ true)
      ->check_count("= aten::copy_(", 1, /*exactly*/ true)
      ->check_count("= aten::slice(", 1, /*exactly*/ true)
      ->check_count("= aten::copy_(", 1, /*exactly*/ true)
      ->check_count("= aten::cat(", 0, /*exactly*/ true)
      ->run(*graph);
}

TEST(ConcatOptTest, ConcatWithoutResultShape) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %2 : int = prim::Constant[value=0]()
          %3 : float = prim::Constant[value=0.5]()
          # CHECK: clamp_max
          %4 : Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::clamp_max(%0, %3)
          # CHECK: clamp_max
          %5 : Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::clamp_max(%1, %3)
          # CHECK: prim::ListConstruct
          %6 : Tensor[] = prim::ListConstruct(%4, %5)
          # CHECK: aten::cat
          %7 : Tensor = aten::cat(%6, %2)
          return (%7)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU), at::rand({32, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  ExpandConcatAndEliminateRedundancy(graph);
  graph->lint();
  auto opt_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // No optimizations should have happened in this case since the output
  // shape of `aten::cat` is not known.
  testing::FileCheck().run(input, *graph);
}

TEST(ConcatOptTest, ConcatWithoutInputShape) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %2 : int = prim::Constant[value=0]()
          %3 : float = prim::Constant[value=0.5]()
          # CHECK: clamp_max
          %4 : Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::clamp_max(%0, %3)
          # CHECK: clamp_max
          %5 : Tensor = aten::clamp_max(%1, %3)
          # CHECK: prim::ListConstruct
          %6 : Tensor[] = prim::ListConstruct(%4, %5)
          # CHECK: aten::cat
          %7 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%6, %2)
          return (%7)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU), at::rand({32, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  ExpandConcatAndEliminateRedundancy(graph);
  graph->lint();
  auto opt_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // No optimizations should have happened in this case since the shape of %5,
  // which is an input to `aten::cat`, is not known.
  testing::FileCheck().run(input, *graph);
}

TEST(ConcatOptTest, UseVariadicCat) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %3: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %4: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %5: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %10 : int = prim::Constant[value=0]()
          %input : Tensor[] = prim::ListConstruct(%0, %1, %2, %3, %4, %5)
          %concat : Float(224, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%input, %10)
          return (%concat)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(UseVariadicCat(graph));
  graph->lint();
  auto opt_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // After replacing `aten::cat` with `prim::VarConcat` we should have the
  // following graph:
  //
  //  graph(%0 : ...,
  //        %1 : ...):
  //    %zero : int = prim:Constant[value=0]()
  //    %varcat : Tensor = prim::VarConcat(%0, %1, %2, %3, %4, %5, %zero)
  //    return (%varcat)
  testing::FileCheck()
      .check_count("= prim::VarConcat(", 1, /*exactly*/ true)
      ->check_count("= aten::cat(", 0, /*exactly*/ true)
      ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
      ->run(*graph);
}

TEST(OptimizeConcatTest, UseVariadicCatReplaceMultiple) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %3: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %10 : int = prim::Constant[value=0]()
          %input1 : Tensor[] = prim::ListConstruct(%0, %1)
          %concat1 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%input1, %10)
          %input2 : Tensor[] = prim::ListConstruct(%2, %3)
          %concat2 : Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%input2, %10)
          return (%concat1, %concat2)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(UseVariadicCat(graph));
  graph->lint();
  auto opt_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // After full concat optimization we should have the following graph:
  //
  //  graph(%0 : ...,
  //        %1 : ...,
  //        %2 : ...,
  //        %3 : ....):
  //    %zero : int = prim:Constant[value=0]()
  //    %varcat1 : Tensor = prim::VarConcat(%0, %1, %zero)
  //    %varcat2 : Tensor = prim::VarConcat(%2, %3, %zero)
  //    return (%varcat1, %varcat2)
  testing::FileCheck()
      .check_count("= prim::VarConcat(", 2, /*exactly*/ true)
      ->check_count("= aten::cat(", 0, /*exactly*/ true)
      ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
      ->run(*graph);
}

TEST(ConcatOptTest, UseVariadicCatWithMultipleListUses) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %2 : int = prim::Constant[value=0]()
          %input : Tensor[] = prim::ListConstruct(%0, %1)
          %concat : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%input, %2)
          return (%concat, %input)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU), at::rand({32, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(UseVariadicCat(graph));
  graph->lint();
  auto opt_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // After replacing `aten::cat` with `prim::VarConcat` we should have the
  // following graph:
  //
  //  graph(%0 : ...,
  //        %1 : ...):
  //    %zero : int = prim:Constant[value=0]()
  //    %input : Tensor[] = prim::ListConstruct(%0, %1)
  //    %varcat : Tensor = prim::VarConcat(%0, %1, %zero)
  //    return (%varcat, %input)
  testing::FileCheck()
      .check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
      ->check_count("= prim::VarConcat(", 1, /*exactly*/ true)
      ->check_count("= aten::cat(", 0, /*exactly*/ true)
      ->run(*graph);
}

TEST(ConcatOptTest, UseVariadicCatWithListMutationAfterCat) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %10 : int = prim::Constant[value=0]()
          %input : Tensor[] = prim::ListConstruct(%0, %1)
          %concat : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%input, %10)
          %11 : Tensor = aten::append(%input, %2)
          return (%concat, %input)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(UseVariadicCat(graph));
  graph->lint();
  auto opt_outputs = runGraph(graph, inputs);
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // The input list to `aten::cat` is mutated only after `aten::cat` op. So,
  // it should have been replaced with `prim::VarConcat`. The transformed graph
  // should look like the following:
  //
  //  graph(%0 : ...,
  //        %1 : ...,
  //        %2 : ...):
  //    %3 : int = prim:Constant[value=0]()
  //    %4 : Tensor[] = prim::ListConstruct(%0, %1)
  //    %7 : Tensor = prim::VarConcat(%0, %1, %3)
  //    %6 : Tensor = aten::append(%4, %2)
  //    return (%7, %4)
  testing::FileCheck()
      .check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
      ->check_count("= prim::VarConcat(", 1, /*exactly*/ true)
      ->check_count("= aten::cat(", 0, /*exactly*/ true)
      ->run(*graph);
}

TEST(ConcatOptTest, UseVariadicCatWithListMutationBeforeCat) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %10 : int = prim::Constant[value=0]()
          %input : Tensor[] = prim::ListConstruct(%0, %1)
          %11 : Tensor = aten::append(%input, %2)
          %concat : Float(128, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%input, %10)
          return (%concat)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  {
    ASSERT_FALSE(UseVariadicCat(graph));
    graph->lint();
    auto opt_outputs = runGraph(graph, inputs);
    ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

    // No transformation should have happened since the `prim::ListConstruct` is
    // mutated before `aten::cat`.
    testing::FileCheck()
        .check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
        ->check_count("= aten::cat(", 1, /*exactly*/ true)
        ->check_count("= prim::VarConcat(", 0, /*exactly*/ true)
        ->run(*graph);
  }

  {
    ASSERT_TRUE(RemoveListMutationAndUseVariadicCat(graph));
    graph->lint();
    auto opt_outputs = runGraph(graph, inputs);
    ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

    // The mutation of the list must be removed and the `aten::cat` op must
    // be replaced with the `prim::VarConcat` op in the graph. The transformed
    // graph should look like the following:
    //
    //  graph(%0 : ...,
    //        %1 : ...,
    //        %2 : ...):
    //    %3 : int = prim:Constant[value=0]()
    //    %7 : Tensor = prim::VarConcat(%0, %1, %2, %3)
    //    return (%7)
    testing::FileCheck()
        .check_count("= prim::VarConcat(", 1, /*exactly*/ true)
        ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
        ->check_count("= aten::cat(", 0, /*exactly*/ true)
        ->run(*graph);
  }
}

TEST(ConcatOptTest, UseVariadicCatWithMultipleListMutations) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %3: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %4: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %10 : int = prim::Constant[value=0]()
          %input : Tensor[] = prim::ListConstruct(%0, %1)
          %concat.1 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%input, %10)
          %11 : Tensor = aten::append(%input, %2)
          %concat.2 : Float(128, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%input, %10)
          %12 : Tensor = aten::append(%input, %3)
          %concat.3 : Float(160, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%input, %10)
          %13 : Tensor = aten::append(%input, %4)
          %concat.4 : Float(192, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%input, %10)
          return (%concat.1, %concat.2, %concat.3, %concat.4)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(RemoveListMutationAndUseVariadicCat(graph));
  graph->lint();
  auto opt_outputs = runGraph(graph, inputs);
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // All the mutations of the list must be removed and the `aten::cat` ops must
  // be replaced with `prim::VarConcat` ops in the graph. The transformed graph
  // should look like the following:
  //
  //  graph(%0 : ...,
  //        %1 : ...,
  //        %2 : ...,
  //        %3 : ...,
  //        %4 : ...):
  //    %10 : int = prim:Constant[value=0]()
  //    %5 : Tensor = prim::VarConcat(%0, %1, %10)
  //    %6 : Tensor = prim::VarConcat(%0, %1, %2, %10)
  //    %7 : Tensor = prim::VarConcat(%0, %1, %2, %3, %10)
  //    %8 : Tensor = prim::VarConcat(%0, %1, %2, %3, %4, %10)
  //    return (%5, %6, %7, %8)
  testing::FileCheck()
      .check_count("= prim::VarConcat(", 4, /*exactly*/ true)
      ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
      ->check_count("= aten::cat(", 0, /*exactly*/ true)
      ->run(*graph);
}

TEST(
    ConcatOptTest,
    RemoveListMutationUseVariadicCatAndCommonInputsElimination) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %5 : int = prim::Constant[value=0]()

          %features.2 : Tensor[] = prim::ListConstruct(%0, %1)
          %6 : Tensor [] = aten::append(%features.2, %2)
          %concat.2 : Float(128, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%features.2, %5)

          %7 : Tensor [] = aten::append(%features.2, %0)
          %concat.3 : Float(160, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%features.2, %5)

          %res : Tensor[] = prim::ListConstruct(%concat.2, %concat.3)
          return (%res)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(RemoveListMutationAndUseVariadicCat(graph));
  ASSERT_TRUE(EliminateConcatCommonInputs(graph));
  graph->lint();
  auto opt_outputs = runGraph(graph, inputs);
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // After performing:
  //     * Remove list mutation
  //     * Use variadic cat
  //     * Eliminate common inputs
  // we should have the following graph:
  //
  //  graph(%0 : ...,
  //        %1 : ...,
  //        %2 : ...):
  //    %3 : int = prim::Constant[value=0]()
  //    %10 : Tensor = prim::VarConcat(%0, %1, %2, %3)
  //    %12 : Tensor = prim::VarConcat(%10, %0, %3) // UPDATED
  //    %8 : Tensor[] = prim::ListConstruct(%10, %12)
  //    return (%8)
  testing::FileCheck()
      .check_count("= prim::VarConcat(%0, %1, %2, %3)", 1, /*exactly*/ true)
      ->check_count("= prim::VarConcat(%10, %0, %3)", 1, /*exactly*/ true)
      ->check_count("= prim::ListConstruct(%10, %12)", 1, /*exactly*/ true)
      ->check_count("= aten::cat(", 0, /*exactly*/ true)
      ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
      ->run(*graph);
}

TEST(ConcatOpt, CombineConcatsSimpleCase) {
  auto graph = std::make_shared<Graph>();
  const std::string input =
      R"IR(
        graph(%0: Tensor):
          %dim : int = prim::Constant[value=0]()
          %input.1 : Tensor[] = prim::ListConstruct(%0, %0)
          %concat.1 : Tensor = aten::cat(%input.1, %dim)
          %input.2 : Tensor[] = prim::ListConstruct(%concat.1, %0)
          %concat.2 : Tensor = aten::cat(%input.2, %dim)
          return (%concat.2)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {at::rand({1})};
  auto orig_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(CombineConcats(graph));
  graph->lint();
  auto opt_outputs = runGraph(graph, inputs);
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // After performing CombineConcats:
  //  graph(%0 : Tensor):
  //    %dim : int = prim::Constant[value=0]()
  //    %input : Tensor[] = prim::ListConstruct(%0, %0, %0)
  //    %concat : Tensor = aten::cat(%input, %dim)
  //    return (%concat)
  testing::FileCheck()
      .check_count("prim::ListConstruct", 1, /*exactly*/ true)
      ->check_count("aten::cat", 1, /*exactly*/ true)
      ->run(*graph);
}

TEST(ConcatOpt, CombineConcatsLongChain) {
  auto graph = std::make_shared<Graph>();
  const std::string input =
      R"IR(
        graph(%0: Tensor, %1 : Tensor):
          %dim : int = prim::Constant[value=0]()
          %input.1 : Tensor[] = prim::ListConstruct(%0, %0)
          %concat.1 : Tensor = aten::cat(%input.1, %dim)
          %input.2 : Tensor[] = prim::ListConstruct(%1, %concat.1, %1)
          %concat.2 : Tensor = aten::cat(%input.2, %dim)
          %input.3 : Tensor[] = prim::ListConstruct(%0, %concat.2, %0)
          %concat.3 : Tensor = aten::cat(%input.3, %dim)
          return (%concat.3)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {at::rand({1}), at::randn({1})};
  auto orig_outputs = runGraph(graph, inputs);

  ASSERT_TRUE(CombineConcats(graph));
  graph->lint();
  auto opt_outputs = runGraph(graph, inputs);
  ASSERT_TRUE(exactlyEqual(orig_outputs, opt_outputs));

  // After performing CombineConcats:
  //  graph(%0 : Tensor):
  //    %dim : int = prim::Constant[value=0]()
  //    %input : Tensor[] = prim::ListConstruct(%0, %1, %0, %0, %1, %0)
  //    %concat : Tensor = aten::cat(%input, %dim)
  //    return (%concat)
  testing::FileCheck()
      .check_count("prim::ListConstruct", 1, /*exactly*/ true)
      ->check_count("aten::cat", 1, /*exactly*/ true)
      ->run(*graph);
}

TEST(ConcatOpt, CombineConcatsMutation) {
  auto graph = std::make_shared<Graph>();
  const std::string input =
      R"IR(
        graph(%0: Tensor, %1 : Tensor):
          %dim : int = prim::Constant[value=0]()
          %input.1 : Tensor[] = prim::ListConstruct(%0, %0)
          %concat.1 : Tensor = aten::cat(%input.1, %dim)
          %input.2 : Tensor[] = prim::ListConstruct(%1, %concat.1, %1)
          %input.3 : Tensor[] = aten::append(%input.2, %0)
          %concat.2 : Tensor = aten::cat(%input.2, %dim)
          return (%concat.2)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {at::rand({1}), at::randn({1})};
  // No modifications due to aten::append
  ASSERT_FALSE(CombineConcats(graph));
}

} // namespace jit
} // namespace torch
