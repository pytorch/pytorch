#include <gtest/gtest.h>

#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/concat_opt.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {

namespace {

void checkOutputs(
    const std::vector<at::Tensor>& out1,
    const std::vector<at::Tensor>& out2) {
  ASSERT_EQ(out1.size(), out2.size());
  for (size_t i = 0; i < out1.size(); ++i) {
    ASSERT_EQ(out1[i].sizes(), out2[i].sizes());
    float max_diff = (out1[i] - out2[i]).abs().max().item<double>();
    ASSERT_EQ(max_diff, 0);
  }
}

std::vector<at::Tensor> runGraph(
    std::shared_ptr<Graph> graph,
    const std::vector<at::Tensor> inputs) {
  std::vector<IValue> stack = fmap<IValue>(inputs);
  Code code(graph, "test");
  InterpreterState(code).run(stack);
  TORCH_INTERNAL_ASSERT(!stack.empty());
  // Graph outputs that are handled below:
  //   * A list of Tensors.
  //   * 1 Tensor.
  if (stack.front().isTensorList()) {
    return stack.front().toTensorVector();
  }
  TORCH_INTERNAL_ASSERT(stack.front().isTensor());
  return {stack.front().toTensor()};
}

} // namespace

TEST(OptimizeConcatTest, ConcatWithDifferentOrderInput) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %5 : int = prim::Constant[value=0]()

          #CHECK: prim::ListConstruct(%0, %1)
          %features.1 : Tensor[] = prim::ListConstruct(%0, %1)
          #CHECK: aten::cat
          %concat.1 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%features.1, %5)

          #CHECK: prim::ListConstruct(%1, %0)
          %features.2 : Tensor[] = prim::ListConstruct(%1, %0)
          #CHECK: aten::cat
          %concat.2 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%features.2, %5)

          #CHECK: prim::ListConstruct
          %res : Tensor[] = prim::ListConstruct(%concat.1, %concat.2)
          return (%res)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU), at::rand({32, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  EliminateConcatCommonInputs(graph);
  graph->lint();
  auto opt_outputs = runGraph(graph, inputs);

  checkOutputs(orig_outputs, opt_outputs);

  // No optimizations should have happened in this case since the inputs
  // to the `cat` are in different order.
  testing::FileCheck().run(input, *graph);
}

TEST(OptimizeConcatTest, ExpandConcat) {
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

  checkOutputs(orig_outputs, opt_outputs);

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

TEST(OptimizeConcatTest, SimpleCommonInputsElimination) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %5 : int = prim::Constant[value=0]()

          %features.2 : Tensor[] = prim::ListConstruct(%0, %1)
          %concat.2 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%features.2, %5)

          %features.3 : Tensor[] = prim::ListConstruct(%0, %1, %2)
          %concat.3 : Float(128, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%features.3, %5)

          %res : Tensor[] = prim::ListConstruct(%concat.2, %concat.3)
          return (%res)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  {
    // Check EliminateConcatCommonInputs pass.
    auto graph1 = graph->copy();
    EliminateConcatCommonInputs(graph1);
    graph1->lint();
    auto opt_outputs = runGraph(graph1, inputs);
    checkOutputs(orig_outputs, opt_outputs);

    // After EliminateConcatCommonInputs, only the common elements in the list
    // input of `cat` ops will be replaced with the previous `cat` results, if
    // found. The number of `cat` ops and their inputs, `ListConstruct` ops,
    // will remain the same as in the input.
    //
    // Graph after EliminateConcatCommonInputs:
    //  graph(%0 : ...,
    //        %1 : ...,
    //        %2 : ...):
    //    %3 : int = prim::Constant[value=0]()
    //    %4 : Tensor[] = prim::ListConstruct(%0, %1)
    //    %5 : Tensor = aten::cat(%4, %3)
    //    %9 : Tensor[] = prim::ListConstruct(%5, %2) // UPDATED
    //    %7 : Tensor = aten::cat(%9, %3)
    //    %8 : Tensor[] = prim::ListConstruct(%5, %7)
    //    return (%8)

    testing::FileCheck()
        .check_count("= prim::ListConstruct(%0, %1)", 1, /*exactly*/ true)
        ->check_count("= aten::cat(", 1, /*exactly*/ true)
        ->check_count("= prim::ListConstruct(%5, %2)", 1, /*exactly*/ true)
        ->check_count("= aten::cat(", 1, /*exactly*/ true)
        ->check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
        ->check_count("= aten::cat(", 0, /*exactly*/ true)
        ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
        ->run(*graph1);
  }

  {
    // Check the entire Concat opt pass.
    OptimizeConcat(graph);
    graph->lint();
    auto opt_outputs = runGraph(graph, inputs);
    checkOutputs(orig_outputs, opt_outputs);

    // After full concat optimization we should have the following graph:
    //
    //  graph(%0 : ...,
    //        %1 : ...,
    //        %2 : ...):
    //    ...
    //    %29 : int[] = prim::ListConstruct(%26, %13, %13)
    //    %30 : Tensor = aten::empty(%29, ...) // concat.3 buffer
    //    %33 : Tensor = aten::slice(%30, ...) // slice for concat.2
    //    %19 : Tensor = aten::slice(%33, ...) // slice for concat.2 inp %0
    //    %20 : Tensor = aten::copy_(%19, %0)
    //    %22 : Tensor = aten::slice(%33, ...) // slice for concat.2 inp %1
    //    %23 : Tensor = aten::copy_(%22, %1)
    //    %36 : Tensor = aten::slice(%30, ...) // slice for concat.3 inp %2
    //    %37 : Tensor = aten::copy_(%36, %2)
    //    %8 : Tensor[] = prim::ListConstruct(%33, %30)
    //    return (%8)
    testing::FileCheck()
        .check_count("= aten::cat(", 0, /*exactly*/ true)
        ->check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
        ->check_count("= aten::empty(", 1, /*exactly*/ true)
        ->check_count("= aten::slice(", 2, /*exactly*/ true)
        ->check_count("= aten::copy_(", 1, /*exactly*/ true)
        ->check_count("= aten::slice(", 1, /*exactly*/ true)
        ->check_count("= aten::copy_(", 1, /*exactly*/ true)
        ->check_count("= aten::slice(", 1, /*exactly*/ true)
        ->check_count("= aten::copy_(", 1, /*exactly*/ true)
        ->check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
        ->run(*graph);
  }
}

TEST(OptimizeConcatTest, SimpleCommonInputsElimination2) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %5 : int = prim::Constant[value=0]()

          %features.2 : Tensor[] = prim::ListConstruct(%1, %2)
          %concat.2 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%features.2, %5)

          %features.3 : Tensor[] = prim::ListConstruct(%0, %1, %2)
          %concat.3 : Float(128, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%features.3, %5)

          %res : Tensor[] = prim::ListConstruct(%concat.2, %concat.3)
          return (%res)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  {
    // Check EliminateConcatCommonInputs pass.
    auto graph1 = graph->copy();
    EliminateConcatCommonInputs(graph1);
    graph1->lint();
    auto opt_outputs = runGraph(graph1, inputs);
    checkOutputs(orig_outputs, opt_outputs);

    // After EliminateConcatCommonInputs, only the common elements in the list
    // input of `cat` ops will be replaced with the previous `cat` results, if
    // found. The number of `cat` ops and their inputs, `ListConstruct` ops,
    // will remain the same as in the input.
    //
    // Graph after EliminateConcatCommonInputs:
    //  graph(%0 : ...,
    //        %1 : ...,
    //        %2 : ...):
    //    %3 : int = prim::Constant[value=0]()
    //    %4 : Tensor[] = prim::ListConstruct(%1, %2)
    //    %5 : Tensor = aten::cat(%4, %3)
    //    %9 : Tensor[] = prim::ListConstruct(%0, %5) // UPDATED
    //    %7 : Tensor = aten::cat(%9, %3)
    //    %8 : Tensor[] = prim::ListConstruct(%5, %7)
    //    return (%8)

    testing::FileCheck()
        .check_count("= prim::ListConstruct(%1, %2)", 1, /*exactly*/ true)
        ->check_count("= aten::cat(", 1, /*exactly*/ true)
        ->check_count("= prim::ListConstruct(%0, %5)", 1, /*exactly*/ true)
        ->check_count("= aten::cat(", 1, /*exactly*/ true)
        ->check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
        ->check_count("= aten::cat(", 0, /*exactly*/ true)
        ->check_count("= prim::ListConstruct(", 0, /*exactly*/ true)
        ->run(*graph1);
  }

  {
    // Check the entire Concat opt pass.
    OptimizeConcat(graph);
    graph->lint();
    auto opt_outputs = runGraph(graph, inputs);
    checkOutputs(orig_outputs, opt_outputs);

    // After full concat optimization we should have the following graph:
    //
    //  graph(%0 : ...,
    //        %1 : ...,
    //        %2 : ...):
    //    ...
    //    %29 : int[] = prim::ListConstruct(%26, %13, %13)
    //    %30 : Tensor = aten::empty(%29, ...) // concat.3 buffer
    //    %33 : Tensor = aten::slice(%30, ...) // slice for concat.2
    //    %19 : Tensor = aten::slice(%33, ...) // slice for concat.2 inp %0
    //    %20 : Tensor = aten::copy_(%19, %0)
    //    %22 : Tensor = aten::slice(%33, ...) // slice for concat.2 inp %1
    //    %23 : Tensor = aten::copy_(%22, %1)
    //    %36 : Tensor = aten::slice(%30, ...) // slice for concat.3 inp %2
    //    %37 : Tensor = aten::copy_(%36, %2)
    //    %8 : Tensor[] = prim::ListConstruct(%33, %30)
    //    return (%8)
    testing::FileCheck()
        .check_count("= aten::cat(", 0, /*exactly*/ true)
        ->check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
        ->check_count("= aten::empty(", 1, /*exactly*/ true)
        ->check_count("= aten::slice(", 2, /*exactly*/ true)
        ->check_count("= aten::copy_(", 1, /*exactly*/ true)
        ->check_count("= aten::slice(", 1, /*exactly*/ true)
        ->check_count("= aten::copy_(", 1, /*exactly*/ true)
        ->check_count("= aten::slice(", 1, /*exactly*/ true)
        ->check_count("= aten::copy_(", 1, /*exactly*/ true)
        ->check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
        ->run(*graph);
  }
}

TEST(OptimizeConcatTest, MoreCommonInputsElimination) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %5 : int = prim::Constant[value=0]()
          %features.1 : Tensor[] = prim::ListConstruct(%0)
          %concat.1 : Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%features.1, %5)

          %features.2 : Tensor[] = prim::ListConstruct(%0, %1)
          %concat.2 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%features.2, %5)

          %features.3 : Tensor[] = prim::ListConstruct(%0, %1, %2)
          %concat.3 : Float(128, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%features.3, %5)

          %res : Tensor[] = prim::ListConstruct(%concat.1, %concat.2, %concat.3)
          return (%res)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  {
    // Check EliminateConcatCommonInputs pass.
    auto graph1 = graph->copy();
    EliminateConcatCommonInputs(graph1);
    graph1->lint();
    auto opt_outputs = runGraph(graph1, inputs);
    checkOutputs(orig_outputs, opt_outputs);

    // After EliminateConcatCommonInputs, only the common elements in the list
    // input of `cat` ops will be replaced with the previous `cat` results, if
    // found. The number of `cat` ops and their inputs, `ListConstruct` ops,
    // will remain the same as in the input.
    testing::FileCheck()
        .check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
        ->check_count("= aten::cat(", 1, /*exactly*/ true)
        ->check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
        ->check_count("= aten::cat(", 1, /*exactly*/ true)
        ->check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
        ->check_count("= aten::cat(", 1, /*exactly*/ true)
        ->check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
        ->run(*graph1);
  }

  {
    // Check the entire Concat opt pass.
    OptimizeConcat(graph);
    graph->lint();
    auto opt_outputs = runGraph(graph, inputs);
    checkOutputs(orig_outputs, opt_outputs);

    // After full concat optimization we should have the following:
    //   prim::ListConstruct - to construct the input sizes for empty
    //   aten::empty - for the final `aten::cat` buffer.
    //   aten::slice - slice for concat.2
    //   aten::slice - slice for concat.1
    //   aten::copy_ - copy %0
    //   aten::slice - slice for concat.2 input 1
    //   aten::copy_ - copy %1
    //   aten::slice - slice for concat.3 input 2
    //   aten::copy_ - copy %2
    //   prim::ListConstruct - for the result
    testing::FileCheck()
        .check_count("= aten::cat(", 0, /*exactly*/ true)
        ->check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
        ->check_count("= aten::empty(", 1, /*exactly*/ true)
        ->check_count("= aten::slice(", 3, /*exactly*/ true)
        ->check_count("= aten::copy_(", 1, /*exactly*/ true)
        ->check_count("= aten::slice(", 1, /*exactly*/ true)
        ->check_count("= aten::copy_(", 1, /*exactly*/ true)
        ->check_count("= aten::slice(", 1, /*exactly*/ true)
        ->check_count("= aten::copy_(", 1, /*exactly*/ true)
        ->check_count("= prim::ListConstruct(", 1, /*exactly*/ true)
        ->run(*graph);
  }
}

TEST(OptimizeConcatTest, MoreCommonInputsElimination2) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %3: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %4: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %5 : int = prim::Constant[value=0]()
          %features.1 : Tensor[] = prim::ListConstruct(%0)
          %concat.1 : Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%features.1, %5)

          %features.2 : Tensor[] = prim::ListConstruct(%0, %1)
          %concat.2 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%features.2, %5)

          %features.3 : Tensor[] = prim::ListConstruct(%0, %1, %2)
          %concat.3 : Float(128, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%features.3, %5)

          %features.4 : Tensor[] = prim::ListConstruct(%0, %1, %2, %3)
          %concat.4 : Float(160, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%features.4, %5)

          %features.5 : Tensor[] = prim::ListConstruct(%0, %1, %2, %3, %4)
          %concat.5 : Float(192, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%features.5, %5)

          %res : Tensor[] = prim::ListConstruct(%concat.1, %concat.2, %concat.3, %concat.4, %concat.5)
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

  OptimizeConcat(graph);
  graph->lint();
  auto opt_outputs = runGraph(graph, inputs);

  checkOutputs(orig_outputs, opt_outputs);

  testing::FileCheck()
      .check_count("= aten::cat(", 0, /*exactly*/ true)
      ->check_count("= aten::empty(", 1, /*exactly*/ true)
      ->check_count("= aten::copy_(", 5, /*exactly*/ true)
      ->check_count("= aten::cat(", 0, /*exactly*/ true)
      ->check_count("= aten::empty(", 0, /*exactly*/ true)
      ->run(*graph);
}

TEST(OptimizeConcatTest, ConcatWithoutResultShape) {
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

  checkOutputs(orig_outputs, opt_outputs);

  // No optimizations should have happened in this case since the output
  // shape of `aten::cat` is not known.
  testing::FileCheck().run(input, *graph);
}

TEST(OptimizeConcatTest, ConcatWithoutInputShape) {
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

  checkOutputs(orig_outputs, opt_outputs);

  // No optimizations should have happened in this case since the shape of %5,
  // which is an input to `aten::cat`, is not known.
  testing::FileCheck().run(input, *graph);
}

TEST(OptimizeConcatTest, NoOptimizationWhenInputListIsMutated) {
  auto graph = std::make_shared<Graph>();

  const std::string input =
      R"IR(
        graph(%0: Float(64, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %1: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu),
              %2: Float(32, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu)):
          %5 : int = prim::Constant[value=0]()

          # CHECK: ListConstruct
          %features.2 : Tensor[] = prim::ListConstruct(%0, %1)
          # CHECK: aten::append
          %6 : Tensor [] = aten::append(%features.2, %2)
          # CHECK: aten::cat
          %concat.2 : Float(96, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%features.2, %5)

          # CHECK: ListConstruct
          %features.3 : Tensor[] = prim::ListConstruct(%0, %1, %2)
          # CHECK: aten::append
          %7 : Tensor [] = aten::append(%features.3, %0)
          # CHECK: aten::cat
          %concat.3 : Float(160, 56, 56, strides=[3136, 56, 1], requires_grad=0, device=cpu) = aten::cat(%features.3, %5)

          %res : Tensor[] = prim::ListConstruct(%concat.2, %concat.3)
          return (%res)
      )IR";
  parseIR(input, graph.get());
  std::vector<at::Tensor> inputs = {
      at::rand({64, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU),
      at::rand({32, 56, 56}, at::kCPU)};
  auto orig_outputs = runGraph(graph, inputs);

  {
    // Check EliminateConcatCommonInputs pass.
    auto graph1 = graph->copy();
    EliminateConcatCommonInputs(graph1);
    graph1->lint();
    auto opt_outputs = runGraph(graph1, inputs);
    checkOutputs(orig_outputs, opt_outputs);

    // No optimizations should have happened since the input lists to cat
    // are being mutated in the graph.
    testing::FileCheck().run(input, *graph);
  }

  {
    // Check the entire Concat opt pass.
    OptimizeConcat(graph);
    graph->lint();
    auto opt_outputs = runGraph(graph, inputs);
    checkOutputs(orig_outputs, opt_outputs);

    // No optimizations should have happened since the input lists to cat
    // are being mutated in the graph.
    testing::FileCheck().run(input, *graph);
  }
}

} // namespace jit
} // namespace torch
