#include <gtest/gtest.h>

#include <ATen/code_template.h>
#include <c10/core/DeviceType.h>
#include <test/cpp/tensorexpr/test_base.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/symbolic_shape_runtime_fusion.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace torch {
namespace jit {

using namespace torch::indexing;
using namespace torch::jit::tensorexpr;

TEST(DynamicShapes, SimpleGraph) {
#ifdef TORCH_ENABLE_LLVM
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  const auto graph_string = R"IR(
      graph(%x : Tensor,
            %SS_2 : int,
            %SS_3 : int):
        %3 : Tensor = aten::tanh(%x)
        %4 : Tensor = aten::erf(%3)
        return (%4))IR";
  torch::jit::parseIR(graph_string, graph.get());

  auto x_inp = graph->inputs()[0];
  auto x_type = TensorType::create(at::rand({10, 5}));
  std::vector<ShapeSymbol> x_sym_dims(
      {c10::ShapeSymbol::newSymbol(), c10::ShapeSymbol::newSymbol()});
  auto x_sym_type = x_type->withSymbolicShapes(x_sym_dims);
  graph->inputs().at(0)->setType(x_sym_type);
  for (const auto n : graph->nodes()) {
    n->output()->setType(x_sym_type);
  }

  // Graph with symbolic shapes:
  //
  // graph(%x : Float(SS(-2), SS(-3)),
  //       %SS_2 : int,
  //       %SS_3 : int):
  //   %3 : Float(SS(-2), SS(-3)) = aten::tanh(%x)
  //   %4 : Float(SS(-2), SS(-3)) = aten::erf(%3)
  //   return (%4)

  std::vector<torch::jit::StrideInput> input_desc = {
      torch::jit::StrideInput::TENSOR_CONT};
  std::unordered_map<
      const torch::jit::Value*,
      std::vector<torch::jit::StrideInput>>
      symbolic_strides;
  symbolic_strides[x_inp] = input_desc;
  symbolic_strides[graph->outputs().at(0)] = input_desc;
  std::vector<int64_t> symbolic_shape_inputs = c10::fmap(
      x_sym_dims,
      [](const c10::ShapeSymbol& shapeSym) { return shapeSym.value(); });

  TensorExprKernel kernel(
      graph, {}, symbolic_shape_inputs, false, symbolic_strides);
  // Run with the same static dims as the one we initialized the graph with.
  {
    auto a = at::rand({10, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto ref = at::erf(at::tanh(a));

    std::vector<IValue> stack = fmap<IValue>(std::vector<at::Tensor>({a}));
    stack.push_back(10);
    stack.push_back(5);
    kernel.run(stack);

    auto o = stack[0].toTensor();
    ASSERT_TRUE(at::allclose(o, ref));
  }

  // Run with inputs having different dims.
  {
    auto a = at::rand({50, 100}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto ref = at::erf(at::tanh(a));

    std::vector<IValue> stack = fmap<IValue>(std::vector<at::Tensor>({a}));
    stack.push_back(50);
    stack.push_back(100);
    kernel.run(stack);

    auto o = stack[0].toTensor();
    ASSERT_TRUE(at::allclose(o, ref));
  }
#endif
}

TEST(DynamicShapes, GraphWith2InputsSameDims) {
#ifdef TORCH_ENABLE_LLVM
  // The two inputs in this graph must have the same dims.
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  const auto graph_string = R"IR(
      graph(%x : Tensor,
            %y : Tensor,
            %SS_2 : int,
            %SS_3 : int):
        %3 : Tensor = aten::tanh(%x)
        %4 : Tensor = aten::erf(%3)
        %5 : Tensor = aten::mul(%4, %y)
        return (%5))IR";
  torch::jit::parseIR(graph_string, graph.get());

  auto x_inp = graph->inputs()[0];
  auto y_inp = graph->inputs()[1];
  auto x_type = TensorType::create(at::rand({10, 5}));
  std::vector<ShapeSymbol> x_sym_dims(
      {c10::ShapeSymbol::newSymbol(), c10::ShapeSymbol::newSymbol()});
  auto x_sym_type = x_type->withSymbolicShapes(x_sym_dims);
  graph->inputs().at(0)->setType(x_sym_type);
  graph->inputs().at(1)->setType(x_sym_type);
  for (const auto n : graph->nodes()) {
    n->output()->setType(x_sym_type);
  }

  // Graph with symbolic shapes:
  //
  // graph(%x : Float(SS(-4), SS(-5)),
  //       %y : Float(SS(-4), SS(-5)),
  //       %SS_2 : int,
  //       %SS_3 : int):
  //   %4 : Float(SS(-4), SS(-5)) = aten::tanh(%x)
  //   %5 : Float(SS(-4), SS(-5)) = aten::erf(%4)
  //   %6 : Float(SS(-4), SS(-5)) = aten::mul(%5, %y)
  //   return (%6)

  std::vector<int64_t> symbolic_shape_inputs = c10::fmap(
      x_sym_dims,
      [](const c10::ShapeSymbol& shapeSym) { return shapeSym.value(); });

  std::vector<torch::jit::StrideInput> input_desc = {
      torch::jit::StrideInput::TENSOR_CONT};
  std::unordered_map<
      const torch::jit::Value*,
      std::vector<torch::jit::StrideInput>>
      symbolic_strides;
  symbolic_strides[x_inp] = input_desc;
  symbolic_strides[y_inp] = input_desc;
  symbolic_strides[graph->outputs().at(0)] = input_desc;

  TensorExprKernel kernel(
      graph, {}, symbolic_shape_inputs, false, symbolic_strides);

  // Run with the same static dims as the one we initialized the graph with.
  {
    auto a = at::rand({10, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto b = at::rand({10, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto ref = at::mul(at::erf(at::tanh(a)), b);

    std::vector<IValue> stack = fmap<IValue>(std::vector<at::Tensor>({a, b}));
    stack.push_back(10);
    stack.push_back(5);
    kernel.run(stack);

    auto o = stack[0].toTensor();
    ASSERT_TRUE(at::allclose(o, ref));
  }

  // Run with inputs having different dims.
  {
    auto a = at::rand({50, 100}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto b = at::rand({50, 100}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto ref = at::mul(at::erf(at::tanh(a)), b);

    std::vector<IValue> stack = fmap<IValue>(std::vector<at::Tensor>({a, b}));
    stack.push_back(50);
    stack.push_back(100);
    kernel.run(stack);

    auto o = stack[0].toTensor();
    ASSERT_TRUE(at::allclose(o, ref));
  }
#endif
}

TEST(DynamicShapes, GraphWith2InputsAndBroadcast) {
#ifdef TORCH_ENABLE_LLVM
  // The second input to the graph has a dim of size 1 which should be
  // broadcasted in the at::mul op.
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  const auto graph_string = R"IR(
      graph(%x : Float(10, 5, requires_grad=0, device=cpu),
            %y : Float(1, 5, requires_grad=0, device=cpu),
            %SS_2 : int,
            %SS_3 : int):
        %3 : Tensor = aten::tanh(%x)
        %4 : Tensor = aten::erf(%3)
        %5 : Tensor = aten::mul(%4, %y)
        return (%5))IR";
  torch::jit::parseIR(graph_string, graph.get());

  auto x_inp = graph->inputs()[0];
  auto y_inp = graph->inputs()[1];
  auto x_type = TensorType::create(at::rand({10, 5}));
  auto y_type = TensorType::create(at::rand({1, 5}));
  auto x_dim0_sym = c10::ShapeSymbol::newSymbol();
  auto x_dim1_sym = c10::ShapeSymbol::newSymbol();
  auto x_sym_type = x_type->withSymbolicShapes(
      std::vector<ShapeSymbol>({x_dim0_sym, x_dim1_sym}));
  auto y_sym_type = y_type->withSymbolicShapes(std::vector<ShapeSymbol>(
      {c10::ShapeSymbol::fromStaticSize(1), x_dim1_sym}));
  graph->inputs().at(0)->setType(x_sym_type);
  graph->inputs().at(1)->setType(y_sym_type);
  for (const auto n : graph->nodes()) {
    n->output()->setType(x_sym_type);
  }

  // Graph with symbolic shapes:
  //
  // graph(%x : Float(SS(-6), SS(-7)),
  //       %y : Float(1, SS(-7)),
  //       %SS_2 : int,
  //       %SS_3 : int):
  //   %4 : Float(SS(-6), SS(-7)) = aten::tanh(%x)
  //   %5 : Float(SS(-6), SS(-7)) = aten::erf(%4)
  //   %6 : Float(SS(-6), SS(-7)) = aten::mul(%5, %y)
  //   return (%6)

  std::vector<int64_t> symbolic_shape_inputs(
      {x_dim0_sym.value(), x_dim1_sym.value()});

  std::vector<torch::jit::StrideInput> input_desc = {
      torch::jit::StrideInput::TENSOR_CONT};
  std::unordered_map<
      const torch::jit::Value*,
      std::vector<torch::jit::StrideInput>>
      symbolic_strides;
  symbolic_strides[x_inp] = input_desc;
  symbolic_strides[y_inp] = input_desc;
  symbolic_strides[graph->outputs().at(0)] = input_desc;

  TensorExprKernel kernel(
      graph, {}, symbolic_shape_inputs, false, symbolic_strides);

  // Run with the same static dims as the one we initialized the graph with.
  {
    auto a = at::rand({10, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto b = at::rand({1, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto ref = at::mul(at::erf(at::tanh(a)), b);

    std::vector<IValue> stack = fmap<IValue>(std::vector<at::Tensor>({a, b}));
    stack.push_back(10);
    stack.push_back(5);
    kernel.run(stack);

    auto o = stack[0].toTensor();
    ASSERT_TRUE(at::allclose(o, ref));
  }

  // Run with inputs having different dims.
  {
    auto a = at::rand({50, 100}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto b = at::rand({1, 100}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto ref = at::mul(at::erf(at::tanh(a)), b);

    std::vector<IValue> stack = fmap<IValue>(std::vector<at::Tensor>({a, b}));
    stack.push_back(50);
    stack.push_back(100);
    kernel.run(stack);

    auto o = stack[0].toTensor();
    ASSERT_TRUE(at::allclose(o, ref));
  }
#endif
}

TEST(DynamicShapes, GraphWithPartiallySymbolicOutput) {
#ifdef TORCH_ENABLE_LLVM
  // The second input to the graph has a dim of size 1 which should be
  // broadcasted in the at::mul op.
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  const auto graph_string = R"IR(
      graph(%x : Float(1, 5, requires_grad=0, device=cpu),
            %y : Float(1, 5, requires_grad=0, device=cpu),
            %SS_2 : int):
        %4 : Tensor = aten::tanh(%x)
        %5 : Tensor = aten::mul(%4, %y)
        return (%5))IR";
  torch::jit::parseIR(graph_string, graph.get());

  auto x_inp = graph->inputs()[0];
  auto y_inp = graph->inputs()[1];
  auto x_type = TensorType::create(at::rand({1, 5}));
  auto x_dim1_sym = c10::ShapeSymbol::newSymbol();
  auto x_sym_type = x_type->withSymbolicShapes(std::vector<ShapeSymbol>(
      {c10::ShapeSymbol::fromStaticSize(1), x_dim1_sym}));
  graph->inputs().at(0)->setType(x_sym_type);
  graph->inputs().at(1)->setType(x_sym_type);
  for (const auto n : graph->nodes()) {
    n->output()->setType(x_sym_type);
  }

  // Graph with symbolic shapes:
  //
  // graph(%x : Float(1, SS(-2)),
  //       %y : Float(1, SS(-2)),
  //       %SS_2 : int):
  //   %3 : Float(1, SS(-2)) = aten::tanh(%x)
  //   %4 : Float(1, SS(-2)) = aten::mul(%3, %y)
  //   return (%4)

  std::vector<int64_t> symbolic_shape_inputs({x_dim1_sym.value()});

  std::vector<torch::jit::StrideInput> input_desc = {
      torch::jit::StrideInput::TENSOR_CONT};
  std::unordered_map<
      const torch::jit::Value*,
      std::vector<torch::jit::StrideInput>>
      symbolic_strides;
  symbolic_strides[x_inp] = input_desc;
  symbolic_strides[y_inp] = input_desc;
  symbolic_strides[graph->outputs().at(0)] = input_desc;

  TensorExprKernel kernel(
      graph, {}, symbolic_shape_inputs, false, symbolic_strides);

  // Run with the same static dims as the one we initialized the graph with.
  {
    auto a = at::rand({1, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto b = at::rand({1, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto ref = at::mul(at::tanh(a), b);

    std::vector<IValue> stack = fmap<IValue>(std::vector<at::Tensor>({a, b}));
    stack.push_back(5);
    kernel.run(stack);

    auto o = stack[0].toTensor();
    ASSERT_TRUE(at::allclose(o, ref));
  }

  // Run with inputs having different dims.
  {
    auto a = at::rand({1, 100}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto b = at::rand({1, 100}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto ref = at::mul(at::tanh(a), b);

    std::vector<IValue> stack = fmap<IValue>(std::vector<at::Tensor>({a, b}));
    stack.push_back(100);
    kernel.run(stack);

    auto o = stack[0].toTensor();
    ASSERT_TRUE(at::allclose(o, ref));
  }
#endif
}

TEST(DynamicShapes, GraphWithSymbolicStrides) {
#ifdef TORCH_ENABLE_LLVM
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  const auto graph_string = R"IR(
    graph(%0 : Float(SS(-2), SS(-3), requires_grad=0, device=cpu),
          %1 : Float(SS(-2), SS(-3), requires_grad=0, device=cpu),
          %SS_3 : int,
          %SS_2 : int):
      %15 : int = prim::Constant[value=1]()
      %21 : Float(SS(-2), SS(-3), requires_grad=0, device=cpu) = aten::add(%0, %1, %15)
      %22 : Float(SS(-2), SS(-3), requires_grad=0, device=cpu) = aten::mul(%21, %0)
      return (%22))IR";
  parseIR(graph_string, &*graph);

  std::vector<torch::jit::StrideInput> input_desc = {
      torch::jit::StrideInput::S_AS_ARG, torch::jit::StrideInput::S_ONE};
  std::vector<torch::jit::StrideInput> output_desc = {
      torch::jit::StrideInput::TENSOR_CONT};
  std::unordered_map<
      const torch::jit::Value*,
      std::vector<torch::jit::StrideInput>>
      symbolic_strides;
  symbolic_strides[graph->inputs().at(0)] = input_desc;
  symbolic_strides[graph->inputs().at(1)] = input_desc;
  symbolic_strides[graph->outputs().at(0)] = output_desc;
  std::vector<int64_t> symbolic_shape_inputs = {-3, -2};
  TensorExprKernel k(graph, {}, symbolic_shape_inputs, false, symbolic_strides);

  {
    auto x0 = at::rand({10, 32}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto x1 = at::rand({10, 32}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto ref = at::mul(at::add(x0, x1, 1), x0);

    std::vector<at::Tensor> inputs = {x0, x1};
    std::vector<IValue> stack = at::fmap<at::IValue>(inputs);
    stack.push_back(32);
    stack.push_back(10);
    k.run(stack);

    auto o = stack[0].toTensor();
    ASSERT_TRUE(at::allclose(o, ref));
  }

  {
    auto x0 = at::rand({10, 32}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto x1 = at::rand({10, 32}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto out =
        at::rand({10, 32}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto ref = at::mul(at::add(x0, x1, 1), x0);

    std::vector<at::Tensor> inputs = {out, x0, x1};
    std::vector<IValue> stack = at::fmap<at::IValue>(inputs);
    stack.push_back(32);
    stack.push_back(10);
    k.runWithAllocatedOutputs(stack);

    ASSERT_TRUE(at::allclose(out, ref));
  }
#endif
}

TEST(DynamicShapes, GraphWithCatAndBroadcast) {
#ifdef TORCH_ENABLE_LLVM
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  const auto graph_string = R"IR(
      graph(%x : Float(10, 5, requires_grad=0, device=cpu),
            %y : Float(4, 5, requires_grad=0, device=cpu),
            %z : Float(1, 1, requires_grad=0, device=cpu),
            %SS_2 : int,
            %SS_3 : int,
            %SS_4 : int,
            %SS_5 : int):
        %11 : int = prim::Constant[value=0]()
        %3 : Tensor = aten::tanh(%x)
        %out1 : Tensor = aten::erf(%3)
        %out2 : Tensor = aten::relu(%y)
        %10 : Tensor[] = prim::ListConstruct(%out1, %out2)
        %25 : Tensor = aten::cat(%10, %11)
        %28 : Tensor = aten::hardswish(%25)
        %29 : Tensor = aten::mul(%28, %z)
        return (%29))IR";
  torch::jit::parseIR(graph_string, graph.get());

  auto x_inp = graph->inputs()[0];
  auto y_inp = graph->inputs()[1];
  auto z_inp = graph->inputs()[2];
  auto x_type = TensorType::create(at::rand({10, 5}));
  auto y_type = TensorType::create(at::rand({4, 5}));
  auto z_type = TensorType::create(at::rand({1, 1}));
  auto x_dim0_sym = c10::ShapeSymbol::newSymbol();
  auto x_dim1_sym = c10::ShapeSymbol::newSymbol();
  auto x_sym_type = x_type->withSymbolicShapes(
      std::vector<ShapeSymbol>({x_dim0_sym, x_dim1_sym}));
  auto y_dim0_sym = c10::ShapeSymbol::newSymbol();
  auto y_sym_type = y_type->withSymbolicShapes(
      std::vector<ShapeSymbol>({y_dim0_sym, x_dim1_sym}));
  graph->inputs().at(0)->setType(x_sym_type);
  graph->inputs().at(1)->setType(y_sym_type);
  auto cat_dim0_sym = c10::ShapeSymbol::newSymbol();
  auto cat_out_type = x_type->withSymbolicShapes(
      std::vector<ShapeSymbol>({cat_dim0_sym, x_dim1_sym}));
  auto nodeIt = graph->nodes().begin();
  ++nodeIt;
  nodeIt->output()->setType(x_sym_type); // aten::tanh
  ++nodeIt;
  nodeIt->output()->setType(x_sym_type); // aten::erf
  ++nodeIt;
  nodeIt->output()->setType(y_sym_type); // aten::relu
  ++nodeIt;
  ++nodeIt;
  nodeIt->output()->setType(cat_out_type); // aten::cat
  ++nodeIt;
  nodeIt->output()->setType(cat_out_type); // aten::hardswish
  ++nodeIt;
  nodeIt->output()->setType(cat_out_type); // aten::mul

  // Graph with symbolic shapes:
  //
  // graph(%x : Float(SS(-2), SS(-3)),
  //       %y : Float(SS(-4), SS(-3)),
  //       %z : Float(1, 1),
  //       %SS_2 : int,
  //       %SS_3 : int,
  //       %SS_4 : int,
  //       %SS_5 : int):
  //   %7 : int = prim::Constant[value=0]()
  //   %8 : Float(SS(-2), SS(-3)) = aten::tanh(%x)
  //   %9 : Float(SS(-2), SS(-3)) = aten::erf(%8)
  //   %10 : Float(SS(-4), SS(-3)) = aten::relu(%y)
  //   %11 : Tensor[] = prim::ListConstruct(%9, %10)
  //   %12 : Float(SS(-5), SS(-3)) = aten::cat(%11, %7)
  //   %13 : Float(SS(-5), SS(-3)) = aten::hardswish(%12)
  //   %14 : Float(SS(-5), SS(-3)) = aten::mul(%13, %z)
  //   return (%14)

  std::vector<int64_t> symbolic_shape_inputs(
      {x_dim0_sym.value(),
       x_dim1_sym.value(),
       y_dim0_sym.value(),
       cat_dim0_sym.value()});

  std::vector<torch::jit::StrideInput> input_desc = {
      torch::jit::StrideInput::TENSOR_CONT};
  std::unordered_map<
      const torch::jit::Value*,
      std::vector<torch::jit::StrideInput>>
      symbolic_strides;
  symbolic_strides[x_inp] = input_desc;
  symbolic_strides[y_inp] = input_desc;
  symbolic_strides[z_inp] = input_desc;
  symbolic_strides[graph->outputs().at(0)] = input_desc;

  TensorExprKernel kernel(
      graph, {}, symbolic_shape_inputs, false, symbolic_strides);

  auto a = at::rand({10, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto b = at::rand({4, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto c = at::rand({1, 1}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto ref = at::mul(
      at::hardswish(at::cat({at::erf(at::tanh(a)), at::relu(b)}, 0)), c);

  std::vector<IValue> stack = fmap<IValue>(std::vector<at::Tensor>({a, b, c}));
  stack.push_back(10);
  stack.push_back(5);
  stack.push_back(4);
  stack.push_back(14);
  kernel.run(stack);

  auto o = stack[0].toTensor();
  ASSERT_TRUE(at::allclose(o, ref));
#endif
}

TEST(DynamicShapes, GraphFromModel) {
#ifdef TORCH_ENABLE_LLVM
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  const auto graph_string = R"IR(
    graph(%0 : Float(SS(-2), SS(-3), requires_grad=0, device=cpu),
          %1 : Float(SS(-2), SS(-4), requires_grad=0, device=cpu),
          %2 : Float(SS(-2), SS(-5), requires_grad=0, device=cpu),
          %input.4 : Long(SS(-2), SS(-6), requires_grad=0, device=cpu),
          %4 : Float(SS(-7), requires_grad=0, device=cpu),
          %5 : Float(SS(-7), requires_grad=0, device=cpu),
          %SS_10 : int,
          %SS_9 : int,
          %SS_8 : int,
          %SS_7 : int,
          %SS_6 : int,
          %SS_5 : int,
          %SS_4 : int,
          %SS_3 : int,
          %SS_2 : int):
      %15 : int = prim::Constant[value=1]()
      %16 : bool = prim::Constant[value=0]()
      %17 : int = prim::Constant[value=6]()
      %18 : Float(SS(-2), SS(-6), strides=[139, 1], requires_grad=0, device=cpu) = aten::to(%input.4, %17, %16, %16)
      %19 : Tensor[] = prim::ListConstruct(%0, %1, %18, %2)
      %20 : Float(SS(-2), SS(-8), strides=[261, 1], requires_grad=0, device=cpu) = aten::cat(%19, %15)
      %21 : Float(SS(-2), SS(-9), strides=[261, 1], requires_grad=0, device=cpu) = aten::add(%20, %5, %15)
      %22 : Float(SS(-2), SS(-10), requires_grad=0, device=cpu) = aten::mul(%21, %4)
      return (%22))IR";
  parseIR(graph_string, &*graph);

  std::vector<torch::jit::StrideInput> input_desc = {
      torch::jit::StrideInput::TENSOR_CONT};
  std::unordered_map<
      const torch::jit::Value*,
      std::vector<torch::jit::StrideInput>>
      symbolic_strides;
  symbolic_strides[graph->inputs().at(0)] = input_desc;
  symbolic_strides[graph->inputs().at(1)] = input_desc;
  symbolic_strides[graph->inputs().at(2)] = input_desc;
  symbolic_strides[graph->inputs().at(3)] = input_desc;
  symbolic_strides[graph->inputs().at(4)] = input_desc;
  symbolic_strides[graph->inputs().at(5)] = input_desc;
  symbolic_strides[graph->outputs().at(0)] = input_desc;
  std::vector<int64_t> symbolic_shape_inputs = {
      -10, -9, -8, -7, -6, -5, -4, -3, -2};
  TensorExprKernel k(graph, {}, symbolic_shape_inputs, false, symbolic_strides);

  int64_t i2 = 10;
  int64_t i3 = 32;
  int64_t i4 = 19;
  int64_t i5 = 71;
  int64_t i6 = 139;
  int64_t i7 = 261;
  int64_t i8 = 261;
  int64_t i9 = 261;
  int64_t i10 = 261;
  auto x0 = at::rand({i2, i3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto x1 = at::rand({i2, i4}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto x2 = at::rand({i2, i5}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto x3 = at::ones({i2, i6}, at::TensorOptions(at::kCPU).dtype(at::kLong));
  auto x4 = at::rand({i7}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto x5 = at::rand({i8}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto ref = at::mul(at::add(at::cat({x0, x1, x3, x2}, 1), x5), x4);

  {
    std::vector<at::Tensor> inputs = {x0, x1, x2, x3, x4, x5};
    std::vector<IValue> stack = at::fmap<at::IValue>(inputs);
    stack.emplace_back(i10);
    stack.emplace_back(i9);
    stack.emplace_back(i8);
    stack.emplace_back(i7);
    stack.emplace_back(i6);
    stack.emplace_back(i5);
    stack.emplace_back(i4);
    stack.emplace_back(i3);
    stack.emplace_back(i2);
    k.run(stack);

    auto o = stack[0].toTensor();
    ASSERT_TRUE(at::allclose(o, ref));
  }

  {
    auto out =
        at::rand({i2, i10}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    std::vector<at::Tensor> inputs = {out, x0, x1, x2, x3, x4, x5};
    std::vector<IValue> stack = at::fmap<at::IValue>(inputs);
    stack.emplace_back(i10);
    stack.emplace_back(i9);
    stack.emplace_back(i8);
    stack.emplace_back(i7);
    stack.emplace_back(i6);
    stack.emplace_back(i5);
    stack.emplace_back(i4);
    stack.emplace_back(i3);
    stack.emplace_back(i2);
    k.runWithAllocatedOutputs(stack);

    ASSERT_TRUE(at::allclose(out, ref));
  }
#endif
}

TEST(DynamicShapes, MultiThreadedExecution) {
#ifdef TORCH_ENABLE_LLVM
  const auto graph_template = R"IR(
      graph(%x : Float(SS(-2), SS(-3), requires_grad=0, device=${device}),
            %y : Float(SS(-2), SS(-3), requires_grad=0, device=${device}),
            %SS_2 : int,
            %SS_3 : int):
        %3 : Float(SS(-2), SS(-3), requires_grad=0, device=${device}) = aten::tanh(%x)
        %4 : Float(SS(-2), SS(-3), requires_grad=0, device=${device}) = aten::erf(%3)
        %5 : Float(SS(-2), SS(-3), requires_grad=0, device=${device}) = aten::mul(%4, %y)
        return (%5))IR";
  for (bool use_cuda : {false, true}) {
    if (!torch::cuda::is_available() && use_cuda) {
      continue;
    }
    auto device = use_cuda ? at::kCUDA : at::kCPU;
    at::jit::TemplateEnv env;
    env.s("device", use_cuda ? "cuda:0" : "cpu");
    const auto graph_string = format(graph_template, env);
    std::shared_ptr<Graph> graph = std::make_shared<Graph>();
    torch::jit::parseIR(graph_string, graph.get());

    std::vector<int64_t> symbolic_shape_inputs = {-2, -3};

    std::vector<torch::jit::StrideInput> input_desc = {
        torch::jit::StrideInput::TENSOR_CONT};
    std::unordered_map<
        const torch::jit::Value*,
        std::vector<torch::jit::StrideInput>>
        symbolic_strides;
    symbolic_strides[graph->inputs().at(0)] = input_desc;
    symbolic_strides[graph->inputs().at(1)] = input_desc;
    symbolic_strides[graph->outputs().at(0)] = input_desc;

    TensorExprKernel kernel(
        graph, {}, symbolic_shape_inputs, false, symbolic_strides);

    auto run_kernel = [&](int dim1, int dim2) {
      auto a =
          at::rand({dim1, dim2}, at::TensorOptions(device).dtype(at::kFloat));
      auto b =
          at::rand({dim1, dim2}, at::TensorOptions(device).dtype(at::kFloat));

      auto ref = at::mul(at::erf(at::tanh(a)), b);

      std::vector<IValue> stack = fmap<IValue>(std::vector<at::Tensor>({a, b}));
      stack.emplace_back(dim1);
      stack.emplace_back(dim2);
      kernel.run(stack);

      auto o = stack[0].toTensor();
      ASSERT_TRUE(at::allclose(o, ref));
    };

    // Run the kernel in parallel to ensure that the run() method calls in
    // TensorExprKernel are not changing any state.
    constexpr size_t kNumThreads = 4;
    std::vector<std::thread> threads;
    for (size_t id = 0; id < kNumThreads; ++id) {
      threads.emplace_back(run_kernel, id + 5, id + 20);
    }
    for (auto& t : threads) {
      t.join();
    }
  }
#endif
}

} // namespace jit
} // namespace torch
