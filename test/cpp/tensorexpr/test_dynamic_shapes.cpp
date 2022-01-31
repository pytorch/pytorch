#include <gtest/gtest.h>

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

} // namespace jit
} // namespace torch
