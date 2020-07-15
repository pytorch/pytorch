#include <test/cpp/tensorexpr/test_base.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/tensorexpr/buffer.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/torch.h>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace torch {
namespace jit {

using namespace torch::indexing;
using namespace torch::jit::tensorexpr;

void testKernel_1() {
  KernelScope kernel_scope;

  const auto graph_string = R"IR(
      graph(%0 : Float(5:3,3:1),
            %1 : Float(5:3,3:1)):
        %2 : Float(5:3,3:1) = aten::mul(%0, %1)
        %3 : Float(5:3,3:1) = aten::mul(%0, %2)
        return (%3))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto o = at::zeros({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = a * (a * b);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {a, b};
  Stmt* s = k.getCodeGenStmt();
  // TODO: verify stmt

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  o = stack[0].toTensor();
  for (size_t i = 0; i < 5 * 3; i++) {
    CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
}

void testKernel_2() {
  KernelScope kernel_scope;

  const auto graph_string = R"IR(
      graph(%0 : Float(5:3,3:1),
            %1 : Float(5:1,3:5)):
        %2 : Float(5:3,3:1) = aten::mul(%0, %1)
        %3 : Float(5:3,3:1) = aten::mul(%0, %2)
        return (%3))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b =
      at::rand({3, 5}, TensorOptions(kCPU).dtype(at::kFloat)).transpose(0, 1);
  auto o = at::zeros({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = a * (a * b);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {a, b};
  Stmt* s = k.getCodeGenStmt();
  // TODO: verify stmt

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  o = stack[0].toTensor();
  for (size_t i = 0; i < 5 * 3; i++) {
    CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
}

void testKernel_3() {
  KernelScope kernel_scope;

  const auto graph_string = R"IR(
      graph(%0 : Float(5:3,3:1),
            %1 : Float(5:12,3:2)):
        %2 : Float(5:3,3:1) = aten::mul(%0, %1)
        %3 : Float(5:3,3:1) = aten::mul(%0, %2)
        return (%3))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({10, 6}, TensorOptions(kCPU).dtype(at::kFloat))
               .index({Slice(None, None, 2), Slice(None, None, 2)});
  auto o = at::zeros({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = a * (a * b);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {a, b};
  Stmt* s = k.getCodeGenStmt();
  // TODO: verify stmt

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  o = stack[0].toTensor();
  for (size_t i = 0; i < 5 * 3; i++) {
    CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
}

void testKernel_4() {
  // Test TensorExpr shape inference capabilities: it should only require shapes
  // for the inputs
  {
    KernelScope kernel_scope;

    const auto graph_string = R"IR(
      graph(%0 : Float(5:3,3:1),
            %1 : Float(5:12,3:2)):
        %2 : Tensor = aten::mul(%0, %1)
        %3 : Tensor = aten::mul(%0, %2)
        return (%3))IR";
    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
    auto b = at::rand({10, 6}, TensorOptions(kCPU).dtype(at::kFloat))
                 .index({Slice(None, None, 2), Slice(None, None, 2)});
    auto o = at::zeros({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
    auto ref = a * (a * b);
    TensorExprKernel k(graph);
    std::vector<at::Tensor> inputs = {a, b};
    Stmt* s = k.getCodeGenStmt();

    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    o = stack[0].toTensor();
    for (size_t i = 0; i < 5 * 3; i++) {
      CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
    }
  }
  {
    KernelScope kernel_scope;

    const auto graph_string = R"IR(
      graph(%0 : Float(8:8,8:1),
            %1 : Float(8:8,8:1)):
        %2 : Tensor = aten::mul(%0, %1)
        %3 : Tensor, %4 : Tensor = prim::ConstantChunk[dim=1,chunks=2](%2)
        %r : Tensor = aten::mul(%3, %4)
        return (%r))IR";
    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    auto a = at::rand({8, 8}, TensorOptions(kCPU).dtype(at::kFloat));
    auto b = at::rand({8, 8}, TensorOptions(kCPU).dtype(at::kFloat));
    auto o = at::zeros({8, 4}, TensorOptions(kCPU).dtype(at::kFloat));
    auto t = torch::chunk(a * b, 2, 1);
    auto ref = t[0] * t[1];
    TensorExprKernel k(graph);
    std::vector<at::Tensor> inputs = {a, b};
    Stmt* s = k.getCodeGenStmt();

    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    o = stack[0].toTensor();
    CHECK_EQ(o.sizes()[0], 8);
    CHECK_EQ(o.sizes()[1], 4);
    for (size_t i = 0; i < 8 * 4; i++) {
      CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
    }
  }
}

} // namespace jit
} // namespace torch
