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
      graph(%0 : Float(5:3,3:1, device=cpu),
            %1 : Float(5:3,3:1, device=cpu)):
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
      graph(%0 : Float(5:3,3:1, device=cpu),
            %1 : Float(5:1,3:5, device=cpu)):
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
      graph(%0 : Float(5:3,3:1, device=cpu),
            %1 : Float(5:12,3:2, device=cpu)):
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
      graph(%0 : Float(5:3,  3:1, device=cpu),
            %1 : Float(5:12, 3:2, device=cpu)):
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
      graph(%0 : Float(8:8, 8:1, device=cpu),
            %1 : Float(8:8, 8:1, device=cpu)):
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
  {
    // Test that shape inference handles aten::unsqueeze
    KernelScope kernel_scope;

    const auto graph_string = R"IR(
      graph(%a : Float(4:2, 2:1, device=cpu),
            %b : Float(4:6, 3:2, 2:1, device=cpu),
            %c : Float(3:4, 2:2, 2:1, device=cpu)):
        %one : int = prim::Constant[value=1]()
        %minus_one : int = prim::Constant[value=-1]()
        %three : int = prim::Constant[value=3]()
        %minus_four : int = prim::Constant[value=-4]()
        %a1 : Tensor = aten::unsqueeze(%a, %one)        # new size: [4,1,2]
        %a2 : Tensor = aten::unsqueeze(%a1, %minus_one) # new size: [4,1,2,1]
        %b1 : Tensor = aten::unsqueeze(%b, %three)      # new size: [4,3,2,1]
        %c1 : Tensor = aten::unsqueeze(%c, %minus_four) # new size: [1,3,2,2]
        %ab : Tensor = aten::mul(%a2, %b1)         # expected size: [4,3,2,1]
        %abc : Tensor = aten::mul(%ab, %c1)        # expected size: [4,3,2,2]
        return (%abc))IR";
    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    auto a = at::rand({4, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto b = at::rand({4, 3, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto c = at::rand({3, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto o = at::zeros({4, 3, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto ref = at::unsqueeze(at::unsqueeze(a, 1), -1) * at::unsqueeze(b, 3) *
        at::unsqueeze(c, -4);

    TensorExprKernel k(graph);
    std::vector<at::Tensor> inputs = {a, b, c};
    Stmt* s = k.getCodeGenStmt();

    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    o = stack[0].toTensor();

    // Check sizes
    CHECK_EQ(o.sizes().size(), ref.sizes().size());
    size_t num_el = 1;
    for (auto idx = 0; idx < ref.sizes().size(); idx++) {
      CHECK_EQ(o.sizes()[idx], ref.sizes()[idx]);
      num_el *= ref.sizes()[idx];
    }

    // Check the contents
    for (size_t i = 0; i < num_el; i++) {
      CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
    }
  }
  {
    // Test that shape inference handles aten::cat
    KernelScope kernel_scope;

    const auto graph_string = R"IR(
      graph(%a : Float(5:6, 3:2, 2:1, device=cpu),
            %b : Float(5:14, 7:2, 2:1, device=cpu),
            %c : Float(5:18, 9:2, 2:1, device=cpu)):
        %dim : int = prim::Constant[value=1]()
        %inputs : Tensor[] = prim::ListConstruct(%a, %b, %c)
        %r : Tensor = aten::cat(%inputs, %dim)               # new size: [5,19,2]
        return (%r))IR";
    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    auto a = at::rand({5, 3, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto b = at::rand({5, 7, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto c = at::rand({5, 9, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto o = at::zeros({5, 19, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto ref = at::cat({a, b, c}, 1);

    TensorExprKernel k(graph);
    std::vector<at::Tensor> inputs = {a, b, c};
    Stmt* s = k.getCodeGenStmt();

    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    o = stack[0].toTensor();

    // Check sizes
    CHECK_EQ(o.sizes().size(), ref.sizes().size());
    size_t num_el = 1;
    for (auto idx = 0; idx < ref.sizes().size(); idx++) {
      CHECK_EQ(o.sizes()[idx], ref.sizes()[idx]);
      num_el *= ref.sizes()[idx];
    }

    // Check the contents
    for (size_t i = 0; i < num_el; i++) {
      CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
    }
  }
}

} // namespace jit
} // namespace torch
