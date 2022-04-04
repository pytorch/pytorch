#include <benchmark/benchmark.h>

#include <ATen/code_template.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>

using namespace torch::jit;
using namespace torch::jit::tensorexpr;

static const std::string kernel_static_shapes_template = R"IR(
    graph(%0 : Float(${dim}, strides=[1], device=cpu),
          %1 : Float(${dim}, strides=[1], device=cpu)):
        %2 : Float(${dim}, strides=[1]) = aten::mul(%0, %1)
        %4 : Float(${dim}, strides=[1]) = aten::mul(%0, %2)
        return (%4))IR";

static const std::string kernel_symbolic_shapes = R"IR(
    graph(%0 : Float(SS(-2), strides=[1], device=cpu),
          %1 : Float(SS(-2), strides=[1], device=cpu),
          %SS_2 : int):
        %2 : Float(SS(-2), strides=[1]) = aten::mul(%0, %1)
        %4 : Float(SS(-2), strides=[1]) = aten::mul(%0, %2)
        return (%4))IR";

class KernelBench : public benchmark::Fixture {
 public:
  void Eager(benchmark::State& state) {
    auto dim = state.range(0);
    auto a = at::rand({dim}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto b = at::rand({dim}, at::TensorOptions(at::kCPU).dtype(at::kFloat));

    for (auto _ : state) {
      auto o = at::mul(a, at::mul(a, b));
    }
  }

  void GraphWithStaticShapes(benchmark::State& state) {
    auto dim = state.range(0);
    auto graph = std::make_shared<Graph>();
    at::jit::TemplateEnv env;
    env.d("dim", dim);
    const auto kernel_static_shapes =
        format(kernel_static_shapes_template, env);
    parseIR(kernel_static_shapes, &*graph);
    TensorExprKernel k(graph);

    auto a = at::rand({dim}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto b = at::rand({dim}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    std::vector<at::Tensor> inputs = {a, b};

    for (auto _ : state) {
      std::vector<IValue> stack = at::fmap<at::IValue>(inputs);
      k.run(stack);
    }
  }

  void GraphWithSymbolicShapes(benchmark::State& state) {
    auto dim = state.range(0);
    auto graph = std::make_shared<Graph>();
    parseIR(kernel_symbolic_shapes, &*graph);

    std::vector<torch::jit::StrideInput> input_desc = {
        torch::jit::StrideInput::TENSOR_CONT};
    std::unordered_map<
        const torch::jit::Value*,
        std::vector<torch::jit::StrideInput>>
        symbolic_strides;
    symbolic_strides[graph->inputs().at(0)] = input_desc;
    symbolic_strides[graph->inputs().at(1)] = input_desc;
    symbolic_strides[graph->outputs().at(0)] = input_desc;
    std::vector<int64_t> symbolic_shape_inputs = {-2};
    TensorExprKernel k(
        graph, {}, symbolic_shape_inputs, false, symbolic_strides);

    auto a = at::rand({dim}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto b = at::rand({dim}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    std::vector<at::Tensor> inputs = {a, b};

    for (auto _ : state) {
      std::vector<IValue> stack = at::fmap<at::IValue>(inputs);
      stack.push_back(dim);
      k.run(stack);
    }
  }
};

BENCHMARK_DEFINE_F(KernelBench, Eager)(benchmark::State& state) {
  Eager(state);
}

BENCHMARK_DEFINE_F(KernelBench, StaticShapes)(benchmark::State& state) {
  GraphWithStaticShapes(state);
}
BENCHMARK_DEFINE_F(KernelBench, SymbolicShapes)(benchmark::State& state) {
  GraphWithSymbolicShapes(state);
}

BENCHMARK_REGISTER_F(KernelBench, Eager)->Range(32, 2048);
BENCHMARK_REGISTER_F(KernelBench, StaticShapes)->Range(32, 2048);
BENCHMARK_REGISTER_F(KernelBench, SymbolicShapes)->Range(32, 2048);
