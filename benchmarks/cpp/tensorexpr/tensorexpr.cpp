#include <benchmark/benchmark.h>
#include "torch/torch.h"
#include "torch/csrc/jit/tensorexpr/ir_simplifier.h"
#include "torch/csrc/jit/tensorexpr/loopnest.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

namespace te = torch::jit::tensorexpr;

class Gemm : public benchmark::Fixture {
 public:
  void SetUp(const benchmark::State& state) {
    M = state.range(0);
    N = state.range(1);
    K = state.range(2);
    A = torch::randn({M, K});
    B = torch::randn({K, N});
    C = torch::mm(A, B);
  }

  void TearDown(benchmark::State& state) {
    state.counters["GFLOPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * 2 * M * N * K,
                         benchmark::Counter::kIsRate);
  }

  int M;
  int N;
  int K;
  at::Tensor A;
  at::Tensor B;
  at::Tensor C;
};

BENCHMARK_DEFINE_F(Gemm, Torch)(benchmark::State& state) {
  for (auto _ : state) {
    torch::mm_out(C, A, B);
  }
}

BENCHMARK_DEFINE_F(Gemm, TensorExprNoopt)(benchmark::State& state) {
  te::KernelScope ks;

  te::Placeholder AP(te::BufHandle("A", {M, K}, te::kFloat));
  te::Placeholder BP(te::BufHandle("B", {K, N}, te::kFloat));
  te::Tensor* CT = te::Reduce(
      "gemm",
      {{M, "M"}, {N, "N"}},
      te::Sum(),
      [&](const te::ExprHandle& m, const te::ExprHandle& n, const te::ExprHandle& k) {
        return AP.load(m, k) * BP.load(k, n);
      },
      {{K, "K"}});
  te::LoopNest loop({CT});
  loop.prepareForCodegen();
  te::Stmt* s = loop.root_stmt();
  s = te::IRSimplifier::simplify(s);
  auto cg = CreateCodeGen("llvm_codegen", s, {AP, BP, CT});

  for (auto _ : state) {
    cg->call({A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>()});
  }
}

BENCHMARK_REGISTER_F(Gemm, Torch)->Args({128, 128, 128});
BENCHMARK_REGISTER_F(Gemm, TensorExprNoopt)->Args({128, 128, 128});

BENCHMARK_MAIN();
