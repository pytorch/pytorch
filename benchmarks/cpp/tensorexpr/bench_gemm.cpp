#include <benchmark/benchmark.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/torch.h>

namespace te = torch::jit::tensorexpr;

namespace {
class Gemm : public benchmark::Fixture {
 public:
  void SetUp(const benchmark::State& state) override {
    M = state.range(0);
    N = state.range(1);
    K = state.range(2);
    A = torch::randn({M, K});
    B = torch::randn({K, N});
    C = torch::mm(A, B);
  }

  void TearDown(benchmark::State& state) override {
    state.counters["GFLOPS"] = benchmark::Counter(
        uint64_t(state.iterations()) * 2 * M * N * K,
        benchmark::Counter::kIsRate);
  }

  int M;
  int N;
  int K;
  at::Tensor A;
  at::Tensor B;
  at::Tensor C;
};
}

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
      [&](const te::ExprHandle& m,
          const te::ExprHandle& n,
          const te::ExprHandle& k) { return AP.load(m, k) * BP.load(k, n); },
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

BENCHMARK_DEFINE_F(Gemm, TensorExprTile32x32)(benchmark::State& state) {
  te::KernelScope ks;

  te::Placeholder AP(te::BufHandle("A", {M, K}, te::kFloat));
  te::Placeholder BP(te::BufHandle("B", {K, N}, te::kFloat));
  te::Tensor* CT = te::Reduce(
      "gemm",
      {{M, "M"}, {N, "N"}},
      te::Sum(),
      [&](const te::ExprHandle& m,
          const te::ExprHandle& n,
          const te::ExprHandle& k) { return AP.load(m, k) * BP.load(k, n); },
      {{K, "K"}});
  te::LoopNest loop({CT});

  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* m = loops[0];
    te::For* mo;
    te::For* mi;
    loop.splitWithMask(m, 32, &mo, &mi);
  }
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* n = loops[2];
    te::For* no;
    te::For* ni;
    loop.splitWithMask(n, 32, &no, &ni);
  }
  // mo, mi, no, ni, k ->
  // mo, no, mi, ni, k
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* mi = loops[1];
    te::For* no = loops[2];
    loop.reorderAxis(mi, no);
  }
  // mo, no, mi, ni, k ->
  // mo, no, mi, k, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* ni = loops[3];
    te::For* k = loops[4];
    loop.reorderAxis(ni, k);
  }
  // mo, no, mi, k, ni ->
  // mo, no, k, mi, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* mi = loops[2];
    te::For* k = loops[3];
    loop.reorderAxis(mi, k);
  }

  loop.prepareForCodegen();
  te::Stmt* s = loop.root_stmt();
  s = te::IRSimplifier::simplify(s);
  auto cg = CreateCodeGen("llvm_codegen", s, {AP, BP, CT});

  for (auto _ : state) {
    cg->call({A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>()});
  }
}

BENCHMARK_DEFINE_F(Gemm, TensorExprTile4x16)(benchmark::State& state) {
  te::KernelScope ks;

  te::Placeholder AP(te::BufHandle("A", {M, K}, te::kFloat));
  te::Placeholder BP(te::BufHandle("B", {K, N}, te::kFloat));
  te::Tensor* CT = te::Reduce(
      "gemm",
      {{M, "M"}, {N, "N"}},
      te::Sum(),
      [&](const te::ExprHandle& m,
          const te::ExprHandle& n,
          const te::ExprHandle& k) { return AP.load(m, k) * BP.load(k, n); },
      {{K, "K"}});
  te::LoopNest loop({CT});

  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* m = loops[0];
    te::For* mo;
    te::For* mi;
    loop.splitWithMask(m, 4, &mo, &mi);
  }
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* n = loops[2];
    te::For* no;
    te::For* ni;
    loop.splitWithMask(n, 16, &no, &ni);
  }
  // mo, mi, no, ni, k ->
  // mo, no, mi, ni, k
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* mi = loops[1];
    te::For* no = loops[2];
    loop.reorderAxis(mi, no);
  }
  // mo, no, mi, ni, k ->
  // mo, no, mi, k, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* ni = loops[3];
    te::For* k = loops[4];
    loop.reorderAxis(ni, k);
  }
  // mo, no, mi, k, ni ->
  // mo, no, k, mi, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* mi = loops[2];
    te::For* k = loops[3];
    loop.reorderAxis(mi, k);
  }

  loop.prepareForCodegen();
  te::Stmt* s = loop.root_stmt();
  s = te::IRSimplifier::simplify(s);
  auto cg = CreateCodeGen("llvm_codegen", s, {AP, BP, CT});

  for (auto _ : state) {
    cg->call({A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>()});
  }
}

BENCHMARK_DEFINE_F(Gemm, TensorExprTile4x16VecUnroll)(benchmark::State& state) {
  te::KernelScope ks;

  te::Placeholder AP(te::BufHandle("A", {M, K}, te::kFloat));
  te::Placeholder BP(te::BufHandle("B", {K, N}, te::kFloat));
  te::Tensor* CT = te::Reduce(
      "gemm",
      {{M, "M"}, {N, "N"}},
      te::Sum(),
      [&](const te::ExprHandle& m,
          const te::ExprHandle& n,
          const te::ExprHandle& k) { return AP.load(m, k) * BP.load(k, n); },
      {{K, "K"}});
  te::LoopNest loop({CT});

  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* m = loops[0];
    te::For* mo;
    te::For* mi;
    loop.splitWithMask(m, 4, &mo, &mi);
  }
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* n = loops[2];
    te::For* no;
    te::For* ni;
    loop.splitWithMask(n, 16, &no, &ni);
  }
  // mo, mi, no, ni, k ->
  // mo, no, mi, ni, k
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* mi = loops[1];
    te::For* no = loops[2];
    loop.reorderAxis(mi, no);
  }
  // mo, no, mi, ni, k ->
  // mo, no, mi, k, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* ni = loops[3];
    te::For* k = loops[4];
    loop.reorderAxis(ni, k);
  }
  // mo, no, mi, k, ni ->
  // mo, no, k, mi, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* mi = loops[2];
    te::For* k = loops[3];
    loop.reorderAxis(mi, k);
  }
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* mi = loops[3];
    te::For* ni = loops[4];
    te::Stmt* unrolled;
    loop.vectorize(ni);
    loop.unroll(mi, &unrolled);
  }

  loop.prepareForCodegen();
  te::Stmt* s = loop.root_stmt();
  s = te::IRSimplifier::simplify(s);
  auto cg = CreateCodeGen("llvm_codegen", s, {AP, BP, CT});

  for (auto _ : state) {
    cg->call({A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>()});
  }
}

BENCHMARK_DEFINE_F(Gemm, TensorExprTile4x16Cache)(benchmark::State& state) {
  te::KernelScope ks;

  te::Placeholder AP(te::BufHandle("A", {M, K}, te::kFloat));
  te::Placeholder BP(te::BufHandle("B", {K, N}, te::kFloat));
  te::Tensor* CT = te::Reduce(
      "gemm",
      {{M, "M"}, {N, "N"}},
      te::Sum(),
      [&](const te::ExprHandle& m,
          const te::ExprHandle& n,
          const te::ExprHandle& k) { return AP.load(m, k) * BP.load(k, n); },
      {{K, "K"}});
  te::LoopNest loop({CT});

  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* m = loops[0];
    te::For* mo;
    te::For* mi;
    loop.splitWithMask(m, 4, &mo, &mi);
  }
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* n = loops[2];
    te::For* no;
    te::For* ni;
    loop.splitWithMask(n, 16, &no, &ni);
  }
  // mo, mi, no, ni, k ->
  // mo, no, mi, ni, k
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* mi = loops[1];
    te::For* no = loops[2];
    loop.reorderAxis(mi, no);
  }
  // mo, no, mi, ni, k ->
  // mo, no, mi, k, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* ni = loops[3];
    te::For* k = loops[4];
    loop.reorderAxis(ni, k);
  }
  // mo, no, mi, k, ni ->
  // mo, no, k, mi, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::For* mi = loops[2];
    te::For* k = loops[3];
    loop.reorderAxis(mi, k);
  }
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    loop.cacheAccesses(CT->buf(), "C_regs", loops[2]);
  }

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
BENCHMARK_REGISTER_F(Gemm, TensorExprTile32x32)->Args({128, 128, 128});
BENCHMARK_REGISTER_F(Gemm, TensorExprTile4x16)->Args({128, 128, 128});
BENCHMARK_REGISTER_F(Gemm, TensorExprTile4x16VecUnroll)->Args({128, 128, 128});
BENCHMARK_REGISTER_F(Gemm, TensorExprTile4x16Cache)->Args({128, 128, 128});
