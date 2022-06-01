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
} // namespace

BENCHMARK_DEFINE_F(Gemm, Torch)(benchmark::State& state) {
  for (auto _ : state) {
    torch::mm_out(C, A, B);
  }
}

BENCHMARK_DEFINE_F(Gemm, TensorExprNoopt)(benchmark::State& state) {
  te::BufHandle AP("A", {M, K}, te::kFloat);
  te::BufHandle BP("B", {K, N}, te::kFloat);
  te::Tensor CT = te::Reduce(
      "gemm",
      {M, N},
      te::Sum(),
      [&](const te::ExprHandle& m,
          const te::ExprHandle& n,
          const te::ExprHandle& k) { return AP.load(m, k) * BP.load(k, n); },
      {K});
  te::LoopNest loop({CT});
  loop.prepareForCodegen();
  te::StmtPtr s = loop.root_stmt();
  s = te::IRSimplifier::simplify(s);
  auto cg = CreateCodeGen("llvm_codegen", s, {AP, BP, CT});

  for (auto _ : state) {
    cg->call({A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>()});
  }
}

BENCHMARK_DEFINE_F(Gemm, TensorExprTile32x32)(benchmark::State& state) {
  te::BufHandle AP("A", {M, K}, te::kFloat);
  te::BufHandle BP("B", {K, N}, te::kFloat);
  te::Tensor CT = te::Reduce(
      "gemm",
      {M, N},
      te::Sum(),
      [&](const te::ExprHandle& m,
          const te::ExprHandle& n,
          const te::ExprHandle& k) { return AP.load(m, k) * BP.load(k, n); },
      {K});
  te::LoopNest loop({CT});

  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::ForPtr m = loops[0];
    loop.splitWithMask(m, 32);
  }
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::ForPtr n = loops[2];
    loop.splitWithMask(n, 32);
  }
  // mo, mi, no, ni, k ->
  // mo, no, mi, ni, k
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::ForPtr mi = loops[1];
    te::ForPtr no = loops[2];
    loop.reorderAxis(mi, no);
  }
  // mo, no, mi, ni, k ->
  // mo, no, mi, k, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::ForPtr ni = loops[3];
    te::ForPtr k = loops[4];
    loop.reorderAxis(ni, k);
  }
  // mo, no, mi, k, ni ->
  // mo, no, k, mi, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::ForPtr mi = loops[2];
    te::ForPtr k = loops[3];
    loop.reorderAxis(mi, k);
  }

  loop.prepareForCodegen();
  te::StmtPtr s = loop.root_stmt();
  s = te::IRSimplifier::simplify(s);
  auto cg = CreateCodeGen("llvm_codegen", s, {AP, BP, CT});

  for (auto _ : state) {
    cg->call({A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>()});
  }
}

BENCHMARK_DEFINE_F(Gemm, TensorExprTile4x16)(benchmark::State& state) {
  te::BufHandle AP("A", {M, K}, te::kFloat);
  te::BufHandle BP("B", {K, N}, te::kFloat);
  te::Tensor CT = te::Reduce(
      "gemm",
      {M, N},
      te::Sum(),
      [&](const te::ExprHandle& m,
          const te::ExprHandle& n,
          const te::ExprHandle& k) { return AP.load(m, k) * BP.load(k, n); },
      {K});
  te::LoopNest loop({CT});

  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::ForPtr m = loops[0];
    loop.splitWithMask(m, 4);
  }
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::ForPtr n = loops[2];
    loop.splitWithMask(n, 16);
  }
  // mo, mi, no, ni, k ->
  // mo, no, mi, ni, k
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::ForPtr mi = loops[1];
    te::ForPtr no = loops[2];
    loop.reorderAxis(mi, no);
  }
  // mo, no, mi, ni, k ->
  // mo, no, mi, k, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::ForPtr ni = loops[3];
    te::ForPtr k = loops[4];
    loop.reorderAxis(ni, k);
  }
  // mo, no, mi, k, ni ->
  // mo, no, k, mi, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::ForPtr mi = loops[2];
    te::ForPtr k = loops[3];
    loop.reorderAxis(mi, k);
  }

  loop.prepareForCodegen();
  te::StmtPtr s = loop.root_stmt();
  s = te::IRSimplifier::simplify(s);
  auto cg = CreateCodeGen("llvm_codegen", s, {AP, BP, CT});

  for (auto _ : state) {
    cg->call({A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>()});
  }
}

BENCHMARK_DEFINE_F(Gemm, TensorExprTile4x16VecUnroll)(benchmark::State& state) {
  te::BufHandle AP("A", {M, K}, te::kFloat);
  te::BufHandle BP("B", {K, N}, te::kFloat);
  te::Tensor CT = te::Reduce(
      "gemm",
      {M, N},
      te::Sum(),
      [&](const te::ExprHandle& m,
          const te::ExprHandle& n,
          const te::ExprHandle& k) { return AP.load(m, k) * BP.load(k, n); },
      {K});
  te::LoopNest loop({CT});

  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::ForPtr m = loops[0];
    loop.splitWithMask(m, 4);
  }
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::ForPtr n = loops[2];
    loop.splitWithMask(n, 16);
  }
  // mo, mi, no, ni, k ->
  // mo, no, mi, ni, k
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::ForPtr mi = loops[1];
    te::ForPtr no = loops[2];
    loop.reorderAxis(mi, no);
  }
  // mo, no, mi, ni, k ->
  // mo, no, mi, k, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::ForPtr ni = loops[3];
    te::ForPtr k = loops[4];
    loop.reorderAxis(ni, k);
  }
  // mo, no, mi, k, ni ->
  // mo, no, k, mi, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::ForPtr mi = loops[2];
    te::ForPtr k = loops[3];
    loop.reorderAxis(mi, k);
  }
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::ForPtr mi = loops[3];
    te::ForPtr ni = loops[4];
    te::StmtPtr unrolled;
    loop.vectorize(ni);
    loop.fullUnroll(mi, &unrolled);
  }

  loop.prepareForCodegen();
  te::StmtPtr s = loop.root_stmt();
  s = te::IRSimplifier::simplify(s);
  auto cg = CreateCodeGen("llvm_codegen", s, {AP, BP, CT});

  for (auto _ : state) {
    cg->call({A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>()});
  }
}

BENCHMARK_DEFINE_F(Gemm, TensorExprTile4x16Cache)(benchmark::State& state) {
  te::BufHandle AP("A", {M, K}, te::kFloat);
  te::BufHandle BP("B", {K, N}, te::kFloat);
  te::Tensor CT = te::Reduce(
      "gemm",
      {M, N},
      te::Sum(),
      [&](const te::ExprHandle& m,
          const te::ExprHandle& n,
          const te::ExprHandle& k) { return AP.load(m, k) * BP.load(k, n); },
      {K});
  te::LoopNest loop({CT});

  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::ForPtr m = loops[0];
    loop.splitWithMask(m, 4);
  }
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::ForPtr n = loops[2];
    loop.splitWithMask(n, 16);
  }
  // mo, mi, no, ni, k ->
  // mo, no, mi, ni, k
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::ForPtr mi = loops[1];
    te::ForPtr no = loops[2];
    loop.reorderAxis(mi, no);
  }
  // mo, no, mi, ni, k ->
  // mo, no, mi, k, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::ForPtr ni = loops[3];
    te::ForPtr k = loops[4];
    loop.reorderAxis(ni, k);
  }
  // mo, no, mi, k, ni ->
  // mo, no, k, mi, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    te::ForPtr mi = loops[2];
    te::ForPtr k = loops[3];
    loop.reorderAxis(mi, k);
  }
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    loop.cacheAccesses(CT.buf(), "C_regs", loops[2]);
  }

  loop.prepareForCodegen();
  te::StmtPtr s = loop.root_stmt();
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
