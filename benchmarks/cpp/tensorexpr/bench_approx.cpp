#include <benchmark/benchmark.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/torch.h>

using namespace torch::jit::tensorexpr;

static void log_sleef(benchmark::State& state) {
  KernelScope ks;
  auto N = VarHandle("N", kInt);
  Placeholder A("A", kFloat, {N});
  torch::jit::tensorexpr::Tensor* B =
      Compute("B", {N}, [&](const VarHandle& i) {
        return log(A.load(i));
      });
  LoopNest ln({B});
  ln.prepareForCodegen();
  ln.vectorizeInnerLoops();
  Stmt* s = ln.root_stmt();
  s = torch::jit::tensorexpr::IRSimplifier::simplify(s);
  std::vector<CodeGen::BufferArg> args;
  args.emplace_back(B);
  args.emplace_back(A);
  args.emplace_back(N);
  LLVMCodeGen cg(s, args);
  at::Tensor A_t = torch::abs(torch::randn({state.range(0)}));
  at::Tensor B_t = torch::randn({state.range(0)});
  auto B_ref = at::log(A_t);
  cg.call({B_t.data_ptr<float>(), A_t.data_ptr<float>(), state.range(0)});
  assert(at::allclose(B_t, B_ref));
  for (auto _ : state) {
    cg.call({B_t.data_ptr<float>(), A_t.data_ptr<float>(), state.range(0)});
  }
  state.counters["log/s"] = benchmark::Counter(
      uint64_t(state.range(0) * state.iterations()), benchmark::Counter::kIsRate);
}

static void log_fast(benchmark::State& state) {
  KernelScope ks;
  auto N = VarHandle("N", kInt);
  Placeholder A("A", kFloat, {N});
  torch::jit::tensorexpr::Tensor* B =
      Compute("B", {N}, [&](const VarHandle& i) {
        return fast_log(A.load(i));
      });
  LoopNest ln({B});
  ln.prepareForCodegen();
  ln.vectorizeInnerLoops();
  Stmt* s = ln.root_stmt();
  s = torch::jit::tensorexpr::IRSimplifier::simplify(s);
  std::vector<CodeGen::BufferArg> args;
  args.emplace_back(B);
  args.emplace_back(A);
  args.emplace_back(N);
  LLVMCodeGen cg(s, args);
  at::Tensor A_t = torch::abs(torch::randn({state.range(0)}));
  at::Tensor B_t = torch::randn({state.range(0)});
  auto B_ref = at::log(A_t);
  cg.call({B_t.data_ptr<float>(), A_t.data_ptr<float>(), state.range(0)});
  assert(at::allclose(B_t, B_ref));
  for (auto _ : state) {
    cg.call({B_t.data_ptr<float>(), A_t.data_ptr<float>(), state.range(0)});
  }
  state.counters["log/s"] = benchmark::Counter(
      uint64_t(state.range(0) * state.iterations()), benchmark::Counter::kIsRate);
}

static void log_aten(benchmark::State& state) {
  at::Tensor A_t = torch::abs(torch::randn({state.range(0)}));
  at::Tensor B_t = torch::randn({state.range(0)});
  for (auto _ : state) {
    at::native::log_out(B_t, A_t);
  }
  state.counters["log/s"] = benchmark::Counter(
      uint64_t(state.range(0) * state.iterations()), benchmark::Counter::kIsRate);
}

static void logit_fast(benchmark::State& state) {
  KernelScope ks;
  auto N = VarHandle("N", kInt);
  Placeholder A("A", kFloat, {N});
  torch::jit::tensorexpr::Tensor* B =
      Compute("B", {N}, [&](const VarHandle& i) {
          auto A_elem = A.load(i);
          return fast_log(A_elem / (FloatImm::make(1.0f) - A_elem));
      });
  LoopNest ln({B});
  ln.prepareForCodegen();
  ln.vectorizeInnerLoops();
  Stmt* s = ln.root_stmt();
  s = torch::jit::tensorexpr::IRSimplifier::simplify(s);
  std::vector<CodeGen::BufferArg> args;
  args.emplace_back(B);
  args.emplace_back(A);
  args.emplace_back(N);
  LLVMCodeGen cg(s, args);
  at::Tensor A_t = torch::abs(torch::randn({state.range(0)}));
  at::Tensor B_t = torch::randn({state.range(0)});
  auto B_ref = at::logit(A_t);
  cg.call({B_t.data_ptr<float>(), A_t.data_ptr<float>(), state.range(0)});
  assert(at::allclose(B_t, B_ref));
  for (auto _ : state) {
    cg.call({B_t.data_ptr<float>(), A_t.data_ptr<float>(), state.range(0)});
  }
  state.counters["logit/s"] = benchmark::Counter(
      uint64_t(state.range(0) * state.iterations()), benchmark::Counter::kIsRate);
}

static void logit_aten(benchmark::State& state) {
  at::Tensor A_t = torch::abs(torch::randn({state.range(0)}));
  at::Tensor B_t = torch::randn({state.range(0)});
  for (auto _ : state) {
    at::native::logit_out(B_t, A_t);
  }
  state.counters["logit/s"] = benchmark::Counter(
      uint64_t(state.range(0) * state.iterations()), benchmark::Counter::kIsRate);
}

BENCHMARK(log_sleef)
  ->Args({2<<5})
  ->Args({2<<8})
  ->Args({2<<12})
  ->Args({2<<14});
BENCHMARK(log_fast)
  ->Args({2<<5})
  ->Args({2<<8})
  ->Args({2<<12})
  ->Args({2<<14});
BENCHMARK(log_aten)
  ->Args({2<<5})
  ->Args({2<<8})
  ->Args({2<<12})
  ->Args({2<<14});
BENCHMARK(logit_fast)
  ->Args({2<<5})
  ->Args({2<<8})
  ->Args({2<<12})
  ->Args({2<<14});
BENCHMARK(logit_aten)
  ->Args({2<<5})
  ->Args({2<<8})
  ->Args({2<<12})
  ->Args({2<<14});
