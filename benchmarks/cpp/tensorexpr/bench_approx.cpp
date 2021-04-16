#include <benchmark/benchmark.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/torch.h>
#include "caffe2/operators/tanh_op.h"
#include "caffe2/operators/logit_op.h"

using namespace torch::jit;
using namespace torch::jit::tensorexpr;

void vectorize(tensorexpr::LoopNest* ln, tensorexpr::Tensor* target, int width) {
  auto loops = ln->getLoopStmtsFor(target);
  For *outer, *inner, *tail;
  ln->splitWithTail(loops[0], width, &outer, &inner, &tail);
  ln->vectorize(inner);
}

void optimizePointwise(tensorexpr::LoopNest* ln, tensorexpr::Tensor* target) {
  std::vector<For*> loops = ln->getLoopStmtsFor(target);
  For *outer, *inner, *tail;
  ln->splitWithTail(loops[0], 16 * 8, &outer, &inner, &tail);
  ln->vectorize(inner);
  ln->splitWithTail(outer, 8, &outer, &inner, &tail);
  Stmt* unrolled;
  LoopNest::unroll(inner, &unrolled);
}

static void relu_nnc(benchmark::State& state) {
  KernelScope ks;
  auto N = VarHandle("N", kInt);
  Placeholder A("A", kFloat, {N});
  auto clamp = 0;
  torch::jit::tensorexpr::Tensor* B = Compute("B", {N}, [&](const VarHandle& i){
    auto A_elem = [&]() {
      auto elem = A.load(i);
      auto min = FloatImm::make(clamp);
      return CompareSelect::make(elem, min, min, elem, kLT);
    }();
    return A_elem;
  });
  LoopNest ln({B});
  optimizePointwise(&ln, B);
  ln.prepareForCodegen();
  Stmt* s = ln.root_stmt();
  s = torch::jit::tensorexpr::IRSimplifier::simplify(s);
  std::vector<CodeGen::BufferArg> args;
  args.emplace_back(B);
  args.emplace_back(A);
  args.emplace_back(N);
  LLVMCodeGen cg(s, args);
  at::Tensor A_t = torch::randn({state.range(0)});
  at::Tensor B_t = torch::randn(state.range(0));
  auto B_ref = at::relu(A_t);
  cg.call({B_t.data_ptr<float>(), A_t.data_ptr<float>(), state.range(0)});
  TORCH_CHECK(at::allclose(B_t, B_ref));
  for (auto _ : state){
    cg.call({B_t.data_ptr<float>(), A_t.data_ptr<float>(), state.range(0)});
  }
  state.counters["log/s"] = benchmark::Counter(
    uint64_t(state.range(0) * state.iterations()), benchmark::Counter::kIsRate);
}

static void log_nnc_sleef(benchmark::State& state) {
  KernelScope ks;
  auto N = VarHandle("N", kInt);
  Placeholder A("A", kFloat, {N});
  torch::jit::tensorexpr::Tensor* B =
      Compute("B", {N}, [&](const VarHandle& i) {
        return log(A.load(i));
      });
  LoopNest ln({B});
  ln.prepareForCodegen();
  vectorize(&ln, B, 8);
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
  TORCH_CHECK(at::allclose(B_t, B_ref));
  for (auto _ : state) {
    cg.call({B_t.data_ptr<float>(), A_t.data_ptr<float>(), state.range(0)});
  }
  state.counters["log/s"] = benchmark::Counter(
      uint64_t(state.range(0) * state.iterations()), benchmark::Counter::kIsRate);
}

static void log_nnc_fast(benchmark::State& state) {
  KernelScope ks;
  auto N = VarHandle("N", kInt);
  Placeholder A("A", kFloat, {N});
  torch::jit::tensorexpr::Tensor* B =
      Compute("B", {N}, [&](const VarHandle& i) {
        return fast_log(A.load(i));
      });
  LoopNest ln({B});
  optimizePointwise(&ln, B);
  ln.prepareForCodegen();
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
  TORCH_CHECK(at::allclose(B_t, B_ref));
  for (auto _ : state) {
    cg.call({B_t.data_ptr<float>(), A_t.data_ptr<float>(), state.range(0)});
  }
  state.counters["log/s"] = benchmark::Counter(
      uint64_t(state.range(0) * state.iterations()), benchmark::Counter::kIsRate);
}

static void log_nnc_vml(benchmark::State& state) {
  KernelScope ks;
  auto N = VarHandle("N", kInt);
  Placeholder A("A", kFloat, {N});
  torch::jit::tensorexpr::Tensor* B =
      Compute("B", {N}, [&](const VarHandle& i) {
        return log_vml(A.load(i));
      });
  LoopNest ln({B});
  vectorize(&ln, B, 8);
  ln.prepareForCodegen();
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
  TORCH_CHECK(at::allclose(B_t, B_ref));
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
    at::log_out(B_t, A_t);
  }
  state.counters["log/s"] = benchmark::Counter(
      uint64_t(state.range(0) * state.iterations()), benchmark::Counter::kIsRate);
}

static void logit_nnc_sleef(benchmark::State& state) {
  KernelScope ks;
  auto N = VarHandle("N", kInt);
  Placeholder A("A", kFloat, {N});
  auto clamp = 1e-6f;
  tensorexpr::Tensor* B = Compute("B", {N}, [&](const VarHandle& i) {
    auto A_elem = [&]() {
      auto elem = A.load(i);
      auto min = FloatImm::make(clamp);
      auto max = FloatImm::make(1.0f - clamp);
      elem = CompareSelect::make(elem, min, min, elem, kLT);
      return CompareSelect::make(elem, max, max, elem, kGT);
    }();
    return log(A_elem / (FloatImm::make(1.0f) - A_elem));
  });
  LoopNest ln({B});
  ln.prepareForCodegen();
  optimizePointwise(&ln, B);
  Stmt* s = ln.root_stmt();
  s = torch::jit::tensorexpr::IRSimplifier::simplify(s);
  std::vector<CodeGen::BufferArg> args;
  args.emplace_back(B);
  args.emplace_back(A);
  args.emplace_back(N);
  LLVMCodeGen cg(s, args);
  at::Tensor A_t = torch::abs(torch::randn({state.range(0)}));
  at::Tensor B_t = torch::randn({state.range(0)});
  auto B_ref = at::logit(A_t, clamp);
  cg.call({B_t.data_ptr<float>(), A_t.data_ptr<float>(), state.range(0)});
  TORCH_CHECK(at::allclose(at::nan_to_num(B_t), at::nan_to_num(B_ref)));
  for (auto _ : state) {
    cg.call({B_t.data_ptr<float>(), A_t.data_ptr<float>(), state.range(0)});
  }
  state.counters["logit/s"] = benchmark::Counter(
      uint64_t(state.range(0) * state.iterations()), benchmark::Counter::kIsRate);
}

static void logit_nnc_fast(benchmark::State& state) {
  KernelScope ks;
  auto N = VarHandle("N", kInt);
  Placeholder A("A", kFloat, {N});
  auto clamp = 1e-6f;
  tensorexpr::Tensor* B = Compute("B", {N}, [&](const VarHandle& i) {
    auto A_elem = [&]() {
      auto elem = A.load(i);
      auto min = FloatImm::make(clamp);
      auto max = FloatImm::make(1.0f - clamp);
      elem = CompareSelect::make(elem, min, min, elem, kLT);
      return CompareSelect::make(elem, max, max, elem, kGT);
    }();
    return fast_log(A_elem / (FloatImm::make(1.0f) - A_elem));
  });
  LoopNest ln({B});
  ln.prepareForCodegen();
  optimizePointwise(&ln, B);
  Stmt* s = ln.root_stmt();
  s = torch::jit::tensorexpr::IRSimplifier::simplify(s);
  std::vector<CodeGen::BufferArg> args;
  args.emplace_back(B);
  args.emplace_back(A);
  args.emplace_back(N);
  LLVMCodeGen cg(s, args);
  at::Tensor A_t = torch::abs(torch::randn({state.range(0)}));
  at::Tensor B_t = torch::randn({state.range(0)});
  auto B_ref = at::logit(A_t, clamp);
  cg.call({B_t.data_ptr<float>(), A_t.data_ptr<float>(), state.range(0)});
  TORCH_CHECK(at::allclose(at::nan_to_num(B_t), at::nan_to_num(B_ref)));
  for (auto _ : state) {
    cg.call({B_t.data_ptr<float>(), A_t.data_ptr<float>(), state.range(0)});
  }
  state.counters["logit/s"] = benchmark::Counter(
      uint64_t(state.range(0) * state.iterations()), benchmark::Counter::kIsRate);
}

static void logit_nnc_vml(benchmark::State& state) {
  KernelScope ks;
  auto N = VarHandle("N", kInt);
  Placeholder A("A", kFloat, {N});
  auto clamp = 1e-6f;
  tensorexpr::Tensor* B = Compute("B", {N}, [&](const VarHandle& i) {
    auto A_elem = [&]() {
      auto elem = A.load(i);
      auto min = FloatImm::make(clamp);
      auto max = FloatImm::make(1.0f - clamp);
      elem = CompareSelect::make(elem, min, min, elem, kLT);
      return CompareSelect::make(elem, max, max, elem, kGT);
    }();
    return log_vml(A_elem / (FloatImm::make(1.0f) - A_elem));
  });
  LoopNest ln({B});
  ln.prepareForCodegen();
  vectorize(&ln, B, 16);
  Stmt* s = ln.root_stmt();
  s = torch::jit::tensorexpr::IRSimplifier::simplify(s);
  std::vector<CodeGen::BufferArg> args;
  args.emplace_back(B);
  args.emplace_back(A);
  args.emplace_back(N);
  LLVMCodeGen cg(s, args);
  at::Tensor A_t = torch::abs(torch::randn({state.range(0)}));
  at::Tensor B_t = torch::randn({state.range(0)});
  auto B_ref = at::logit(A_t, clamp);
  cg.call({B_t.data_ptr<float>(), A_t.data_ptr<float>(), state.range(0)});
  TORCH_CHECK(at::allclose(at::nan_to_num(B_t), at::nan_to_num(B_ref)));
  for (auto _ : state) {
    cg.call({B_t.data_ptr<float>(), A_t.data_ptr<float>(), state.range(0)});
  }
  state.counters["logit/s"] = benchmark::Counter(
      uint64_t(state.range(0) * state.iterations()), benchmark::Counter::kIsRate);
}

static void logit_aten(benchmark::State& state) {
  at::Tensor A_t = torch::abs(torch::randn({state.range(0)}));
  at::Tensor B_t = torch::randn({state.range(0)});
  auto clamp = 1e-6f;
  for (auto _ : state) {
    at::native::logit_out(A_t, clamp, B_t);
  }
  state.counters["logit/s"] = benchmark::Counter(
      uint64_t(state.range(0) * state.iterations()), benchmark::Counter::kIsRate);
}

template <typename T>
void logit_caffe2_impl(int size, const T* X, T* Y, float eps_ = 1e-6f) {
  using namespace caffe2;
  ConstEigenVectorMap<T> X_vec(X, size);
  EigenVectorMap<T> Y_vec(Y, size);
  Y_vec = X_vec.array().min(static_cast<T>(1.0f - eps_));
  Y_vec = Y_vec.array().max(eps_);
  Y_vec = (Y_vec.array() / (T(1) - Y_vec.array())).log();
}

static void logit_caffe2(benchmark::State& state) {
  at::Tensor A_t = torch::abs(torch::randn({state.range(0)}));
  at::Tensor B_t = torch::randn({state.range(0)});
  at::Tensor B_ref = torch::randn({state.range(0)});
  auto N = state.range(0);
  auto X = A_t.data_ptr<float>();
  auto Y = B_t.data_ptr<float>();
  auto clamp = 1e-6f;
  at::native::logit_out(A_t, clamp, B_ref);
  logit_caffe2_impl(N, X, Y, clamp);
  TORCH_CHECK(at::allclose(at::nan_to_num(B_t), at::nan_to_num(B_ref)));

  for (auto _ : state) {
    logit_caffe2_impl(N, X, Y, clamp);
  }

  state.counters["logit/s"] = benchmark::Counter(
      uint64_t(state.range(0) * state.iterations()), benchmark::Counter::kIsRate);
}

static void tanh_nnc_fast(benchmark::State& state) {
  KernelScope ks;
  auto N = VarHandle("N", kInt);
  Placeholder A("A", kFloat, {N});
  torch::jit::tensorexpr::Tensor* B =
      Compute("B", {N}, [&](const VarHandle& i) {
        return fast_tanh(A.load(i));
      });
  LoopNest ln({B});
  optimizePointwise(&ln, B);
  ln.prepareForCodegen();
  Stmt* s = ln.root_stmt();
  s = torch::jit::tensorexpr::IRSimplifier::simplify(s);
  std::vector<CodeGen::BufferArg> args;
  args.emplace_back(B);
  args.emplace_back(A);
  args.emplace_back(N);
  LLVMCodeGen cg(s, args);
  at::Tensor A_t = torch::abs(torch::randn({state.range(0)}));
  at::Tensor B_t = torch::randn({state.range(0)});
  auto B_ref = at::tanh(A_t);
  cg.call({B_t.data_ptr<float>(), A_t.data_ptr<float>(), state.range(0)});
  TORCH_CHECK(at::allclose(B_t, B_ref, 1e-3f, 1e-6f));
  for (auto _ : state) {
    cg.call({B_t.data_ptr<float>(), A_t.data_ptr<float>(), state.range(0)});
  }
  state.counters["tanh/s"] = benchmark::Counter(
      uint64_t(state.range(0) * state.iterations()), benchmark::Counter::kIsRate);
}

static void tanh_aten(benchmark::State& state) {
  at::Tensor A_t = torch::abs(torch::randn({state.range(0)}));
  at::Tensor B_t = torch::randn({state.range(0)});
  for (auto _ : state) {
    at::tanh_out(A_t, B_t);
  }
  state.counters["tanh/s"] = benchmark::Counter(
      uint64_t(state.range(0) * state.iterations()), benchmark::Counter::kIsRate);
}

static void tanh_caffe2(benchmark::State& state) {
  at::Tensor A_t = torch::abs(torch::randn({state.range(0)}));
  at::Tensor B_t = torch::randn({state.range(0)});
  at::Tensor B_ref = torch::randn({state.range(0)});

  auto N = state.range(0);
  auto X = A_t.data_ptr<float>();
  auto Y = B_t.data_ptr<float>();
  caffe2::CPUContext c;
  auto tanh = caffe2::TanhFunctor<caffe2::CPUContext>();
  at::tanh_out(A_t, B_ref);
  tanh(N, X, Y, &c);
  TORCH_CHECK(at::native::allclose(B_t, B_ref, 1e-3f, 1e-6f));

  for (auto _ : state) {
    tanh(N, X, Y, &c);
  }
  state.counters["tanh/s"] = benchmark::Counter(
      uint64_t(state.range(0) * state.iterations()), benchmark::Counter::kIsRate);
}

BENCHMARK(relu_nnc)
  ->Args({2<<5})
  ->Args({2<<8})
  ->Args({2<<12})
  ->Args({2<<14});
BENCHMARK(log_nnc_sleef)
  ->Args({2<<5})
  ->Args({2<<8})
  ->Args({2<<12})
  ->Args({2<<14});
BENCHMARK(log_nnc_fast)
  ->Args({2<<5})
  ->Args({2<<8})
  ->Args({2<<12})
  ->Args({2<<14});
BENCHMARK(log_nnc_vml)
  ->Args({2<<5})
  ->Args({2<<8})
  ->Args({2<<12})
  ->Args({2<<14});
BENCHMARK(log_aten)
  ->Args({2<<5})
  ->Args({2<<8})
  ->Args({2<<12})
  ->Args({2<<14});
BENCHMARK(logit_nnc_sleef)
  ->Args({2<<5})
  ->Args({2<<8})
  ->Args({2<<12})
  ->Args({2<<14});
BENCHMARK(logit_nnc_fast)
  ->Args({2<<5})
  ->Args({2<<8})
  ->Args({2<<12})
  ->Args({2<<14});
BENCHMARK(logit_nnc_vml)
  ->Args({2<<5})
  ->Args({2<<8})
  ->Args({2<<12})
  ->Args({2<<14});
BENCHMARK(logit_aten)
  ->Args({2<<5})
  ->Args({2<<8})
  ->Args({2<<12})
  ->Args({2<<14});
BENCHMARK(logit_caffe2)
  ->Args({2<<5})
  ->Args({2<<8})
  ->Args({2<<12})
  ->Args({2<<14});
BENCHMARK(tanh_nnc_fast)
  ->Args({2<<5})
  ->Args({2<<8})
  ->Args({2<<12})
  ->Args({2<<14});
BENCHMARK(tanh_aten)
  ->Args({2<<5})
  ->Args({2<<8})
  ->Args({2<<12})
  ->Args({2<<14});
BENCHMARK(tanh_caffe2)
  ->Args({2<<5})
  ->Args({2<<8})
  ->Args({2<<12})
  ->Args({2<<14});
