#include <benchmark/benchmark.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/torch.h>

using namespace torch::jit::tensorexpr;

namespace {
class BatchNorm : public benchmark::Fixture {
 public:
  void SetUp(const benchmark::State& state) override {
    N_ = state.range(0);
    C_ = state.range(1);
    H_ = state.range(2);
    W_ = state.range(3);
    input_ = torch::ones({N_, C_, H_, W_});
    weight_ = torch::ones({C_});
    bias_ = torch::ones({C_});
    mean_ = torch::ones({C_}) * 0.5f;
    var_ = torch::ones({C_}) * 0.1f;
    ref_ = at::batch_norm(
        input_,
        weight_,
        bias_,
        mean_,
        var_,
        training_,
        momentum_,
        eps_,
        cudnn_enabled_);
    output_ = at::empty_like(ref_);
  }

  void TearDown(benchmark::State& state) override {
    TORCH_CHECK(at::allclose(ref_, output_));
    state.counters["GB/s"] = benchmark::Counter(
        uint64_t(state.iterations()) * (input_.nbytes() + ref_.nbytes()),
        benchmark::Counter::kIsRate);
  }

  int N_;
  int C_;
  int H_;
  int W_;
  at::Tensor input_;
  at::Tensor weight_;
  at::Tensor bias_;
  at::Tensor mean_;
  at::Tensor var_;
  at::Tensor output_;
  at::Tensor ref_;
  bool training_{false};
  float momentum_{0.1};
  float eps_{1.0e-5f};
  bool cudnn_enabled_{false};
};
} // namespace

BENCHMARK_DEFINE_F(BatchNorm, ATen)(benchmark::State& state) {
  for (auto _ : state) {
    output_ = at::batch_norm(
        input_,
        weight_,
        bias_,
        mean_,
        var_,
        training_,
        momentum_,
        eps_,
        cudnn_enabled_);
  }
}

BENCHMARK_DEFINE_F(BatchNorm, NNC)(benchmark::State& state) {
  BufHandle input("input", {N_, C_, H_, W_}, kFloat);
  BufHandle weight("weight", {C_}, kFloat);
  BufHandle bias("bias", {C_}, kFloat);
  BufHandle mean("mean", {C_}, kFloat);
  BufHandle var("var", {C_}, kFloat);
  VarHandle eps("eps", kFloat);

  using axis = const VarHandle&;
  Tensor output =
      Compute("output", {N_, C_, H_, W_}, [&](axis n, axis c, axis h, axis w) {
        // Compute affine terms.
        auto inv_var = FloatImm::make(1.0f) / sqrt(var.load(c) + eps);
        auto weight_v = weight.load(c);
        auto bias_v = bias.load(c);
        auto alpha = inv_var * weight_v;
        auto beta = bias_v - mean.load(c) * alpha;

        return input.load(n, c, h, w) * alpha + beta;
      });
  LoopNest nest({output});
  auto loops = nest.getLoopStmtsFor(output);
  LoopNest::flatten({loops[2], loops[3]});
  loops = nest.getLoopStmtsFor(output);
  LoopNest::flatten({loops[0], loops[1]});
  loops = nest.getLoopStmtsFor(output);
  loops[0]->set_parallel();
  nest.prepareForCodegen();
  StmtPtr s = IRSimplifier::simplify(nest.root_stmt());
  LLVMCodeGen cg(s, {input, weight, bias, mean, var, output, eps});

  std::vector<CodeGen::CallArg> args;
  for (auto _ : state) {
    args.clear();
    output_ = at::empty_like(input_);
    for (auto const& t : {input_, weight_, bias_, mean_, var_, output_}) {
      args.push_back(t.data_ptr<float>());
    }
    args.push_back(eps_);
    cg.call(args);
  }
}

BENCHMARK_DEFINE_F(BatchNorm, ATenRelu)(benchmark::State& state) {
  for (auto _ : state) {
    output_ = at::batch_norm(
        input_,
        weight_,
        bias_,
        mean_,
        var_,
        training_,
        momentum_,
        eps_,
        cudnn_enabled_);
    output_.relu_();
  }
}

BENCHMARK_DEFINE_F(BatchNorm, NNCRelu)(benchmark::State& state) {
  BufHandle input("input", {N_, C_, H_, W_}, kFloat);
  BufHandle weight("weight", {C_}, kFloat);
  BufHandle bias("bias", {C_}, kFloat);
  BufHandle mean("mean", {C_}, kFloat);
  BufHandle var("var", {C_}, kFloat);
  VarHandle eps("eps", kFloat);

  using axis = const VarHandle&;
  Tensor output =
      Compute("output", {N_, C_, H_, W_}, [&](axis n, axis c, axis h, axis w) {
        // Compute affine terms.
        auto inv_var = FloatImm::make(1.0f) / sqrt(var.load(c) + eps);
        auto weight_v = weight.load(c);
        auto bias_v = bias.load(c);
        auto alpha = inv_var * weight_v;
        auto beta = bias_v - mean.load(c) * alpha;

        auto bn = input.load(n, c, h, w) * alpha + beta;
        return CompareSelect::make(bn, 0.f, 0.f, bn, kLT);
      });
  LoopNest nest({output});
  nest.prepareForCodegen();
  StmtPtr s = IRSimplifier::simplify(nest.root_stmt());
  LLVMCodeGen cg(s, {input, weight, bias, mean, var, output, eps});

  std::vector<CodeGen::CallArg> args;
  for (auto _ : state) {
    args.clear();
    output_ = at::empty_like(input_);
    for (auto const& t : {input_, weight_, bias_, mean_, var_, output_}) {
      args.push_back(t.data_ptr<float>());
    }
    args.push_back(eps_);
    cg.call(args);
  }
}

BENCHMARK_REGISTER_F(BatchNorm, ATen)
    ->Args({1, 64, 112, 112})
    ->Args({1, 256, 14, 14})
    ->Args({1, 128, 28, 28})
    ->Args({1, 64, 56, 56})
    ->Args({1, 512, 7, 7})
    ->Args({5, 64, 112, 112})
    ->Args({5, 256, 14, 14})
    ->Args({5, 128, 28, 28})
    ->Args({5, 64, 56, 56})
    ->Args({5, 512, 7, 7});
BENCHMARK_REGISTER_F(BatchNorm, NNC)
    ->Args({1, 64, 112, 112})
    ->Args({1, 256, 14, 14})
    ->Args({1, 128, 28, 28})
    ->Args({1, 64, 56, 56})
    ->Args({1, 512, 7, 7})
    ->Args({5, 64, 112, 112})
    ->Args({5, 256, 14, 14})
    ->Args({5, 128, 28, 28})
    ->Args({5, 64, 56, 56})
    ->Args({5, 512, 7, 7});
BENCHMARK_REGISTER_F(BatchNorm, ATenRelu)
    ->Args({1, 64, 112, 112})
    ->Args({1, 256, 14, 14})
    ->Args({1, 128, 28, 28})
    ->Args({1, 64, 56, 56})
    ->Args({1, 512, 7, 7})
    ->Args({5, 64, 112, 112})
    ->Args({5, 256, 14, 14})
    ->Args({5, 128, 28, 28})
    ->Args({5, 64, 56, 56})
    ->Args({5, 512, 7, 7});
BENCHMARK_REGISTER_F(BatchNorm, NNCRelu)
    ->Args({1, 64, 112, 112})
    ->Args({1, 256, 14, 14})
    ->Args({1, 128, 28, 28})
    ->Args({1, 64, 56, 56})
    ->Args({1, 512, 7, 7})
    ->Args({5, 64, 112, 112})
    ->Args({5, 256, 14, 14})
    ->Args({5, 128, 28, 28})
    ->Args({5, 64, 56, 56})
    ->Args({5, 512, 7, 7});
