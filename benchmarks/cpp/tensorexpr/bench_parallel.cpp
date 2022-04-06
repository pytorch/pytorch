#include <benchmark/benchmark.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/torch.h>

#include <immintrin.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class ParallelAdd : public benchmark::Fixture {
 public:
  void SetUp(const benchmark::State& state) override {
    at::set_num_threads(4);
    torch::manual_seed(0x12345678);
    M = state.range(0);
    A = torch::randn({M});
    B = torch::randn({M});
    C = torch::zeros({M});
  }

  void TearDown(benchmark::State& state) override {
    state.counters["tasks"] = benchmark::Counter(
        uint64_t(state.iterations()) * M, benchmark::Counter::kIsRate);
  }

  int M;
  at::Tensor A;
  at::Tensor B;
  at::Tensor C;
};

BENCHMARK_DEFINE_F(ParallelAdd, Simple)(benchmark::State& state) {
  BufHandle a_buf("a", {M}, kFloat);
  BufHandle b_buf("b", {M}, kFloat);
  Tensor c_tensor = Compute("c", {M}, [&](const VarHandle& m) {
    return a_buf.load(m) + b_buf.load(m);
  });
  LoopNest loop_nest({c_tensor});
  auto const& loops = loop_nest.getLoopStmtsFor(c_tensor);
  ForPtr m = loops[0];
  m->set_parallel();
  loop_nest.prepareForCodegen();
  StmtPtr stmt = loop_nest.root_stmt();
  LLVMCodeGen cg(stmt, {c_tensor, a_buf, b_buf});

  float* a_ptr = A.data_ptr<float>();
  float* b_ptr = B.data_ptr<float>();
  float* c_ptr = C.data_ptr<float>();
  std::vector<void*> args({c_ptr, a_ptr, b_ptr});
  cg.value<int>(args);
  for (const auto i : c10::irange(M)) {
    float diff = fabs(a_ptr[i] + b_ptr[i] - c_ptr[i]);
    TORCH_CHECK(diff < 1e-5);
  }

  for (auto _ : state) {
    cg.value<int>(args);
  }
}

BENCHMARK_REGISTER_F(ParallelAdd, Simple)->Args({1 << 16});

} // namespace tensorexpr
} // namespace jit
} // namespace torch
