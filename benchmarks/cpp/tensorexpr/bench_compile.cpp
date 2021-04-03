#include <benchmark/benchmark.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>

#ifdef TORCH_ENABLE_LLVM
namespace te = torch::jit::tensorexpr;

static void BM_CompileSwish(benchmark::State& state) {
  for (auto _ : state) {
    constexpr int N = 512;
    te::KernelScope ks;
    te::VarHandle n("n", te::kInt);
    te::Placeholder A(te::BufHandle("A", {N}, te::kFloat));
    te::Tensor* relu = te::Compute("relu", {{n, "n"}}, [&](const te::VarHandle& i) {
      return te::Max::make(A.load(i), 0.f, false);
    });
    te::Tensor* min6 = te::Compute("min6", {{n, "n"}}, [&](const te::VarHandle& i) {
      return te::Min::make(relu->call(i), 6.f, false);
    });
    te::Tensor* plus3 = te::Compute("plus3", {{n, "n"}}, [&](const te::VarHandle& i) {
      return min6->call(i) + 3.f;
    });
    te::Tensor* times = te::Compute("times", {{n, "n"}}, [&](const te::VarHandle& i) {
      return A.load(i) * plus3->call(i);
    });
    te::Tensor* sixth = te::Compute("sixth", {{n, "n"}}, [&](const te::VarHandle& i) {
      return times->call(i) * 1.f / 6.f;
    });
    te::LoopNest nest({sixth}, {relu, min6, plus3, times});
    for (auto tensor : {relu, min6, plus3, times}) {
      nest.computeInline(tensor->buf());
    }
    nest.prepareForCodegen();
    te::Stmt* s = te::IRSimplifier::simplify(nest.root_stmt());
    te::LLVMCodeGen cg(s, {A, sixth});
  }
}

static void BM_CompileSwishLLVMOnly(benchmark::State& state) {
  constexpr int N = 512;
  te::KernelScope ks;
  te::VarHandle n("n", te::kInt);
  te::Placeholder A(te::BufHandle("A", {N}, te::kFloat));
  te::Tensor* relu = te::Compute("relu", {{n, "n"}}, [&](const te::VarHandle& i) {
    return te::Max::make(A.load(i), 0.f, false);
  });
  te::Tensor* min6 = te::Compute("min6", {{n, "n"}}, [&](const te::VarHandle& i) {
    return te::Min::make(relu->call(i), 6.f, false);
  });
  te::Tensor* plus3 = te::Compute("plus3", {{n, "n"}}, [&](const te::VarHandle& i) {
    return min6->call(i) + 3.f;
  });
  te::Tensor* times = te::Compute("times", {{n, "n"}}, [&](const te::VarHandle& i) {
    return A.load(i) * plus3->call(i);
  });
  te::Tensor* sixth = te::Compute("sixth", {{n, "n"}}, [&](const te::VarHandle& i) {
    return times->call(i) * 1.f / 6.f;
  });
  te::LoopNest nest({sixth}, {relu, min6, plus3, times, sixth});
  for (auto tensor : {relu, min6, plus3, times}) {
    nest.computeInline(tensor->buf());
  }
  nest.prepareForCodegen();
  te::Stmt* s = te::IRSimplifier::simplify(nest.root_stmt());
  for (auto _ : state) {
    te::LLVMCodeGen cg(s, {A, sixth});
  }
}

BENCHMARK(BM_CompileSwish);
BENCHMARK(BM_CompileSwishLLVMOnly);
#endif // TORCH_ENABLE_LLVM
