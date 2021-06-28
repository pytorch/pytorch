#include <gtest/gtest.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/operators/operators.h>
#include <torch/torch.h>

using namespace torch::jit::tensorexpr;

using Tensors = std::vector<Tensor*>;
using Args = std::vector<CodeGen::BufferArg>;
std::unique_ptr<SimpleIREvaluator> compile(
    const Args& inputs,
    const Tensors& outputs) {
  LoopNest nest({outputs});
  nest.prepareForCodegen();
  nest.simplify();
  auto join = inputs;
  join.insert(join.end(), outputs.begin(), outputs.end());
  return std::make_unique<SimpleIREvaluator>(nest.root_stmt(), join);
}

TEST(Ops, Sum) {
  KernelScope ks;

  std::vector<IntList> testDims = {{0}, {1}, {0, 1}};
  for (auto const& dims : testDims) {
    constexpr int M = 8;
    constexpr int N = 16;

    Placeholder a("a", kFloat, {M, N});
    Tensor* b = computeSum({a.handle(), dims, false}, c10::kFloat);
    auto cg = compile({a}, {b});

    auto at = at::arange(M * N, at::kFloat).view({M, N});
    auto ref = at::sum(at, dims);
    auto bt = at::empty_like(ref);

    cg->call({at.data_ptr<float>(), bt.data_ptr<float>()});

    ASSERT_TRUE(at::allclose(bt, ref));
  }
}
