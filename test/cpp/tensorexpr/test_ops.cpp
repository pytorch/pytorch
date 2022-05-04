#include <gtest/gtest.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/operators/operators.h>
#include <torch/torch.h>

using namespace torch::jit::tensorexpr;

using Tensors = std::vector<Tensor>;
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
  constexpr int M = 8;
  constexpr int N = 16;
  std::vector<IntList> testDims = {{0}, {1}, {0, 1}};
  std::vector<std::vector<ExprHandle>> outputShapes = {{N}, {M}, {}};
  for (unsigned idx = 0; idx < testDims.size(); idx++) {
    const auto& dims = testDims[idx];
    const auto& outShape = outputShapes[idx];

    BufHandle a("a", {M, N}, kFloat);
    std::vector<ExprHandle> outStrides =
        c10::fmap<ExprHandle>(make_contiguous_strides(outShape));
    Tensor b = computeSum(
        {a, dims, false}, outShape, outStrides, c10::kFloat, at::kCPU);
    auto cg = compile({a}, {b});

    auto at = at::arange(M * N, at::kFloat).view({M, N});
    auto ref = at::sum(at, dims);
    auto bt = at::empty_like(ref);

    cg->call({at.data_ptr<float>(), bt.data_ptr<float>()});

    ASSERT_TRUE(at::allclose(bt, ref));
  }
}

TEST(Ops, ChannelsLastSum) {
  constexpr int A = 2;
  constexpr int B = 3;
  constexpr int C = 4;
  constexpr int D = 5;
  constexpr int E = 6;
  std::vector<IntList> testDims = {{0}, {1}, {0, 1}};

  std::vector<std::vector<ExprHandle>> outputShapes = {
      {B, C, D, E}, {A, C, D, E}, {C, D, E}};
  for (unsigned idx = 0; idx < testDims.size(); idx++) {
    const auto& dims = testDims[idx];
    const auto& outShape = outputShapes[idx];

    BufHandle a("a", {A, B, C, D, E}, kFloat);
    std::vector<ExprHandle> outStrides =
        c10::fmap<ExprHandle>(make_channels_last_strides(outShape));
    Tensor b = computeSum(
        {a, dims, false}, outShape, outStrides, c10::kFloat, at::kCPU);
    auto cg = compile({a}, {b});

    auto at = at::arange(A * B * C * D * E, at::kFloat).view({A, B, C, D, E});
    auto ref = at::sum(at, dims);
    auto bt = at::empty_like(ref);

    cg->call({at.data_ptr<float>(), bt.data_ptr<float>()});

    ASSERT_TRUE(at::allclose(bt, ref));
  }
}
