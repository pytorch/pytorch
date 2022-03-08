#include <gtest/gtest.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
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
  for (int idx = 0; idx < testDims.size(); idx++) {
    const auto& dims = testDims[idx];
    const auto& outShape = outputShapes[idx];

    BufHandle a("a", {M, N}, kFloat);
    Tensor b = computeSum({a, dims, false}, outShape, c10::kFloat, at::kCPU);
    auto cg = compile({a}, {b});

    auto at = at::arange(M * N, at::kFloat).view({M, N});
    auto ref = at::sum(at, dims);
    auto bt = at::empty_like(ref);

    cg->call({at.data_ptr<float>(), bt.data_ptr<float>()});

    ASSERT_TRUE(at::allclose(bt, ref));
  }
}

TEST(Ops, Stack) {
  constexpr int N = 2;
  std::vector<IntList> testDims = {{0}, {1}, {0, 1}, {1, 0}, {2, 3}, {3, 2, 2}};
  std::vector<std::vector<ExprHandle>> outputShapes = {
      {N, ExprHandle(0)},
      {N, ExprHandle(1)},
      {N, ExprHandle(0), ExprHandle(1)}};
  for (int idx = 0; idx < testDims.size(); idx++) {
    // Construct the input buffer list
    const auto& dims = testDims[idx];
    std::vector<ExprHandle> inShape;
    for (auto d : dims) {
      inShape.push_back(ExprHandle(d));
    }

    BufHandle a("a", inShape, kFloat);
    BufHandle b("b", inShape, kFloat);
    std::vector<BufHandle> buflist;
    buflist.push_back(a);
    buflist.push_back(b);

    auto at = at::rand(dims, at::kFloat);
    auto bt = at::rand(dims, at::kFloat);

    // Vary the stack dim arg from 0 to 1
    for (int stackIdx = 0; stackIdx < 2; stackIdx++) {
      // Compute the output shape
      std::vector<ExprHandle> outShape(inShape.begin(), inShape.end());
      outShape.insert(outShape.begin() + stackIdx, ExprHandle(2));

      int64_t argDim = stackIdx;
      Tensor c =
          computeStack({buflist, argDim}, outShape, c10::kFloat, at::kCPU);
      auto cg = compile({a, b}, {c});

      auto ref = at::stack({at, bt}, argDim);
      auto ct = at::empty_like(ref);

      cg->call(
          {at.data_ptr<float>(), bt.data_ptr<float>(), ct.data_ptr<float>()});

      ASSERT_TRUE(at::allclose(ct, ref));
    }
  }
}
