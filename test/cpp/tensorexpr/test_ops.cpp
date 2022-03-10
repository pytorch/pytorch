#include <gtest/gtest.h>
#include <test/cpp/tensorexpr/test_base.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/operators/operators.h>
#include <torch/csrc/jit/testing/file_check.h>
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
  { // Check that we throw an error for invalid dim
    std::vector<ExprHandle> inShape = {ExprHandle(1)};

    BufHandle a("a", inShape, kFloat);
    BufHandle b("b", inShape, kFloat);
    std::vector<BufHandle> buflist;
    buflist.push_back(a);
    buflist.push_back(b);

    std::vector<ExprHandle> outShape = {ExprHandle(1), ExprHandle(2)};

    int64_t argDim = 2;
    ASSERT_THROWS_WITH(
        computeStack({buflist, argDim}, outShape, c10::kFloat, at::kCPU),
        "Invalid index");

    argDim = -3;
    ASSERT_THROWS_WITH(
        computeStack({buflist, argDim}, outShape, c10::kFloat, at::kCPU),
        "Invalid index");
  }
  { // Check that we throw an error for input buffers with different sizes
    std::vector<ExprHandle> inShape1 = {ExprHandle(1)};
    std::vector<ExprHandle> inShape2 = {ExprHandle(2)};
    int64_t argDim = 0;

    BufHandle a("a", inShape1, kFloat);
    BufHandle b("b", inShape2, kFloat);
    std::vector<BufHandle> buflist;
    buflist.push_back(a);
    buflist.push_back(b);

    std::vector<ExprHandle> outShape = {ExprHandle(2), ExprHandle(2)};

    ASSERT_THROWS_WITH(
        computeStack({buflist, argDim}, outShape, c10::kFloat, at::kCPU),
        "different sizes");
  }
  { // Check the IR
    std::vector<ExprHandle> inShape = {ExprHandle(1)};
    int64_t argDim = 0;

    BufHandle a("a", inShape, kFloat);
    BufHandle b("b", inShape, kFloat);
    std::vector<BufHandle> buflist;
    buflist.push_back(a);
    buflist.push_back(b);

    std::vector<ExprHandle> outShape = {ExprHandle(2), ExprHandle(1)};

    Tensor c = computeStack({buflist, argDim}, outShape, c10::kFloat, at::kCPU);
    auto cg = compile({a, b}, {c});

    StmtPtr s = cg->stmt();
    std::ostringstream oss;
    oss << *s;

    const std::string& verification_pattern =
        R"IR(
  # CHECK: for
  # CHECK-NEXT: aten_stack[i] = i==1 ? (b[0ll]) : (a[0ll]))IR";
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
  }
  { // Check correctness for stacking one tensor
    constexpr int N = 1;
    std::vector<IntList> testDims = {
        {0}, {1}, {0, 1}, {1, 0}, {2, 3}, {3, 2, 2}};
    for (int idx = 0; idx < testDims.size(); idx++) {
      // Construct the input buffer list
      const auto& dims = testDims[idx];
      std::vector<ExprHandle> inShape;
      for (auto d : dims) {
        inShape.push_back(ExprHandle(d));
      }

      BufHandle a("a", inShape, kFloat);
      std::vector<BufHandle> buflist;
      buflist.push_back(a);

      auto at = at::rand(dims, at::kFloat);

      // Vary the stack dim arg from 0 to 1
      for (int stackIdx = 0; stackIdx < 2; stackIdx++) {
        // Compute the output shape
        std::vector<ExprHandle> outShape(inShape.begin(), inShape.end());
        outShape.insert(outShape.begin() + stackIdx, N);

        int64_t argDim = stackIdx;
        Tensor b =
            computeStack({buflist, argDim}, outShape, c10::kFloat, at::kCPU);
        auto cg = compile({a}, {b});

        auto ref = at::stack({at}, argDim);
        auto bt = at::empty_like(ref);

        cg->call({at.data_ptr<float>(), bt.data_ptr<float>()});

        ASSERT_TRUE(at::allclose(bt, ref));
      }

      // Vary the stack dim arg from -1 to -2
      for (int stackIdx = -1; stackIdx > -3; stackIdx--) {
        // Compute the output shape
        std::vector<ExprHandle> outShape(inShape.begin(), inShape.end());
        outShape.insert(outShape.end() + stackIdx + 1, N);

        int64_t argDim = stackIdx;
        Tensor b =
            computeStack({buflist, argDim}, outShape, c10::kFloat, at::kCPU);
        auto cg = compile({a}, {b});

        auto ref = at::stack({at}, argDim);
        auto bt = at::empty_like(ref);

        cg->call({at.data_ptr<float>(), bt.data_ptr<float>()});

        ASSERT_TRUE(at::allclose(bt, ref));
      }
    }
  }
  { // Check correctness for stacking two tensors
    constexpr int N = 2;
    std::vector<IntList> testDims = {
        {0}, {1}, {0, 1}, {1, 0}, {2, 3}, {3, 2, 2}};
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
        outShape.insert(outShape.begin() + stackIdx, N);

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

      // Vary the stack dim arg from -1 to -2
      for (int stackIdx = -1; stackIdx > -3; stackIdx--) {
        // Compute the output shape
        std::vector<ExprHandle> outShape(inShape.begin(), inShape.end());
        outShape.insert(outShape.end() + stackIdx + 1, N);

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
  { // Check correctness for stacking more than two tensors
    constexpr int N = 4;
    std::vector<IntList> testDims = {
        {0}, {1}, {0, 1}, {1, 0}, {2, 3}, {3, 2, 2}};
    for (int idx = 0; idx < testDims.size(); idx++) {
      // Construct the input buffer list
      const auto& dims = testDims[idx];
      std::vector<ExprHandle> inShape;
      for (auto d : dims) {
        inShape.push_back(ExprHandle(d));
      }

      BufHandle a("a", inShape, kFloat);
      BufHandle b("b", inShape, kFloat);
      BufHandle c("c", inShape, kFloat);
      BufHandle d("d", inShape, kFloat);
      std::vector<BufHandle> buflist;
      buflist.push_back(a);
      buflist.push_back(b);
      buflist.push_back(c);
      buflist.push_back(d);

      auto at = at::rand(dims, at::kFloat);
      auto bt = at::rand(dims, at::kFloat);
      auto ct = at::rand(dims, at::kFloat);
      auto dt = at::rand(dims, at::kFloat);

      // Vary the stack dim arg from 0 to 1
      for (int stackIdx = 0; stackIdx < 2; stackIdx++) {
        // Compute the output shape
        std::vector<ExprHandle> outShape(inShape.begin(), inShape.end());
        outShape.insert(outShape.begin() + stackIdx, N);

        int64_t argDim = stackIdx;
        Tensor e =
            computeStack({buflist, argDim}, outShape, c10::kFloat, at::kCPU);
        auto cg = compile({a, b, c, d}, {e});

        auto ref = at::stack({at, bt, ct, dt}, argDim);
        auto et = at::empty_like(ref);

        cg->call(
            {at.data_ptr<float>(),
             bt.data_ptr<float>(),
             ct.data_ptr<float>(),
             dt.data_ptr<float>(),
             et.data_ptr<float>()});

        ASSERT_TRUE(at::allclose(et, ref));
      }

      // Vary the stack dim arg from -1 to -2
      for (int stackIdx = -1; stackIdx > -3; stackIdx--) {
        // Compute the output shape
        std::vector<ExprHandle> outShape(inShape.begin(), inShape.end());
        outShape.insert(outShape.end() + stackIdx + 1, N);

        int64_t argDim = stackIdx;
        Tensor e =
            computeStack({buflist, argDim}, outShape, c10::kFloat, at::kCPU);
        auto cg = compile({a, b, c, d}, {e});

        auto ref = at::stack({at, bt, ct, dt}, argDim);
        auto et = at::empty_like(ref);

        cg->call(
            {at.data_ptr<float>(),
             bt.data_ptr<float>(),
             ct.data_ptr<float>(),
             dt.data_ptr<float>(),
             et.data_ptr<float>()});

        ASSERT_TRUE(at::allclose(et, ref));
      }
    }
  }
}
