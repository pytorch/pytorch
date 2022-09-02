#include <gtest/gtest.h>
#include <torch/csrc/jit/ir/irparser.h>
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

TEST(Ops, Embedding) {
  const auto graph_string = R"IR(
    graph(%embedding_weight : Float(10000, 300, strides=[300, 1], device=cpu),
          %indices : Long(2, 4, strides=[4, 1], device=cpu)):
      %padding_idx : int = prim::Constant[value=-1]()
      %1 : int = prim::Constant[value=1]()
      %2 : bool = prim::Constant[value=0]()
      %3 : Float(2, 4, 300, strides=[1200, 300, 1], requires_grad=0, device=cpu) = aten::embedding(%embedding_weight, %indices, %padding_idx, %2, %2)
      %4 : Float(2, 4, 300, strides=[1200, 300, 1], requires_grad=0, device=cpu) = aten::add(%3, %3, %1)
      %5 : Float(2, 4, 300, strides=[1200, 300, 1], requires_grad=0, device=cpu) = aten::mul(%4, %4)
      return (%5))IR";

  auto graph = std::make_shared<torch::jit::Graph>();
  parseIR(graph_string, &*graph);

  auto indices =
      at::randint(10, {2, 4}, at::TensorOptions(c10::kCPU).dtype(at::kLong));
  auto embedding_weight =
      at::rand({10000, 300}, at::TensorOptions(c10::kCPU).dtype(at::kFloat));
  auto y_1 = at::embedding(embedding_weight, indices);
  auto y_2 = at::add(y_1, y_1, 1);
  auto y_expected = at::mul(y_2, y_2);

  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {embedding_weight, indices};

  std::vector<c10::IValue> stack = at::fmap<c10::IValue>(inputs);
  k.run(stack);
  auto y = stack[0].toTensor();

  bool check = at::allclose(y_expected, y);
  if (!check) {
    std::cout << "indices:\n" << indices << std::endl;
    std::cout << "embedding_weight:\n" << embedding_weight << std::endl;
    std::cout << "y_expected:\n" << y_expected << std::endl;
    std::cout << "y:\n" << y << std::endl;
  }
  TORCH_CHECK_EQ(check, 1);
}
