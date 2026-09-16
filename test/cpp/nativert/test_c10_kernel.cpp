#include <ATen/core/op_registration/op_registration.h>
#include <gtest/gtest.h>
#include <torch/nativert/executor/ExecutionFrame.h>
#include <torch/nativert/graph/Graph.h>
#include <torch/nativert/kernels/C10Kernel.h>
#include <torch/torch.h>

namespace torch::nativert {

at::Tensor foo_kernel(const at::Tensor& a, const at::Tensor& b) {
  return a + b;
}

TEST(C10KernelTest, computeInternal) {
  auto registrar = c10::RegisterOperators().op(
      "test::foo(Tensor a, Tensor b) -> Tensor", &foo_kernel);

  static constexpr std::string_view source =
      R"(graph(%a, %b):
%x = test.foo.default(a=%a, b=%b)
return (%x)
)";

  auto graph = stringToGraph(source);
  const auto& nodes = graph->nodes();
  auto it = nodes.begin();
  std::advance(it, 1);
  const Node& node = *it;

  auto a = at::randn({6, 6, 6});
  auto b = at::randn({6, 6, 6});

  auto frame = ExecutionFrame(*graph);
  frame.setIValue(graph->getValue("a")->id(), a);
  frame.setIValue(graph->getValue("b")->id(), b);

  auto kernel = C10Kernel(&node);

  kernel.computeInternal(frame);

  at::Tensor expected = a + b;
  EXPECT_TRUE(
      torch::equal(frame.getTensor(graph->getValue("x")->id()), expected));
}

TEST(ScalarBinaryOpKernelTest, computeInternal) {
  static constexpr std::string_view source =
      R"(graph(%a, %b):
%x = _operator.add(a=%a, b=%b)
return (%x)
)";

  auto graph = stringToGraph(source);
  const auto& nodes = graph->nodes();
  auto it = nodes.begin();
  std::advance(it, 1);
  const Node& node = *it;

  auto a = 1;
  auto b = 2;

  auto frame = ExecutionFrame(*graph);
  frame.setIValue(graph->getValue("a")->id(), a);
  frame.setIValue(graph->getValue("b")->id(), b);

  auto kernel = ScalarBinaryOpKernel(&node);

  kernel.computeInternal(frame);

  auto expected = a + b;
  EXPECT_EQ(frame.getIValue(graph->getValue("x")->id()).toInt(), expected);
}

} // namespace torch::nativert
