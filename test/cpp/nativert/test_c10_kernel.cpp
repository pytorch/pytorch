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

at::Tensor throwing_tensor_list_kernel(
    const at::TensorList& tensors,
    const at::Tensor& other) {
  TORCH_CHECK(tensors.empty(), "intentional failure");
  return other;
}

TEST(C10KernelTest, errorMessageRebuildsConsumedListArgs) {
  // Unboxing moves a Tensor[] argument out of the stack, so the error path must
  // re-derive the arguments from the frame instead of formatting the stack it
  // just handed to callBoxed -- otherwise every list argument reads as "None".
  auto registrar = c10::RegisterOperators().op(
      "test::throwing_list(Tensor[] tensors, Tensor other) -> Tensor",
      &throwing_tensor_list_kernel);

  static constexpr std::string_view source =
      R"(graph(%a, %b, %other):
%tensors[] = prim.ListPack(l0=%a, l1=%b)
%x = test.throwing_list.default(tensors=%tensors, other=%other)
return (%x)
)";

  auto graph = stringToGraph(source);
  const auto& nodes = graph->nodes();
  auto it = nodes.begin();
  std::advance(it, 2);
  const Node& node = *it;

  // Seed the packed list directly; only the throwing node is executed here.
  auto frame = ExecutionFrame(*graph);
  frame.setIValue(
      graph->getValue("tensors")->id(),
      c10::List<at::Tensor>({at::tensor({1, 2}), at::tensor({3, 4})}));
  frame.setIValue(graph->getValue("other")->id(), at::tensor({5, 6}));

  auto kernel = C10Kernel(&node);

  try {
    kernel.computeInternal(frame);
    FAIL() << "expected test::throwing_list to throw";
  } catch (const c10::Error& e) {
    const std::string message = e.what();
    EXPECT_EQ(message.find("arg0 tensors: None"), std::string::npos) << message;
    EXPECT_NE(message.find("[int[2]cpu, int[2]cpu, ]"), std::string::npos)
        << message;
    // A Tensor bound to `const Tensor&` is borrowed, not moved, so it survives
    // unboxing and was already printed correctly before this change.
    EXPECT_NE(message.find("arg1 other: Tensor int[2]cpu"), std::string::npos)
        << message;
  }
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
