#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/ops/tensor.h>
#include <gtest/gtest.h>
#include <torch/nativert/executor/OpKernel.h>

namespace torch::nativert {

int64_t increment_kernel(const at::Tensor& tensor, int64_t input) {
  return input + 1;
}

TEST(OpKernelTest, GetOperatorForTargetValid) {
  auto registrar = c10::RegisterOperators().op(
      "test::foo(Tensor dummy, int input) -> int", &increment_kernel);
  std::string target = "test.foo.default";
  EXPECT_NO_THROW({
    c10::OperatorHandle handle = getOperatorForTarget(target);
    EXPECT_TRUE(handle.hasSchema());
    EXPECT_EQ(handle.operator_name().name, "test::foo");
    EXPECT_EQ(handle.operator_name().overload_name, "");
  });
}

TEST(OpKernelTest, GetOperatorForTargetInvalid) {
  std::string target = "invalid.target";
  EXPECT_THROW(getOperatorForTarget(target), c10::Error);
}

TEST(OpKernelTest, GetReadableArgs) {
  c10::FunctionSchema schema = c10::FunctionSchema(
      "test_op",
      "",
      {c10::Argument("tensor_arg"),
       c10::Argument("tensor_list_arg"),
       c10::Argument("int_arg"),
       c10::Argument("none_arg")},
      {});
  std::vector<c10::IValue> stack = {
      at::tensor({1, 2, 3}),
      c10::IValue(
          std::vector<at::Tensor>{at::tensor({1, 2}), at::tensor({3, 4})}),
      c10::IValue(1),
      c10::IValue(),
  };
  std::string expected =
      "arg0 tensor_arg: Tensor int[3]cpu\n"
      "arg1 tensor_list_arg: GenericList [int[2]cpu, int[2]cpu, ]\n"
      "arg2 int_arg: Int 1\n"
      "arg3 none_arg: None \n";

  std::string result = readableArgs(schema, stack);
  EXPECT_EQ(result, expected);
}

} // namespace torch::nativert
