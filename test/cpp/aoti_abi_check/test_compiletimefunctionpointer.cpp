#include <gtest/gtest.h>

#include <torch/headeronly/core/CompileTimeFunctionPointer.h>

#include <type_traits>

namespace {
int add(int a, int b) {
  return a + b;
}
} // namespace

TEST(TestCompileTimeFunctionPointer, TestCompileTimeFunctionPointer) {
  using Add = TORCH_FN_TYPE(add);
  EXPECT_EQ(3, Add::func_ptr()(1, 2));
  // Bind to a local first: TORCH_FN expands to a template-id whose comma would
  // otherwise be parsed as an extra gtest macro argument.
  auto fn = TORCH_FN(add);
  EXPECT_EQ(3, fn.func_ptr()(1, 2));

  static_assert(
      torch::headeronly::is_compile_time_function_pointer<Add>::value,
      "Add should be a compile time function pointer");
  static_assert(
      !torch::headeronly::is_compile_time_function_pointer<int>::value,
      "int is not a compile time function pointer");
  static_assert(
      std::is_same_v<
          Add,
          torch::headeronly::CompileTimeFunctionPointer<int(int, int), add>>,
      "TORCH_FN_TYPE should alias torch::headeronly::CompileTimeFunctionPointer");
}
