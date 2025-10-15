#include <c10/core/CompileTimeFunctionPointer.h>
#include <gtest/gtest.h>

namespace test_is_compile_time_function_pointer {
static_assert(!c10::is_compile_time_function_pointer<void()>::value);

static void dummy() {}
static_assert(
    c10::is_compile_time_function_pointer<TORCH_FN_TYPE(dummy)>::value);
} // namespace test_is_compile_time_function_pointer

namespace test_access_through_type {
static void dummy() {}
using dummy_ptr = TORCH_FN_TYPE(dummy);
static_assert(c10::is_compile_time_function_pointer<dummy_ptr>::value);
static_assert(dummy_ptr::func_ptr() == &dummy);
static_assert(std::is_same_v<void(), dummy_ptr::FuncType>);
} // namespace test_access_through_type

namespace test_access_through_value {
static void dummy() {}
constexpr auto dummy_ptr = TORCH_FN(dummy);
static_assert(dummy_ptr.func_ptr() == &dummy);
static_assert(std::is_same_v<void(), decltype(dummy_ptr)::FuncType>);
} // namespace test_access_through_value

namespace test_access_through_type_also_works_if_specified_as_pointer {
static void dummy() {}
using dummy_ptr = TORCH_FN_TYPE(&dummy);
static_assert(c10::is_compile_time_function_pointer<dummy_ptr>::value);
static_assert(dummy_ptr::func_ptr() == &dummy);
static_assert(std::is_same_v<void(), dummy_ptr::FuncType>);
} // namespace test_access_through_type_also_works_if_specified_as_pointer

namespace test_access_through_value_also_works_if_specified_as_pointer {
static void dummy() {}
constexpr auto dummy_ptr = TORCH_FN(&dummy);
static_assert(dummy_ptr.func_ptr() == &dummy);
static_assert(std::is_same_v<void(), decltype(dummy_ptr)::FuncType>);
} // namespace test_access_through_value_also_works_if_specified_as_pointer

namespace test_run_through_type {
static int add(int a, int b) {
  return a + b;
}
using Add = TORCH_FN_TYPE(add);
template <class Func>
struct Executor {
  int execute(int a, int b) {
    return Func::func_ptr()(a, b);
  }
};

TEST(CompileTimeFunctionPointerTest, runFunctionThroughType) {
  Executor<Add> executor;
  EXPECT_EQ(3, executor.execute(1, 2));
}
} // namespace test_run_through_type

namespace test_run_through_value {
static int add(int a, int b) {
  return a + b;
}
template <class Func>
static int execute(Func, int a, int b) {
  return Func::func_ptr()(a, b);
}

TEST(CompileTimeFunctionPointerTest, runFunctionThroughValue) {
  EXPECT_EQ(3, execute(TORCH_FN(add), 1, 2));
}
} // namespace test_run_through_value
