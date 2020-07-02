#pragma once

#include <c10/util/TypeTraits.h>

namespace c10 {

/**
 * Represent a function pointer as a C++ type.
 * This allows using the function pointer as a type
 * in a template and calling it from inside the template
 * allows the compiler to inline the call because it
 * knows the function pointer at compile time.
 *
 * Example 1:
 *  int add(int a, int b) {return a + b;}
 *  using Add = TORCH_FN_TYPE(add);
 *  template<class Func> struct Executor {
 *    int execute(int a, int b) {
 *      return Func::func_ptr()(a, b);
 *    }
 *  };
 *  Executor<Add> executor;
 *  EXPECT_EQ(3, executor.execute(1, 2));
 *
 * Example 2:
 *  int add(int a, int b) {return a + b;}
 *  template<class Func> int execute(Func, int a, int b) {
 *    return Func::func_ptr()(a, b);
 *  }
 *  EXPECT_EQ(3, execute(TORCH_FN(add), 1, 2));
 */
template<class FuncType_, FuncType_* func_ptr_>
struct CompileTimeFunctionPointer final {
  static_assert(guts::is_function_type<FuncType_>::value, "TORCH_FN can only wrap function types.");
  using FuncType = FuncType_;

  static constexpr FuncType* func_ptr() {
    return func_ptr_;
  }
};

template<class T> struct is_compile_time_function_pointer : std::false_type {};
template<class FuncType, FuncType* func_ptr>
struct is_compile_time_function_pointer<CompileTimeFunctionPointer<FuncType, func_ptr>> : std::true_type {};

}

#define TORCH_FN_TYPE(func) ::c10::CompileTimeFunctionPointer<std::remove_pointer_t<std::remove_reference_t<decltype(func)>>, func>
#define TORCH_FN(func) TORCH_FN_TYPE(func)()
