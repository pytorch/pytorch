#include <gtest/gtest.h>
#include <ATen/core/dispatch/KernelFunction.h>
#include <ATen/core/op_registration/test_helpers.h>

using c10::IValue;
using c10::optional;
using std::vector;
using c10::OperatorKernel;
using c10::Stack;
using c10::KernelFunction;

namespace {

namespace boxed {
optional<vector<IValue>> called_boxed_func_with_args;

void boxed_func_with_return(OperatorKernel* functor, Stack* stack) {
  called_boxed_func_with_args = *stack;

  stack->clear();
  stack->push_back(5);
}

void boxed_func_without_return(OperatorKernel* functor, Stack* stack) {
  called_boxed_func_with_args = *stack;

  stack->clear();
}
}

TEST(KernelFunctionTest, givenBoxedFunction_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction(&boxed::boxed_func_with_return);

  boxed::called_boxed_func_with_args = c10::nullopt;
  vector<IValue> stack {3, 4};

  func.callBoxed(&stack);

  EXPECT_TRUE(boxed::called_boxed_func_with_args.has_value());
  EXPECT_EQ(2, boxed::called_boxed_func_with_args->size());
  EXPECT_EQ(3, boxed::called_boxed_func_with_args->at(0).toInt());
  EXPECT_EQ(4, boxed::called_boxed_func_with_args->at(1).toInt());
  EXPECT_EQ(1, stack.size());
  EXPECT_EQ(5, stack[0].toInt());
}

TEST(KernelFunctionTest, givenBoxedFunctionWithReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction(&boxed::boxed_func_with_return);

  boxed::called_boxed_func_with_args = c10::nullopt;

  int64_t result = func.callUnboxed<int64_t, int64_t, int64_t>(3, 4);

  EXPECT_TRUE(boxed::called_boxed_func_with_args.has_value());
  EXPECT_EQ(2, boxed::called_boxed_func_with_args->size());
  EXPECT_EQ(3, boxed::called_boxed_func_with_args->at(0).toInt());
  EXPECT_EQ(4, boxed::called_boxed_func_with_args->at(1).toInt());
  EXPECT_EQ(5, result);
}

TEST(KernelFunctionTest, givenBoxedFunctionWithoutReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction(&boxed::boxed_func_without_return);

  boxed::called_boxed_func_with_args = c10::nullopt;

  func.callUnboxed<void, int64_t, int64_t>(3, 4);

  EXPECT_TRUE(boxed::called_boxed_func_with_args.has_value());
  EXPECT_EQ(2, boxed::called_boxed_func_with_args->size());
  EXPECT_EQ(3, boxed::called_boxed_func_with_args->at(0).toInt());
  EXPECT_EQ(4, boxed::called_boxed_func_with_args->at(1).toInt());
}

TEST(KernelFunctionTest, givenBoxedFunctionWithReturn_whenCallingUnboxedOnly_thenCrashes) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction(&boxed::boxed_func_with_return);

  expectThrows<c10::Error>([&] {
    int64_t result = func.callUnboxedOnly<int64_t, int64_t, int64_t>(3, 4);
  }, "Tried to call KernelFunction::callUnboxedOnly() for a kernel that doesn't have an unboxed version.");
}

TEST(KernelFunctionTest, givenBoxedFunctionWithoutReturn_whenCallingUnboxedOnly_thenCrashes) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction(&boxed::boxed_func_without_return);

  expectThrows<c10::Error>([&] {
    func.callUnboxedOnly<void, int64_t, int64_t>(3, 4);
  }, "Tried to call KernelFunction::callUnboxedOnly() for a kernel that doesn't have an unboxed version.");
}

}

// TODO Test all KernelFunction::makeFromXXX() functions, each with callBoxed, callUnboxed and callUnboxedOnly. Make sure to test both, regular and void returns.
//   - makeFromUnboxedFunctor
//   - makeFromUnboxedFunctor(factory)
//   - makeFromUnboxedOnlyFunctor
//   - makeFromUnboxedFunction
//   - makeFromUnboxedOnlyFunction
//   - makeFromUnboxedRuntimeFunction
//   - makeFromUnboxedLambda
// TODO Also test different variants of calling unboxed with wrong signatures
// TODO Test different argument types?
