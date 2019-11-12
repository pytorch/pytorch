#include <gtest/gtest.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/boxing/test_helpers.h>
#include <ATen/core/op_registration/op_registration.h>

using std::vector;
using std::tuple;
using c10::optional;
using c10::IValue;
using c10::OperatorKernel;
using c10::OperatorHandle;
using c10::Stack;
using c10::KernelFunction;

namespace {

namespace kernels {
// This namespace contains several fake kernels.
// All kernels expect to be called with two int64_t arguments and store
// these arguments in called_with_args.
// The kernels with a return value return a single int value: 5.
// The expectXXX() functions further below use these invariants
// to check that calling a specific kernels works correctly.

optional<tuple<int64_t, int64_t>> called_with_args;

void boxed_func_with_return(const OperatorHandle& /*opHandle*/, Stack* stack) {
  EXPECT_EQ(2, stack->size());
  EXPECT_TRUE(stack->at(0).isInt());
  EXPECT_TRUE(stack->at(1).isInt());
  called_with_args = tuple<int64_t, int64_t>(stack->at(0).toInt(), stack->at(1).toInt());

  stack->clear();
  stack->push_back(5);
}

void boxed_func_without_return(const OperatorHandle& /*opHandle*/, Stack* stack) {
  EXPECT_EQ(2, stack->size());
  EXPECT_TRUE(stack->at(0).isInt());
  EXPECT_TRUE(stack->at(1).isInt());
  called_with_args = tuple<int64_t, int64_t>(stack->at(0).toInt(), stack->at(1).toInt());

  stack->clear();
}

struct unboxed_functor_with_return final : OperatorKernel {
  int64_t operator()(int64_t a, int64_t b) {
    called_with_args = tuple<int64_t, int64_t>(a, b);
    return 5;
  }
};

struct unboxed_functor_without_return final : OperatorKernel {
  void operator()(int64_t a, int64_t b) {
    called_with_args = tuple<int64_t, int64_t>(a, b);
  }
};

struct unboxed_functor_with_return_factory final {
  std::unique_ptr<OperatorKernel> operator()() {
    return c10::guts::make_unique<unboxed_functor_with_return>();
  }
};

struct unboxed_functor_without_return_factory final {
  std::unique_ptr<OperatorKernel> operator()() {
    return c10::guts::make_unique<unboxed_functor_without_return>();
  }
};

int64_t unboxed_function_with_return(int64_t a, int64_t b) {
  called_with_args = tuple<int64_t, int64_t>(a, b);
  return 5;
}

void unboxed_function_without_return(int64_t a, int64_t b) {
  called_with_args = tuple<int64_t, int64_t>(a, b);
}

auto unboxed_lambda_with_return = [] (int64_t a, int64_t b) -> int64_t {
  called_with_args = tuple<int64_t, int64_t>(a, b);
  return 5;
};

auto unboxed_lambda_without_return = [] (int64_t a, int64_t b) -> void{
  called_with_args = tuple<int64_t, int64_t>(a, b);
};

OperatorHandle makeDummyOperatorHandle() {
  static auto registry = torch::RegisterOperators().op("my::dummy() -> ()");
  return c10::Dispatcher::singleton().findSchema({"my::dummy", ""}).value();
}

void expectBoxedCallingWithReturnWorks(const KernelFunction& func) {
  called_with_args = c10::nullopt;
  vector<IValue> stack {3, 4};
  OperatorHandle dummy = makeDummyOperatorHandle();

  func.callBoxed(dummy, &stack);

  EXPECT_TRUE(called_with_args.has_value());
  EXPECT_EQ((tuple<int64_t, int64_t>(3, 4)), *called_with_args);
  EXPECT_EQ(1, stack.size());
  EXPECT_TRUE(stack[0].isInt());
  EXPECT_EQ(5, stack[0].toInt());
}

void expectBoxedCallingWithoutReturnWorks(const KernelFunction& func) {
  called_with_args = c10::nullopt;
  vector<IValue> stack {3, 4};
  OperatorHandle dummy = makeDummyOperatorHandle();

  func.callBoxed(dummy, &stack);

  EXPECT_TRUE(called_with_args.has_value());
  EXPECT_EQ((tuple<int64_t, int64_t>(3, 4)), *called_with_args);
  EXPECT_EQ(0, stack.size());
}

void expectBoxedCallingFailsWith(const KernelFunction& func, const char* errorMessage) {
  called_with_args = c10::nullopt;
  vector<IValue> stack {3, 4};
  OperatorHandle dummy = makeDummyOperatorHandle();

  expectThrows<c10::Error>([&] {
    func.callBoxed(dummy, &stack);
  }, errorMessage);
}

void expectUnboxedCallingWithReturnWorks(const KernelFunction& func) {
  called_with_args = c10::nullopt;
  OperatorHandle dummy = makeDummyOperatorHandle();

  int64_t result = func.callUnboxed<int64_t, int64_t, int64_t>(dummy, 3, 4);

  EXPECT_TRUE(called_with_args.has_value());
  EXPECT_EQ((tuple<int64_t, int64_t>(3, 4)), *called_with_args);
  EXPECT_EQ(5, result);
}

void expectUnboxedCallingWithoutReturnWorks(const KernelFunction& func) {
  called_with_args = c10::nullopt;
  OperatorHandle dummy = makeDummyOperatorHandle();

  func.callUnboxed<void, int64_t, int64_t>(dummy, 3, 4);

  EXPECT_TRUE(called_with_args.has_value());
  EXPECT_EQ((tuple<int64_t, int64_t>(3, 4)), *called_with_args);
}
}

TEST(KernelFunctionTest, givenBoxedFunction_withReturn_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_with_return>();
  kernels::expectBoxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenBoxedFunction_withoutReturn_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_without_return>();
  kernels::expectBoxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenBoxedFunction_withReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_with_return>();
  kernels::expectUnboxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenBoxedFunction_withoutReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_without_return>();
  kernels::expectUnboxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunctor_withReturn_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedFunctor<false, kernels::unboxed_functor_with_return>(std::unique_ptr<OperatorKernel>(c10::guts::make_unique<kernels::unboxed_functor_with_return>()));
  kernels::expectBoxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunctor_withoutReturn_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedFunctor<false, kernels::unboxed_functor_without_return>(std::unique_ptr<OperatorKernel>(c10::guts::make_unique<kernels::unboxed_functor_without_return>()));
  kernels::expectBoxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunctor_withReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedFunctor<false, kernels::unboxed_functor_with_return>(std::unique_ptr<OperatorKernel>(c10::guts::make_unique<kernels::unboxed_functor_with_return>()));
  kernels::expectUnboxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunctor_withoutReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedFunctor<false, kernels::unboxed_functor_without_return>(std::unique_ptr<OperatorKernel>(c10::guts::make_unique<kernels::unboxed_functor_without_return>()));
  kernels::expectUnboxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunctorFactory_withReturn_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedFunctorFactory<kernels::unboxed_functor_with_return>(kernels::unboxed_functor_with_return_factory());
  kernels::expectBoxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunctorFactory_withoutReturn_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedFunctorFactory<kernels::unboxed_functor_without_return>(kernels::unboxed_functor_without_return_factory());
  kernels::expectBoxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunctorFactory_withReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedFunctorFactory<kernels::unboxed_functor_with_return>(kernels::unboxed_functor_with_return_factory());
  kernels::expectUnboxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunctorFactory_withoutReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedFunctorFactory<kernels::unboxed_functor_without_return>(kernels::unboxed_functor_without_return_factory());
  kernels::expectUnboxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedOnlyFunctor_withReturn_whenCallingBoxed_thenFails) {
  KernelFunction func = KernelFunction::makeFromUnboxedOnlyFunctor<kernels::unboxed_functor_with_return>(std::unique_ptr<OperatorKernel>(c10::guts::make_unique<kernels::unboxed_functor_with_return>()));
  kernels::expectBoxedCallingFailsWith(func, "Tried to call KernelFunction::callBoxed() on a KernelFunction that can only be called with KernelFunction::callUnboxed()");
}

TEST(KernelFunctionTest, givenUnboxedOnlyFunctor_withoutReturn_whenCallingBoxed_thenFails) {
  KernelFunction func = KernelFunction::makeFromUnboxedOnlyFunctor<kernels::unboxed_functor_without_return>(std::unique_ptr<OperatorKernel>(c10::guts::make_unique<kernels::unboxed_functor_without_return>()));
  kernels::expectBoxedCallingFailsWith(func, "Tried to call KernelFunction::callBoxed() on a KernelFunction that can only be called with KernelFunction::callUnboxed()");
}

TEST(KernelFunctionTest, givenUnboxedOnlyFunctor_withReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedOnlyFunctor<kernels::unboxed_functor_with_return>(std::unique_ptr<OperatorKernel>(c10::guts::make_unique<kernels::unboxed_functor_with_return>()));
  kernels::expectUnboxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedOnlyFunctor_withoutReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedOnlyFunctor<kernels::unboxed_functor_without_return>(std::unique_ptr<OperatorKernel>(c10::guts::make_unique<kernels::unboxed_functor_without_return>()));
  kernels::expectUnboxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunction_withReturn_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedFunction<decltype(kernels::unboxed_function_with_return), &kernels::unboxed_function_with_return>();
  kernels::expectBoxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunction_withoutReturn_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedFunction<decltype(kernels::unboxed_function_without_return), &kernels::unboxed_function_without_return>();
  kernels::expectBoxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunction_withReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedFunction<decltype(kernels::unboxed_function_with_return), &kernels::unboxed_function_with_return>();
  kernels::expectUnboxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunction_withoutReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedFunction<decltype(kernels::unboxed_function_without_return), &kernels::unboxed_function_without_return>();
  kernels::expectUnboxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedOnlyFunction_withReturn_whenCallingBoxed_thenFails) {
  KernelFunction func = KernelFunction::makeFromUnboxedOnlyFunction<decltype(kernels::unboxed_function_with_return), &kernels::unboxed_function_with_return>();
  kernels::expectBoxedCallingFailsWith(func, "Tried to call KernelFunction::callBoxed() on a KernelFunction that can only be called with KernelFunction::callUnboxed()");
}

TEST(KernelFunctionTest, givenUnboxedOnlyFunction_withoutReturn_whenCallingBoxed_thenFails) {
  KernelFunction func = KernelFunction::makeFromUnboxedOnlyFunction<decltype(kernels::unboxed_function_without_return), &kernels::unboxed_function_without_return>();
  kernels::expectBoxedCallingFailsWith(func, "Tried to call KernelFunction::callBoxed() on a KernelFunction that can only be called with KernelFunction::callUnboxed()");
}

TEST(KernelFunctionTest, givenUnboxedOnlyFunction_withReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedOnlyFunction<decltype(kernels::unboxed_function_with_return), &kernels::unboxed_function_with_return>();
  kernels::expectUnboxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedOnlyFunction_withoutReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedOnlyFunction<decltype(kernels::unboxed_function_without_return), &kernels::unboxed_function_without_return>();
  kernels::expectUnboxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedRuntimeFunction_withReturn_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedRuntimeFunction(&kernels::unboxed_function_with_return);
  kernels::expectBoxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedRuntimeFunction_withoutReturn_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedRuntimeFunction(&kernels::unboxed_function_without_return);
  kernels::expectBoxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedRuntimeFunction_withReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedRuntimeFunction(&kernels::unboxed_function_with_return);
  kernels::expectUnboxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedRuntimeFunction_withoutReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedRuntimeFunction(&kernels::unboxed_function_without_return);
  kernels::expectUnboxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedLambda_withReturn_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedLambda(kernels::unboxed_lambda_with_return);
  kernels::expectBoxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedLambda_withoutReturn_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedLambda(kernels::unboxed_lambda_without_return);
  kernels::expectBoxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedLambda_withReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedLambda(kernels::unboxed_lambda_with_return);
  kernels::expectUnboxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedLambda_withoutReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedLambda(kernels::unboxed_lambda_without_return);
  kernels::expectUnboxedCallingWithoutReturnWorks(func);
}

}

// TODO Also test different variants of calling unboxed with wrong signatures
