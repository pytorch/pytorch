#include <gtest/gtest.h>
#include <ATen/ATen.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/boxing/impl/test_helpers.h>
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
// Some of these kernels expect to be called with two int64_t arguments
// and store these arguments in called_with_args.
// Kernels may return a single value, or multiple values, or no value.
// The kernels with a single return value return int value 5,
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

void boxed_func_with_multi_return(const OperatorHandle& /*opHandle*/, Stack* stack) {
  EXPECT_EQ(2, stack->size());
  EXPECT_TRUE(stack->at(0).isInt());
  int64_t a = stack->at(0).toInt();
  EXPECT_TRUE(stack->at(1).isInt());
  int64_t b = stack->at(1).toInt();
  called_with_args = tuple<int64_t, int64_t>(a, b);

  stack->clear();
  torch::jit::push(stack, a + b);
  torch::jit::push(stack, a * b);
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
    return std::make_unique<unboxed_functor_with_return>();
  }
};

struct unboxed_functor_without_return_factory final {
  std::unique_ptr<OperatorKernel> operator()() {
    return std::make_unique<unboxed_functor_without_return>();
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

//
// boxed kernels that return refs to tensor arguments, a la inplace/outplace kernels
//

void boxed_func_for_inplace_op(const OperatorHandle& /*opHandle*/, Stack* stack) {
  // (Tensor(a!), Scalar) -> Tensor(a!)
  EXPECT_EQ(2, stack->size());

  ASSERT_TRUE(stack->at(0).isTensor());
  auto t = stack->at(0).toTensor();

  ASSERT_TRUE(stack->at(1).isScalar());
  auto s = stack->at(1).toScalar();

  t.add_(s);

  stack->clear();
  torch::jit::push(stack, t);
}

void boxed_func_for_outofplace_op(const OperatorHandle& /*opHandle*/, Stack* stack) {
  // (Scalar, Tensor(a!)) -> Tensor(a!)
  EXPECT_EQ(2, stack->size());

  ASSERT_TRUE(stack->at(0).isScalar());
  auto s = stack->at(0).toScalar();

  ASSERT_TRUE(stack->at(1).isTensor());
  auto t = stack->at(1).toTensor();

  t.add_(s);

  stack->clear();
  torch::jit::push(stack, t);
}

void boxed_func_for_outofplace_multi_op(const OperatorHandle& /*opHandle*/, Stack* stack) {
  // (Tensor(a!), Tensor(b!), Scalar, Scalar) -> (Tensor(a!), Tensor(b!))
  EXPECT_EQ(4, stack->size());

  ASSERT_TRUE(stack->at(0).isTensor());
  auto t1 = stack->at(0).toTensor();

  ASSERT_TRUE(stack->at(1).isTensor());
  auto t2 = stack->at(1).toTensor();

  ASSERT_TRUE(stack->at(2).isScalar());
  auto s1 = stack->at(2).toScalar();

  ASSERT_TRUE(stack->at(3).isScalar());
  auto s2 = stack->at(3).toScalar();

  t1.add_(s1);
  t2.add_(s2);

  stack->clear();
  torch::jit::push(stack, t1);
  torch::jit::push(stack, t2);
}

void boxed_func_for_legacy_outofplace_multi_op(const OperatorHandle& /*opHandle*/, Stack* stack) {
  // (Scalar, Scalar, Tensor(a!), Tensor(b!)) -> (Tensor(a!), Tensor(b!))
  EXPECT_EQ(4, stack->size());

  ASSERT_TRUE(stack->at(0).isScalar());
  auto s1 = stack->at(0).toScalar();

  ASSERT_TRUE(stack->at(1).isScalar());
  auto s2 = stack->at(1).toScalar();

  ASSERT_TRUE(stack->at(2).isTensor());
  auto t1 = stack->at(2).toTensor();

  ASSERT_TRUE(stack->at(3).isTensor());
  auto t2 = stack->at(3).toTensor();

  t1.add_(s1);
  t2.add_(s2);

  stack->clear();
  torch::jit::push(stack, t1);
  torch::jit::push(stack, t2);
}

//
// boxed calling tests:
//

// functional

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

void expectBoxedCallingWithMultiReturnWorks(const KernelFunction& func) {
  called_with_args = c10::nullopt;
  vector<IValue> stack {3, 4};
  OperatorHandle dummy = makeDummyOperatorHandle();

  func.callBoxed(dummy, &stack);

  EXPECT_TRUE(called_with_args.has_value());
  EXPECT_EQ((tuple<int64_t, int64_t>(3, 4)), *called_with_args);
  EXPECT_EQ(2, stack.size());

  EXPECT_TRUE(stack[0].isInt());
  EXPECT_EQ(7, stack[0].toInt());

  EXPECT_TRUE(stack[1].isInt());
  EXPECT_EQ(12, stack[1].toInt());
}

// in/out

void expectInPlaceBoxedCallingWorks(const KernelFunction& func) {
  OperatorHandle dummy = makeDummyOperatorHandle();

  auto t = at::zeros({1});
  auto s = 1.0f;
  vector<IValue> stack {t, s};
  func.callBoxed(dummy, &stack);

  // kernel should have updated out arg and returned it
  EXPECT_EQ(t.item().toFloat(), 1.0f);
  EXPECT_EQ(1, stack.size());
  EXPECT_TRUE(stack[0].isTensor());
  EXPECT_TRUE(stack[0].toTensor().is_same(t));
}

void expectOutOfPlaceBoxedCallingWorks(const KernelFunction& func) {
  OperatorHandle dummy = makeDummyOperatorHandle();

  auto s = 1.0f;
  auto t = at::zeros({1});
  vector<IValue> stack {s, t};
  func.callBoxed(dummy, &stack);

  // kernel should have updated out arg and returned it on the stack
  EXPECT_EQ(t.item().toFloat(), 1.0f);
  EXPECT_EQ(1, stack.size());
  EXPECT_TRUE(stack[0].isTensor());
  EXPECT_TRUE(stack[0].toTensor().is_same(t));
}

void expectOutOfPlaceMultiBoxedCallingWorks(const KernelFunction& func) {
  OperatorHandle dummy = makeDummyOperatorHandle();

  auto t1 = at::zeros({1});
  auto t2 = at::zeros({1});
  auto s1 = 1.0f;
  auto s2 = 2.0f;
  vector<IValue> stack {t1, t2, s1, s2};
  func.callBoxed(dummy, &stack);

  // kernel should have updated output args and returned them on the stack
  EXPECT_EQ(t1.item().toFloat(), 1.0f);
  EXPECT_EQ(t2.item().toFloat(), 2.0f);
  EXPECT_EQ(2, stack.size());
  EXPECT_TRUE(stack[0].isTensor());
  EXPECT_TRUE(stack[0].toTensor().is_same(t1));
  EXPECT_TRUE(stack[1].isTensor());
  EXPECT_TRUE(stack[1].toTensor().is_same(t2));
}

void expectLegacyOutOfPlaceMultiBoxedCallingWorks(const KernelFunction& func) {
  OperatorHandle dummy = makeDummyOperatorHandle();

  auto s1 = 1.0f;
  auto s2 = 2.0f;
  auto t1 = at::zeros({1});
  auto t2 = at::zeros({1});
  vector<IValue> stack {s1, s2, t1, t2};
  func.callBoxed(dummy, &stack);

  // kernel should have updated output args and returned them on the stack
  EXPECT_EQ(t1.item().toFloat(), 1.0f);
  EXPECT_EQ(t2.item().toFloat(), 2.0f);
  EXPECT_EQ(2, stack.size());
  EXPECT_TRUE(stack[0].isTensor());
  EXPECT_TRUE(stack[0].toTensor().is_same(t1));
  EXPECT_TRUE(stack[1].isTensor());
  EXPECT_TRUE(stack[1].toTensor().is_same(t2));
}

void expectBoxedCallingFailsWith(const KernelFunction& func, const char* errorMessage) {
  called_with_args = c10::nullopt;
  vector<IValue> stack {3, 4};
  OperatorHandle dummy = makeDummyOperatorHandle();

  expectThrows<c10::Error>([&] {
    func.callBoxed(dummy, &stack);
  }, errorMessage);
}

//
// unboxed calling tests:
//

// functional

// make an unboxed call to a kernel that returns a single value.
//
void expectUnboxedCallingWithReturnWorks(const KernelFunction& func) {
  called_with_args = c10::nullopt;
  OperatorHandle dummy = makeDummyOperatorHandle();

  int64_t result = func.call<int64_t, int64_t, int64_t>(dummy, 3, 4);

  EXPECT_TRUE(called_with_args.has_value());
  EXPECT_EQ((tuple<int64_t, int64_t>(3, 4)), *called_with_args);
  EXPECT_EQ(5, result);
}

// make an unboxed call to a kernel that returns nothing.
//
void expectUnboxedCallingWithoutReturnWorks(const KernelFunction& func) {
  called_with_args = c10::nullopt;
  OperatorHandle dummy = makeDummyOperatorHandle();

  func.call<void, int64_t, int64_t>(dummy, 3, 4);

  EXPECT_TRUE(called_with_args.has_value());
  EXPECT_EQ((tuple<int64_t, int64_t>(3, 4)), *called_with_args);
}

// make an unboxed call to a kernel that returns two values.
// When calling unboxed, multiple values are returned as a tuple.
//
void expectUnboxedCallingWithMultiReturnWorks(const KernelFunction& func) {
  called_with_args = c10::nullopt;
  OperatorHandle dummy = makeDummyOperatorHandle();

  auto result = func.call<std::tuple<int64_t, int64_t>, int64_t, int64_t>(dummy, 3, 4);

  EXPECT_TRUE(called_with_args.has_value());
  EXPECT_EQ((tuple<int64_t, int64_t>(3, 4)), *called_with_args);

  EXPECT_EQ((tuple<int64_t, int64_t>(7, 12)), result);
}

// in/out

void expectInPlaceUnboxedCallingWorks(const KernelFunction& func) {
  OperatorHandle dummy = makeDummyOperatorHandle();

  auto t = at::zeros({1});
  at::Tensor& t_out = func.call<at::Tensor&, at::Tensor&, at::Scalar>(dummy, t, 1.0f);

  // should have updated first arg and returned it
  EXPECT_EQ(t.item().toFloat(), 1.0f);
  EXPECT_EQ(&t, &t_out);
}

void expectOutOfPlaceUnboxedCallingWorks(const KernelFunction& func) {
  OperatorHandle dummy = makeDummyOperatorHandle();

  auto t = at::zeros({1});
  at::Tensor& t_out = func.call<at::Tensor&, at::Scalar, at::Tensor&>(dummy, 1.0f, t);

  // should have updated out arg and returned it
  EXPECT_EQ(t.item().toFloat(), 1.0f);
  EXPECT_EQ(&t, &t_out);
}

void expectOutOfPlaceMultiUnboxedCallingWorks(const KernelFunction& func) {
  OperatorHandle dummy = makeDummyOperatorHandle();

  auto t1 = at::zeros({1});
  auto t2 = at::zeros({1});
  auto s1 = 1.0f;
  auto s2 = 2.0f;

  std::tuple<at::Tensor&, at::Tensor&> tup = func.call<
    std::tuple<at::Tensor&, at::Tensor&>, at::Tensor&, at::Tensor&, at::Scalar, at::Scalar
  >(dummy, t1, t2, s1, s2);

  // kernel should have updated out args and returned them in a tuple
  EXPECT_EQ(t1.item().toFloat(), 1.0f);
  EXPECT_EQ(t2.item().toFloat(), 2.0f);

  auto t1_out = std::get<0>(tup);
  EXPECT_EQ(t1_out.item().toFloat(), 1.0f);
  EXPECT_TRUE(t1_out.is_same(t1));

  auto t2_out = std::get<1>(tup);
  EXPECT_EQ(t2_out.item().toFloat(), 2.0f);
  EXPECT_TRUE(t2_out.is_same(t2));
}

void expectLegacyOutOfPlaceMultiUnboxedCallingWorks(const KernelFunction& func) {
  OperatorHandle dummy = makeDummyOperatorHandle();

  auto s1 = 1.0f;
  auto s2 = 2.0f;
  auto t1 = at::zeros({1});
  auto t2 = at::zeros({1});

  std::tuple<at::Tensor&, at::Tensor&> tup = func.call<
    std::tuple<at::Tensor&, at::Tensor&>, at::Scalar, at::Scalar, at::Tensor&, at::Tensor&
  >(dummy, s1, s2, t1, t2);

  // kernel should have updated out args and returned them in a tuple
  EXPECT_EQ(t1.item().toFloat(), 1.0f);
  EXPECT_EQ(t2.item().toFloat(), 2.0f);

  auto t1_out = std::get<0>(tup);
  EXPECT_EQ(t1_out.item().toFloat(), 1.0f);
  EXPECT_TRUE(t1_out.is_same(t1));

  auto t2_out = std::get<1>(tup);
  EXPECT_EQ(t2_out.item().toFloat(), 2.0f);
  EXPECT_TRUE(t2_out.is_same(t2));
}

}

// functional, boxed calling

TEST(KernelFunctionTest, givenBoxedFunction_withReturn_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_with_return>();
  kernels::expectBoxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenBoxedFunction_withoutReturn_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_without_return>();
  kernels::expectBoxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenBoxedFunction_withMultiReturn_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_with_multi_return>();
  kernels::expectBoxedCallingWithMultiReturnWorks(func);
}

// in/out, boxed calling

TEST(KernelFunctionTest, givenBoxedFunction_withInPlaceSignature_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_for_inplace_op>();
  kernels::expectInPlaceBoxedCallingWorks(func);
}

TEST(KernelFunctionTest, givenBoxedFunction_withOutOfPlaceSignature_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_for_outofplace_op>();
  kernels::expectOutOfPlaceBoxedCallingWorks(func);
}

TEST(KernelFunctionTest, givenBoxedFunction_withOutOfPlaceMultiSignature_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_for_outofplace_multi_op>();
  kernels::expectOutOfPlaceMultiBoxedCallingWorks(func);
}

TEST(KernelFunctionTest, givenBoxedFunction_withLegacyOutOfPlaceMultiSignature_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_for_legacy_outofplace_multi_op>();
  kernels::expectLegacyOutOfPlaceMultiBoxedCallingWorks(func);
}

// functional, unboxed calling

TEST(KernelFunctionTest, givenBoxedFunction_withReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_with_return>();
  kernels::expectUnboxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenBoxedFunction_withoutReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_without_return>();
  kernels::expectUnboxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenBoxedFunction_withMultiReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_with_multi_return>();
  kernels::expectUnboxedCallingWithMultiReturnWorks(func);
}

// in/out, unboxed calling

TEST(KernelFunctionTest, givenBoxedFunction_withInPlaceSignature_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_for_inplace_op>();
  kernels::expectInPlaceUnboxedCallingWorks(func);
}

TEST(KernelFunctionTest, givenBoxedFunction_withOutOfPlaceSignature_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_for_outofplace_op>();
  kernels::expectOutOfPlaceUnboxedCallingWorks(func);
}

TEST(KernelFunctionTest, givenBoxedFunction_withOutOfPlaceMultiSignature_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_for_outofplace_multi_op>();
  kernels::expectOutOfPlaceMultiUnboxedCallingWorks(func);
}

TEST(KernelFunctionTest, givenBoxedFunction_withLegacyOutOfPlaceMultiSignature_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_for_legacy_outofplace_multi_op>();
  kernels::expectLegacyOutOfPlaceMultiUnboxedCallingWorks(func);
}

// functors etc.

TEST(KernelFunctionTest, givenUnboxedFunctor_withReturn_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedFunctor<false, kernels::unboxed_functor_with_return>(std::unique_ptr<OperatorKernel>(std::make_unique<kernels::unboxed_functor_with_return>()));
  kernels::expectBoxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunctor_withoutReturn_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedFunctor<false, kernels::unboxed_functor_without_return>(std::unique_ptr<OperatorKernel>(std::make_unique<kernels::unboxed_functor_without_return>()));
  kernels::expectBoxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunctor_withReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedFunctor<false, kernels::unboxed_functor_with_return>(std::unique_ptr<OperatorKernel>(std::make_unique<kernels::unboxed_functor_with_return>()));
  kernels::expectUnboxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunctor_withoutReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedFunctor<false, kernels::unboxed_functor_without_return>(std::unique_ptr<OperatorKernel>(std::make_unique<kernels::unboxed_functor_without_return>()));
  kernels::expectUnboxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedOnlyFunctor_withReturn_whenCallingBoxed_thenFails) {
  KernelFunction func = KernelFunction::makeFromUnboxedOnlyFunctor<kernels::unboxed_functor_with_return>(std::unique_ptr<OperatorKernel>(std::make_unique<kernels::unboxed_functor_with_return>()));
  kernels::expectBoxedCallingFailsWith(func, "Tried to call KernelFunction::callBoxed() on a KernelFunction that can only be called with KernelFunction::call()");
}

TEST(KernelFunctionTest, givenUnboxedOnlyFunctor_withoutReturn_whenCallingBoxed_thenFails) {
  KernelFunction func = KernelFunction::makeFromUnboxedOnlyFunctor<kernels::unboxed_functor_without_return>(std::unique_ptr<OperatorKernel>(std::make_unique<kernels::unboxed_functor_without_return>()));
  kernels::expectBoxedCallingFailsWith(func, "Tried to call KernelFunction::callBoxed() on a KernelFunction that can only be called with KernelFunction::call()");
}

TEST(KernelFunctionTest, givenUnboxedOnlyFunctor_withReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedOnlyFunctor<kernels::unboxed_functor_with_return>(std::unique_ptr<OperatorKernel>(std::make_unique<kernels::unboxed_functor_with_return>()));
  kernels::expectUnboxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedOnlyFunctor_withoutReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedOnlyFunctor<kernels::unboxed_functor_without_return>(std::unique_ptr<OperatorKernel>(std::make_unique<kernels::unboxed_functor_without_return>()));
  kernels::expectUnboxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunction_withReturn_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedFunction(TORCH_FN(kernels::unboxed_function_with_return));
  kernels::expectBoxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunction_withoutReturn_whenCallingBoxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedFunction(TORCH_FN(kernels::unboxed_function_without_return));
  kernels::expectBoxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunction_withReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedFunction(TORCH_FN(kernels::unboxed_function_with_return));
  kernels::expectUnboxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunction_withoutReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedFunction(TORCH_FN(kernels::unboxed_function_without_return));
  kernels::expectUnboxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedOnlyFunction_withReturn_whenCallingBoxed_thenFails) {
  KernelFunction func = KernelFunction::makeFromUnboxedOnlyFunction(TORCH_FN(kernels::unboxed_function_with_return));
  kernels::expectBoxedCallingFailsWith(func, "Tried to call KernelFunction::callBoxed() on a KernelFunction that can only be called with KernelFunction::call()");
}

TEST(KernelFunctionTest, givenUnboxedOnlyFunction_withoutReturn_whenCallingBoxed_thenFails) {
  KernelFunction func = KernelFunction::makeFromUnboxedOnlyFunction(TORCH_FN(kernels::unboxed_function_without_return));
  kernels::expectBoxedCallingFailsWith(func, "Tried to call KernelFunction::callBoxed() on a KernelFunction that can only be called with KernelFunction::call()");
}

TEST(KernelFunctionTest, givenUnboxedOnlyFunction_withReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedOnlyFunction(TORCH_FN(kernels::unboxed_function_with_return));
  kernels::expectUnboxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedOnlyFunction_withoutReturn_whenCallingUnboxed_thenWorks) {
  KernelFunction func = KernelFunction::makeFromUnboxedOnlyFunction(TORCH_FN(kernels::unboxed_function_without_return));
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
