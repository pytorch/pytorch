
#include <gtest/gtest.h>
#include <ATen/core/boxing/impl/test_helpers.h>

#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/Tensor.h>
#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <torch/library.h>

#include <ATen/core/LegacyTypeDispatch.h>

using c10::RegisterOperators;
using c10::DispatchKey;
using c10::Stack;
using std::make_unique;
using c10::OperatorHandle;
using std::unique_ptr;

namespace {

void errorKernel(const OperatorHandle&, Stack* stack) {
  EXPECT_TRUE(false); // this kernel should never be called
}

void incrementKernel(const OperatorHandle&, Stack* stack) {
  int input = torch::jit::pop(*stack).toInt();
  torch::jit::pop(*stack); // pop the dummy tensor
  torch::jit::push(*stack, input + 1);
}

void decrementKernel(const OperatorHandle&, Stack* stack) {
  int input = torch::jit::pop(*stack).toInt();
  torch::jit::pop(*stack); // pop the dummy tensor
  torch::jit::push(*stack, input - 1);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool called_redispatching_kernel = false;
void redispatchingKernel_with_DispatchKeySet(const OperatorHandle& op, c10::DispatchKeySet ks, Stack* stack) {
  // this kernel is a no-op- it just redispatches to the lower-priority kernel
  called_redispatching_kernel = true;
  auto updated_ks = ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::TESTING_ONLY_GenericWrapper);
  op.redispatchBoxed(updated_ks, stack);
}

void expectCallsIncrement(c10::DispatchKeySet ks) {
  at::AutoDispatchBelowAutograd mode;

  // assert that schema and cpu kernel are present
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  auto result = callOp(*op, dummyTensor(ks), 5);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(6, result[0].toInt());
}

void expectCallsIncrement(DispatchKey dispatch_key) {
  expectCallsIncrement(c10::DispatchKeySet(dispatch_key));
}

void expectCallsIncrementUnboxed(DispatchKey dispatch_key) {
  at::AutoDispatchBelowAutograd mode;

  // assert that schema and cpu kernel are present
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  int64_t result = callOpUnboxed<int64_t, at::Tensor, int64_t>(*op, dummyTensor(dispatch_key), 5);
  EXPECT_EQ(6, result);
}

void expectCallsDecrement(DispatchKey dispatch_key) {
  at::AutoDispatchBelowAutograd mode;

  // assert that schema and cpu kernel are present
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  auto result = callOp(*op, dummyTensor(dispatch_key), 5);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(4, result[0].toInt());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_StackBasedKernel, givenKernel_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<&incrementKernel>(DispatchKey::CPU));
  expectCallsIncrement(DispatchKey::CPU);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_StackBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInOneRegistrar_thenCallsRightKernel) {
  auto registrar = RegisterOperators()
      .op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<&incrementKernel>(DispatchKey::CPU)
                                                                                      .kernel<&errorKernel>(DispatchKey::CUDA))
      .op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<&errorKernel>(DispatchKey::CPU)
                                                                                      .kernel<&errorKernel>(DispatchKey::CUDA));
  expectCallsIncrement(DispatchKey::CPU);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_StackBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInMultipleRegistrars_thenCallsRightKernel) {
  auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<&incrementKernel>(DispatchKey::CPU)
                                                                                                                       .kernel<&errorKernel>(DispatchKey::CUDA));
  auto registrar2 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<&errorKernel>(DispatchKey::CPU)
                                                                                                                       .kernel<&errorKernel>(DispatchKey::CUDA));
  expectCallsIncrement(DispatchKey::CPU);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_StackBasedKernel, givenKernel_whenRegistrationRunsOutOfScope_thenCannotBeCalledAnymore) {
  {
    auto m = MAKE_TORCH_LIBRARY(_test);
    m.def("_test::my_op(Tensor dummy, int input) -> int");
    auto m_cpu = MAKE_TORCH_LIBRARY_IMPL(_test, CPU);
    m_cpu.impl("my_op", DispatchKey::CPU, torch::CppFunction::makeFromBoxedFunction<incrementKernel>());
    {
      auto m_cuda = MAKE_TORCH_LIBRARY_IMPL(_test, CUDA);
      m_cuda.impl("my_op", DispatchKey::CUDA, torch::CppFunction::makeFromBoxedFunction<decrementKernel>());

      // assert that schema and cpu kernel are present
      expectCallsIncrement(DispatchKey::CPU);
      expectCallsDecrement(DispatchKey::CUDA);
    }

    // now registrar2 is destructed. Assert that schema is still present but cpu kernel is not
    expectCallsIncrement(DispatchKey::CPU);
    expectDoesntFindKernel("_test::my_op", DispatchKey::CUDA);
  }

  // now both registrars are destructed. Assert that the whole schema is gone
  expectDoesntFindOperator("_test::my_op");
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool called = false;

void kernelWithoutInputs(const OperatorHandle&, Stack*) {
  called = true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_StackBasedKernel, givenFallbackKernelWithoutAnyArguments_whenRegistered_thenCanBeCalled) {
  // note: non-fallback kernels without tensor arguments don't work because there
  // is no way to get the dispatch key. For operators that only have a fallback
  // kernel, this must work for backwards compatibility.
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args() -> ()", RegisterOperators::options().catchAllKernel<&kernelWithoutInputs>());

  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
  ASSERT_TRUE(op.has_value());

  called = false;
  auto outputs = callOp(*op);
  EXPECT_TRUE(called);
}

void kernelWithoutTensorInputs(const OperatorHandle&, Stack* stack) {
  stack->back() = stack->back().toInt() + 1;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_StackBasedKernel, givenFallbackKernelWithoutTensorArguments_whenRegistered_thenCanBeCalled) {
  // note: non-fallback kernels without tensor arguments don't work because there
  // is no way to get the dispatch key. For operators that only have a fallback
  // kernel, this must work for backwards compatibility.
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args(int arg) -> int", RegisterOperators::options().catchAllKernel<&kernelWithoutTensorInputs>());

  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

void kernelForSchemaInference(const OperatorHandle&, Stack* stack) {
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_StackBasedKernel, givenKernel_whenRegisteredWithoutSpecifyingSchema_thenFailsBecauseItCannotInferFromStackBasedKernel) {
  expectThrows<c10::Error>([] {
      RegisterOperators().op("_test::no_schema_specified", RegisterOperators::options().catchAllKernel<&kernelForSchemaInference>());
  }, "Cannot infer operator schema for this kind of kernel in registration of operator _test::no_schema_specified");
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_StackBasedKernel, givenKernel_whenRegistered_thenCanAlsoBeCalledUnboxed) {
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<&incrementKernel>(DispatchKey::CPU));
  expectCallsIncrementUnboxed(DispatchKey::CPU);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorRegistrationTest_StackBasedKernel, callKernelsWithDispatchKeySetConvention_redispatchesToLowerPriorityKernels) {
  auto m = MAKE_TORCH_LIBRARY(_test);
  m.def("my_op(Tensor dummy, int input) -> int");
  auto m_cpu = MAKE_TORCH_LIBRARY_IMPL(_, CPU);
  m_cpu.fallback(torch::CppFunction::makeFromBoxedFunction<&incrementKernel>());
  auto m_testing = MAKE_TORCH_LIBRARY_IMPL(_, TESTING_ONLY_GenericWrapper);
  m_testing.fallback(torch::CppFunction::makeFromBoxedFunction<&redispatchingKernel_with_DispatchKeySet>());

  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());

  auto testing_cpu_set = c10::DispatchKeySet()
                                    .add(c10::DispatchKey::TESTING_ONLY_GenericWrapper)
                                    .add(c10::DispatchKey::CPU);
  called_redispatching_kernel = false;

  // call CPU (and not TESTING_ONLY_GenericWrapper)
  expectCallsIncrement(DispatchKey::CPU);
  ASSERT_FALSE(called_redispatching_kernel);

  // call TESTING_ONLY_GenericWrapper -> call CPU
  expectCallsIncrement(testing_cpu_set);
  ASSERT_TRUE(called_redispatching_kernel);
}

}
