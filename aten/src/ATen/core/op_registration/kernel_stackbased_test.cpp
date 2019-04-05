#include <gtest/gtest.h>
#include <ATen/core/op_registration/test_helpers.h>

#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/Tensor.h>

using c10::RegisterOperators;
using c10::kernel;
using c10::dispatchKey;
using c10::TensorTypeId;
using c10::KernelCache;
using c10::Stack;
using c10::guts::make_unique;
using std::unique_ptr;

namespace {

C10_DECLARE_TENSOR_TYPE(TensorType1);
C10_DEFINE_TENSOR_TYPE(TensorType1);
C10_DECLARE_TENSOR_TYPE(TensorType2);
C10_DEFINE_TENSOR_TYPE(TensorType2);

std::unique_ptr<c10::KernelCache> noCache() {
  return nullptr;
}

void errorKernel(Stack* stack, KernelCache* cache) {
  EXPECT_TRUE(false); // this kernel should never be called
}

void incrementKernel(Stack* stack, KernelCache* cache) {
  int input = torch::jit::pop(*stack).toInt();
  torch::jit::pop(*stack); // pop the dummy tensor
  torch::jit::push(*stack, input + 1);
}

void decrementKernel(Stack* stack, KernelCache* cache) {
  int input = torch::jit::pop(*stack).toInt();
  torch::jit::pop(*stack); // pop the dummy tensor
  torch::jit::push(*stack, input - 1);
}

void expectCallsIncrement(TensorTypeId type_id) {
  // assert that schema and cpu kernel are present
  auto op = c10::Dispatcher::singleton().findSchema("_test::my_op", "");
  ASSERT_TRUE(op.has_value());
  auto result = callOp(*op, dummyTensor(type_id), 5);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(6, result[0].toInt());
}

void expectCallsDecrement(TensorTypeId type_id) {
  // assert that schema and cpu kernel are present
  auto op = c10::Dispatcher::singleton().findSchema("_test::my_op", "");
  ASSERT_TRUE(op.has_value());
  auto result = callOp(*op, dummyTensor(type_id), 5);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(4, result[0].toInt());
}

TEST(OperatorRegistrationTest_StackBasedKernel, givenKernel_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", kernel(&incrementKernel, &noCache), dispatchKey(TensorType1()));
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_StackBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInOneRegistrar_thenCallsRightKernel) {
  auto registrar = RegisterOperators()
      .op("_test::my_op(Tensor dummy, int input) -> int", kernel(&incrementKernel, &noCache), dispatchKey(TensorType1()))
      .op("_test::my_op(Tensor dummy, int input) -> int", kernel(&errorKernel, &noCache), dispatchKey(TensorType2()))
      .op("_test::error(Tensor dummy, int input) -> int", kernel(&errorKernel, &noCache), dispatchKey(TensorType1()))
      .op("_test::error(Tensor dummy, int input) -> int", kernel(&errorKernel, &noCache), dispatchKey(TensorType2()));
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_StackBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInMultipleRegistrars_thenCallsRightKernel) {
  auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", kernel(&incrementKernel, &noCache), dispatchKey(TensorType1()));
  auto registrar2 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", kernel(&errorKernel, &noCache), dispatchKey(TensorType2()));
  auto registrar3 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int", kernel(&errorKernel, &noCache), dispatchKey(TensorType1()));
  auto registrar4 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int", kernel(&errorKernel, &noCache), dispatchKey(TensorType2()));
  expectCallsIncrement(TensorType1());
}

TEST(OperatorRegistrationTest_StackBasedKernel, givenKernel_whenRegistrationRunsOutOfScope_thenCannotBeCalledAnymore) {
  {
    auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", kernel(&incrementKernel, &noCache), dispatchKey(TensorType1()));
    {
      auto registrar2 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", kernel(&decrementKernel, &noCache), dispatchKey(TensorType2()));

      // assert that schema and cpu kernel are present
      expectCallsIncrement(TensorType1());
      expectCallsDecrement(TensorType2());
    }

    // now registrar2 is destructed. Assert that schema is still present but cpu kernel is not
    expectCallsIncrement(TensorType1());
    expectDoesntFindKernel("_test::my_op", TensorType2());
  }

  // now both registrars are destructed. Assert that the whole schema is gone
  expectDoesntFindOperator("_test::my_op");
}

struct Cache final : KernelCache {
  int last_value = 4;
};

unique_ptr<KernelCache> make_cache() {
  return make_unique<Cache>();
}

void increment_sequence_kernel(Stack* stack, KernelCache* cache) {
  torch::jit::pop(*stack); // pop dummy tensor
  EXPECT_EQ(0, stack->size());
  torch::jit::push(*stack, static_cast<Cache*>(cache)->last_value++);
}

TEST(OperatorRegistrationTest_StackBasedKernel, givenKernelWithCache_whenCalled_thenCacheIsHandledCorrectly) {
  auto registrar = RegisterOperators().op("_test::increment_sequence(Tensor dummy) -> int", kernel(&increment_sequence_kernel, &make_cache), dispatchKey(TensorType1()));

  auto op = c10::Dispatcher::singleton().findSchema("_test::increment_sequence", "");
  ASSERT_TRUE(op.has_value());

  // expect first time calling returns a 4 (4 is the initial value in the cache)
  auto stack = makeStack(dummyTensor(TensorType1()));
  auto kernel = c10::Dispatcher::singleton().lookup(*op, &stack);
  kernel.call(&stack);
  EXPECT_EQ(1, stack.size());
  EXPECT_EQ(4, stack[0].toInt());

  // expect second time calling returns a 5
  stack = makeStack(dummyTensor(TensorType1()));
  kernel.call(&stack);
  EXPECT_EQ(1, stack.size());
  EXPECT_EQ(5, stack[0].toInt());

  // expect third time calling returns a 6
  stack = makeStack(dummyTensor(TensorType1()));
  kernel.call(&stack);
  EXPECT_EQ(1, stack.size());
  EXPECT_EQ(6, stack[0].toInt());
}

}
