
#include <gtest/gtest.h>
#include <ATen/core/op_registration/test_helpers.h>

#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/Tensor.h>
#include <torch/csrc/jit/script/function_schema_parser.h>

using c10::RegisterOperators;
using c10::TensorTypeId;
using c10::KernelCache;
using c10::Stack;
using c10::guts::make_unique;
using std::unique_ptr;

namespace {

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
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  auto result = callOp(*op, dummyTensor(type_id), 5);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(6, result[0].toInt());
}

void expectCallsDecrement(TensorTypeId type_id) {
  // assert that schema and cpu kernel are present
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  auto result = callOp(*op, dummyTensor(type_id), 5);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(4, result[0].toInt());
}

TEST(OperatorRegistrationTest_StackBasedKernel, givenKernel_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(TensorTypeId::CPUTensorId, &incrementKernel));
  expectCallsIncrement(TensorTypeId::CPUTensorId);
}

TEST(OperatorRegistrationTest_StackBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInOneRegistrar_thenCallsRightKernel) {
  auto registrar = RegisterOperators()
      .op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(TensorTypeId::CPUTensorId, &incrementKernel))
      .op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(TensorTypeId::CUDATensorId, &errorKernel))
      .op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(TensorTypeId::CPUTensorId, &errorKernel))
      .op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(TensorTypeId::CUDATensorId, &errorKernel));
  expectCallsIncrement(TensorTypeId::CPUTensorId);
}

TEST(OperatorRegistrationTest_StackBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInMultipleRegistrars_thenCallsRightKernel) {
  auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(TensorTypeId::CPUTensorId, &incrementKernel));
  auto registrar2 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(TensorTypeId::CUDATensorId, &errorKernel));
  auto registrar3 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(TensorTypeId::CPUTensorId, &errorKernel));
  auto registrar4 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(TensorTypeId::CUDATensorId, &errorKernel));
  expectCallsIncrement(TensorTypeId::CPUTensorId);
}

TEST(OperatorRegistrationTest_StackBasedKernel, givenKernel_whenRegistrationRunsOutOfScope_thenCannotBeCalledAnymore) {
  {
    auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(TensorTypeId::CPUTensorId, &incrementKernel));
    {
      auto registrar2 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(TensorTypeId::CUDATensorId, &decrementKernel));

      // assert that schema and cpu kernel are present
      expectCallsIncrement(TensorTypeId::CPUTensorId);
      expectCallsDecrement(TensorTypeId::CUDATensorId);
    }

    // now registrar2 is destructed. Assert that schema is still present but cpu kernel is not
    expectCallsIncrement(TensorTypeId::CPUTensorId);
    expectDoesntFindKernel("_test::my_op", TensorTypeId::CUDATensorId);
  }

  // now both registrars are destructed. Assert that the whole schema is gone
  expectDoesntFindOperator("_test::my_op");
}

bool called = false;

void kernelWithoutInputs(Stack*, KernelCache*) {
  called = true;
}

TEST(OperatorRegistrationTest_StackBasedKernel, givenFallbackKernelWithoutAnyArguments_whenRegistered_thenCanBeCalled) {
  // note: non-fallback kernels without tensor arguments don't work because there
  // is no way to get the dispatch key. For operators that only have a fallback
  // kernel, this must work for backwards compatibility.
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args() -> ()", RegisterOperators::options().catchAllKernel(&kernelWithoutInputs));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
  ASSERT_TRUE(op.has_value());

  called = false;
  auto outputs = callOp(*op);
  EXPECT_TRUE(called);
}

void kernelWithoutTensorInputs(Stack* stack, KernelCache*) {
  stack->back() = stack->back().toInt() + 1;
}

TEST(OperatorRegistrationTest_StackBasedKernel, givenFallbackKernelWithoutTensorArguments_whenRegistered_thenCanBeCalled) {
  // note: non-fallback kernels without tensor arguments don't work because there
  // is no way to get the dispatch key. For operators that only have a fallback
  // kernel, this must work for backwards compatibility.
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args(int arg) -> int", RegisterOperators::options().catchAllKernel(&kernelWithoutTensorInputs));

  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
  ASSERT_TRUE(op.has_value());

  auto outputs = callOp(*op, 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

void kernelForSchemaInference(Stack* stack, KernelCache* cache) {
}

TEST(OperatorRegistrationTest_StackBasedKernel, givenKernel_whenRegisteredWithoutSpecifyingSchema_thenFailsBecauseItCannotInferFromStackBasedKernel) {
  expectThrows<c10::Error>([] {
      RegisterOperators().op("_test::no_schema_specified", RegisterOperators::options().catchAllKernel(&kernelForSchemaInference));
  }, "Cannot infer operator schema for this kind of kernel in registration of operator _test::no_schema_specified");
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

TEST(OperatorRegistrationTest_StackBasedKernel, givenKernel_whenRegistered_thenCannotBeCalledUnboxed) {
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(TensorTypeId::CPUTensorId, &incrementKernel));
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  expectThrows<c10::Error>(
    [&] {callOpUnboxed<int64_t, at::Tensor, int64_t>(*op, TensorTypeId::CPUTensorId, dummyTensor(TensorTypeId::CPUTensorId), 4);},
    "Tried to call KernelFunction::callUnboxed() for a kernel that doesn't have an unboxed version. This isn't implemented yet."
  );
}

}
